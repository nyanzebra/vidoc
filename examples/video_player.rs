//! Video Player with GOP Compression
//!
//! This example reads a Y4M video file, compresses it using GOP encoding,
//! and plays it back in a window.
//!
//! # Usage
//!
//! ```bash
//! # Basic usage - processes entire video by default
//! cargo run --example video_player --release video.y4m
//!
//! # Process only first 1000 frames
//! cargo run --example video_player --release video.y4m --max-frames 1000
//!
//! # Custom GOP structure (smaller GOPs for better quality)
//! cargo run --example video_player --release video.y4m -a 3 -i 15
//!
//! # Save compressed output
//! cargo run --example video_player --release video.y4m -o compressed.vidoc
//!
//! # Enable verbose decoding output
//! cargo run --example video_player --release video.y4m --verbose
//!
//! # Skip fewer frames and limit to 1000
//! cargo run --example video_player --release video.y4m -s 0 -m 1000
//! ```
//!
//! Run `cargo run --example video_player -- --help` for all options.

use std::{
    cell::RefCell,
    fs::File,
    io::{Cursor, Write},
    path::PathBuf,
    sync::mpsc,
    thread,
    time::Instant,
};

use clap::Parser;
use minifb::{Key, Window, WindowOptions};
use rayon::prelude::*;
use vidoc::{
    bitstream::BitStreamWriter,
    block::Block,
    color::Subsampling,
    dimensions::{BlockDimensions, PixelDimensions},
    error,
    lossy::{
        frame::gop::{
            DecodedFrame, FrameReader, GroupOfPicturesReader, GroupOfPicturesWriter, Ordering,
        },
        SubSampleBlockGroup, SubSampleBlockGroupRef,
    },
};

/// Video Player with GOP Compression
///
/// Reads a Y4M video file, compresses it using GOP encoding, and plays it back in a window.
#[derive(Parser, Debug)]
#[command(name = "vidoc-player")]
#[command(author, version)]
#[command(about = "Video Player with GOP Compression - demonstrates I/P/B frame encoding")]
#[command(
    long_about = "Reads a Y4M video file, compresses it using Group of Pictures (GOP) encoding \
                        with I-frames (keyframes), P-frames (predicted), and B-frames (bidirectional), \
                        then plays it back in a window. Supports configurable GOP structure and frame skipping."
)]
struct Args {
    /// Path to the Y4M video file
    #[arg(value_name = "VIDEO_FILE")]
    video_path: PathBuf,

    /// Maximum number of frames to process (default: all frames)
    #[arg(short, long, default_value_t = usize::MAX)]
    max_frames: usize,

    /// P-frame interval (number of frames between anchor frames)
    #[arg(short, long, default_value_t = 5)]
    anchor_distance: usize,

    /// I-frame interval (number of frames between full image refreshes)
    #[arg(short = 'i', long, default_value_t = 30)]
    full_image_distance: usize,

    /// Number of frames to skip at the beginning
    #[arg(short, long, default_value_t = 100)]
    skip_frames: usize,

    /// Save compressed video to file
    #[arg(short = 'o', long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🎬 Vidoc Video Player - GOP Compression Demo");
    println!("==========================================\n");

    play_y4m_video(
        &args.video_path,
        args.max_frames,
        args.anchor_distance,
        args.full_image_distance,
        args.skip_frames,
        args.output.as_ref(),
        args.verbose,
    )?;

    Ok(())
}

fn play_y4m_video(
    path: &PathBuf,
    max_frames: usize,
    anchor_distance: usize,
    full_image_distance: usize,
    skip_frames: usize,
    output_path: Option<&PathBuf>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📂 Loading Y4M video: {:?}\n", path);

    let file = File::open(path)?;
    let mut decoder = y4m::Decoder::new(file)?;

    let width = decoder.get_width();
    let height = decoder.get_height();
    let framerate = decoder.get_framerate();

    println!("📹 Video information:");
    println!("   Resolution: {}x{} pixels", width, height);
    println!(
        "   Framerate: {}/{} fps ({:.2} fps)",
        framerate.num,
        framerate.den,
        framerate.num as f64 / framerate.den as f64
    );

    println!(
        "   GOP: anchor={}, full_image={} (I/P/B frames)",
        anchor_distance, full_image_distance
    );

    let frame_limit_msg = if max_frames == usize::MAX {
        "full video".to_string()
    } else {
        format!("max {} frames", max_frames)
    };

    println!(
        "\n⏳ Loading frames (skipping first {} grayscale frames, {})...",
        skip_frames, frame_limit_msg
    );
    let start = Instant::now();
    let mut frames = Vec::new();
    let mut skipped = 0;

    while let Ok(frame) = decoder.read_frame() {
        if skipped < skip_frames {
            skipped += 1;
            continue;
        }

        if max_frames > 0 && frames.len() >= max_frames {
            break;
        }

        let subsample_frame = y4m_frame_to_subsample(frame, width, height, frames.len())?;
        frames.push(subsample_frame);

        if frames.len() % 30 == 0 {
            print!("   Loaded {} frames...\r", frames.len());
            std::io::stdout().flush()?;
        }
    }

    println!(
        "   ✓ Loaded {} frames in {:.2}s",
        frames.len(),
        start.elapsed().as_secs_f64()
    );

    // Compress with GOP
    println!("\n📦 Compressing with GOP...");
    let start = Instant::now();
    let compressed = encode_frames(&frames, anchor_distance, full_image_distance)?;
    let encode_time = start.elapsed();

    // YUV 4:2:0
    let original_size = frames.len() * width * height * 3 / 2;
    let compression_ratio = compressed.len() as f64 / original_size as f64;

    println!("   ✓ Compressed in {:.2}s", encode_time.as_secs_f64());
    println!(
        "   Original size: {:.2} MB",
        original_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   Compressed size: {:.2} MB",
        compressed.len() as f64 / (1024.0 * 1024.0)
    );
    println!("   Compression ratio: {:.2}%", compression_ratio * 100.0);
    println!("   Bytes per frame: {}", compressed.len() / frames.len());

    // Save compressed video if requested
    if let Some(output_path) = output_path {
        println!("\n💾 Saving compressed video to: {:?}", output_path);
        let mut output_file = File::create(output_path)?;
        output_file.write_all(&compressed)?;
        println!("   ✓ Saved {} bytes", compressed.len());
    }

    // Decode and play in window
    println!("\n▶️  Opening video player window...");
    play_in_window(
        &compressed,
        width,
        height,
        anchor_distance,
        full_image_distance,
        verbose,
    )?;

    println!("\n✅ Video playback complete!");

    Ok(())
}

fn play_in_window(
    compressed: &[u8],
    width: usize,
    height: usize,
    anchor_distance: usize,
    full_image_distance: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let ordering = Ordering {
        anchor_distance,
        full_image_distance,
    };

    // Target 24 fps playback
    let target_fps = 24.0;
    let frame_duration = std::time::Duration::from_secs_f64(1.0 / target_fps);

    // Create a channel for frame streaming
    // Buffer size: With GOP size of 6, buffer 12 frames (2 GOPs ahead)
    let (tx, rx) = mpsc::sync_channel::<Option<DecodedFrame>>(12);

    // Clone compressed data for the background thread
    let compressed_data = compressed.to_vec();

    // Spawn background thread to decode frames
    let decode_thread = thread::spawn(move || {
        let gop_reader = GroupOfPicturesReader::new(Cursor::new(&compressed_data), ordering);

        let mut frame_num = 0;
        let mut total_decode_time = 0.0;

        // Use iterator to decode one frame at a time without buffering GOPs
        // This avoids stuttering from building entire GOPs before rendering
        for decoded_result in gop_reader {
            let frame_decode_start = Instant::now();

            if let Ok(decoded_frame) = decoded_result {
                let decode_ms = frame_decode_start.elapsed().as_millis();
                total_decode_time += frame_decode_start.elapsed().as_secs_f64();

                if verbose && frame_num % 30 == 0 {
                    println!(
                        "GOP decode: frame {} took {}ms (avg: {:.1}ms/frame)",
                        frame_num,
                        decode_ms,
                        (total_decode_time * 1000.0) / (frame_num + 1) as f64
                    );
                }
                frame_num += 1;

                if tx.send(Some(decoded_frame)).is_err() {
                    // Receiver dropped (window closed)
                    break;
                }
            }
        }

        if verbose {
            println!(
                "✓ Decoding complete: {} frames, avg {:.1}ms/frame",
                frame_num,
                (total_decode_time * 1000.0) / frame_num as f64
            );
        }
        // Send None to signal end of stream
        let _ = tx.send(None);
    });

    let mut window = Window::new(
        "Vidoc Video Player - GOP Compression",
        width,
        height,
        WindowOptions::default(),
    )?;

    // Limit window updates to target framerate
    window.set_target_fps(target_fps as usize);

    let mut frame_count = 0;
    let start = Instant::now();
    let mut buffer = vec![0u32; width * height];

    let mut decode_time = 0.0;
    let mut window_time = 0.0;
    let mut last_frame_time = Instant::now();

    // Receive frames from background thread
    while let Ok(frame_option) = rx.recv() {
        match frame_option {
            Some(decoded_frame) => {
                // Calculate time to wait for proper frame pacing
                let elapsed_since_last = last_frame_time.elapsed();
                if elapsed_since_last < frame_duration {
                    std::thread::sleep(frame_duration - elapsed_since_last);
                }
                last_frame_time = Instant::now();

                let decode_start = Instant::now();
                frame_to_rgb_buffer_inplace(
                    &decoded_frame.data.as_ref(),
                    width,
                    height,
                    &mut buffer,
                );
                decode_time += decode_start.elapsed().as_secs_f64();

                let window_start = Instant::now();
                window.update_with_buffer(&buffer, width, height)?;
                window_time += window_start.elapsed().as_secs_f64();

                frame_count += 1;

                if frame_count % 30 == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let fps = frame_count as f64 / elapsed;
                    let decode_fps = frame_count as f64 / decode_time;
                    println!(
                        "Frame {}: {:.1} fps total | Decode+Convert: {:.1} fps | Window: {:.1} fps",
                        frame_count,
                        fps,
                        decode_fps,
                        frame_count as f64 / window_time
                    );
                }

                if !window.is_open() || window.is_key_down(Key::Escape) {
                    break;
                }
            }
            None => {
                // End of stream
                break;
            }
        }
    }

    // Wait for decode thread to finish
    let _ = decode_thread.join();

    let total_time = start.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_time;
    println!("\nPlayback complete!");
    println!("Average FPS: {:.1}", avg_fps);
    println!(
        "Color conversion time: {:.2}s ({:.1} fps)",
        decode_time,
        frame_count as f64 / decode_time
    );
    println!(
        "Window time: {:.2}s ({:.1} fps)",
        window_time,
        frame_count as f64 / window_time
    );

    Ok(())
}

/// Convert frame to RGB buffer in-place (avoids allocation)
fn frame_to_rgb_buffer_inplace(
    frame: &SubSampleBlockGroupRef<'_, i16>,
    width: usize,
    height: usize,
    buffer: &mut [u32],
) {
    // Convert i16 blocks to f64 in parallel
    let y_f64: Vec<Block<f64>> = frame.y.par_iter().map(|b| b.convert_to()).collect();
    let cb_f64: Vec<Block<f64>> = frame.cb.par_iter().map(|b| b.convert_to()).collect();
    let cr_f64: Vec<Block<f64>> = frame.cr.par_iter().map(|b| b.convert_to()).collect();

    // Reconstruct RGB pixels from YCbCr
    use vidoc::lossy::reconstruct_pixels;
    let rgb_u8: Vec<u8> = reconstruct_pixels(
        PixelDimensions { width, height },
        &y_f64,
        &cb_f64,
        &cr_f64,
        None,
        frame.subsampling,
    );

    // Convert RGB u8 to minifb's u32 format (0RGB) in parallel directly into buffer
    buffer.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let r = rgb_u8[i * 3] as u32;
        let g = rgb_u8[i * 3 + 1] as u32;
        let b = rgb_u8[i * 3 + 2] as u32;
        *pixel = (r << 16) | (g << 8) | b;
    });
}

// Drop-in replacement for y4m_frame_to_subsample in examples/video_player.rs
//
// Key changes vs original:
//  1. LUTs computed once per call (not per pixel) for the limited→full range conversion
//  2. Direct flat-index writes into block.0[] instead of block.set(r, c, value)
//  3. Rayon parallel_iter over block rows for both luma and chroma
//  4. Edge blocks (touching the right/bottom boundary) handled separately,
//     so interior blocks have zero branches in the hot path
//  5. Chroma Cb/Cr built in a single pass (was two separate loops)

fn y4m_frame_to_subsample(
    frame: y4m::Frame,
    width: usize,
    height: usize,
    _frame_num: usize,
) -> Result<SubSampleBlockGroup<i16>, Box<dyn std::error::Error>> {
    let y_plane = frame.get_y_plane();
    let u_plane = frame.get_u_plane();
    let v_plane = frame.get_v_plane();

    let block_width = width.div_ceil(8);
    let block_height = height.div_ceil(8);

    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let chroma_block_width = chroma_width.div_ceil(8);
    let chroma_block_height = chroma_height.div_ceil(8);

    // -----------------------------------------------------------------------
    // Precompute lookup tables once per frame (not per pixel).
    //
    // Y limited range [16, 235] → full range [0, 255]:
    //   full = ((limited - 16) * 255 / 219).clamp(0, 255)
    //
    // U/V limited range [16, 240], center 128 → full range [0, 255], center 128:
    //   full = (((limited - 128) * 255 / 112) + 128).clamp(0, 255)
    // -----------------------------------------------------------------------
    let y_lut: [i16; 256] =
        std::array::from_fn(|i| (((i as f32 - 16.0) * 255.0 / 219.0).clamp(0.0, 255.0)) as i16);
    let uv_lut: [i16; 256] = std::array::from_fn(|i| {
        ((((i as f32 - 128.0) * 255.0 / 112.0) + 128.0).clamp(0.0, 255.0)) as i16
    });

    // -----------------------------------------------------------------------
    // Luma blocks — parallel over block rows
    // -----------------------------------------------------------------------
    let y_blocks: Vec<Block<i16>> = (0..block_height)
        .into_par_iter()
        .flat_map_iter(|block_row| {
            let pixel_row_base = block_row * 8;
            // How many pixel rows does this block row actually cover?
            let row_count = 8.min(height - pixel_row_base);

            (0..block_width).map(move |block_col| {
                let pixel_col_base = block_col * 8;
                let col_count = 8.min(width - pixel_col_base);

                let mut block = Block::<i16>::default();
                let data = &mut block.0;

                if row_count == 8 && col_count == 8 {
                    // Interior block — no bounds checks needed
                    for r in 0..8usize {
                        let row_offset = (pixel_row_base + r) * width + pixel_col_base;
                        for c in 0..8usize {
                            data[r * 8 + c] = y_lut[y_plane[row_offset + c] as usize];
                        }
                    }
                } else {
                    // Edge block — only fill valid pixels, rest stay 0
                    for r in 0..row_count {
                        let row_offset = (pixel_row_base + r) * width + pixel_col_base;
                        for c in 0..col_count {
                            data[r * 8 + c] = y_lut[y_plane[row_offset + c] as usize];
                        }
                    }
                }

                block
            })
        })
        .collect();

    // -----------------------------------------------------------------------
    // Chroma blocks — Cb and Cr in a single parallel pass
    // -----------------------------------------------------------------------
    let chroma_blocks: Vec<(Block<i16>, Block<i16>)> = (0..chroma_block_height)
        .into_par_iter()
        .flat_map_iter(|block_row| {
            let pixel_row_base = block_row * 8;
            let row_count = 8.min(chroma_height - pixel_row_base);

            (0..chroma_block_width).map(move |block_col| {
                let pixel_col_base = block_col * 8;
                let col_count = 8.min(chroma_width - pixel_col_base);

                let mut cb_block = Block::<i16>::default();
                let mut cr_block = Block::<i16>::default();
                let cb_data = &mut cb_block.0;
                let cr_data = &mut cr_block.0;

                if row_count == 8 && col_count == 8 {
                    for r in 0..8usize {
                        let row_offset = (pixel_row_base + r) * chroma_width + pixel_col_base;
                        for c in 0..8usize {
                            let idx = row_offset + c;
                            cb_data[r * 8 + c] = uv_lut[u_plane[idx] as usize];
                            cr_data[r * 8 + c] = uv_lut[v_plane[idx] as usize];
                        }
                    }
                } else {
                    for r in 0..row_count {
                        let row_offset = (pixel_row_base + r) * chroma_width + pixel_col_base;
                        for c in 0..col_count {
                            let idx = row_offset + c;
                            cb_data[r * 8 + c] = uv_lut[u_plane[idx] as usize];
                            cr_data[r * 8 + c] = uv_lut[v_plane[idx] as usize];
                        }
                    }
                }

                (cb_block, cr_block)
            })
        })
        .collect();

    let (cb_blocks, cr_blocks): (Vec<Block<i16>>, Vec<Block<i16>>) =
        chroma_blocks.into_iter().unzip();

    Ok(SubSampleBlockGroup::new(
        BlockDimensions {
            width: block_width,
            height: block_height,
        },
        Subsampling::Sample420,
        y_blocks,
        cb_blocks,
        cr_blocks,
    ))
}

struct FrameSource {
    frames: Vec<SubSampleBlockGroup<i16>>,
    index: RefCell<usize>,
}

impl FrameReader<i16> for FrameSource {
    fn read_frame(&self) -> error::Result<Option<SubSampleBlockGroup<i16>>> {
        let mut index = self.index.borrow_mut();
        if *index < self.frames.len() {
            let frame = self.frames[*index].clone();
            *index += 1;
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }
}

fn encode_frames(
    frames: &[SubSampleBlockGroup<i16>],
    anchor_distance: usize,
    full_image_distance: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    println!("   Encoding {} frames...", frames.len());
    let mut compressed = Vec::new();

    let source = FrameSource {
        frames: frames.to_vec(),
        index: RefCell::new(0),
    };

    let ordering = Ordering {
        anchor_distance,
        full_image_distance,
    };

    let encode_start = Instant::now();
    let gop_writer = GroupOfPicturesWriter::new(source, ordering);
    let mut stream = BitStreamWriter::new(Cursor::new(&mut compressed));
    gop_writer.write(&mut stream)?;
    let encode_time = encode_start.elapsed();

    println!(
        "   ✓ Encoding complete: {:.2}s ({:.1} fps, {:.1}ms/frame)",
        encode_time.as_secs_f64(),
        frames.len() as f64 / encode_time.as_secs_f64(),
        (encode_time.as_secs_f64() * 1000.0) / frames.len() as f64
    );

    Ok(compressed)
}
