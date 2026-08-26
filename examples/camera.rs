use std::{
    io::Cursor,
    sync::mpsc::{channel, Receiver},
    thread,
    time::{Duration, Instant},
};

use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{RequestedFormat, RequestedFormatType},
    Camera,
};
use vidoc::{
    bitstream::BitStreamWriter,
    color::Subsampling,
    dimensions::PixelDimensions,
    image::ImageRgb8,
    lossy::{
        frame::gop::{FrameReader, GroupOfPicturesReader, GroupOfPicturesWriter, Ordering},
        SubSampleBlockGroup,
    },
    pixels::Rgb8,
};

/// A FrameReader that reads from a channel of captured frames
struct ChannelFrameReader {
    frame_rx: Receiver<SubSampleBlockGroup<i16>>,
}

impl FrameReader<i16> for ChannelFrameReader {
    fn read_frame(&self) -> vidoc::error::Result<Option<SubSampleBlockGroup<i16>>> {
        match self.frame_rx.recv() {
            Ok(frame) => Ok(Some(frame)),
            Err(_) => Ok(None), // Channel closed
        }
    }
}

fn main() -> vidoc::error::Result<()> {
    println!("🎥 Camera Real-time Codec Demo");
    println!("================================\n");

    // Query available cameras
    let cameras = nokhwa::query(nokhwa::native_api_backend().unwrap())
        .map_err(|e| vidoc::error::Error::Io(std::io::Error::other(e)))?;
    if cameras.is_empty() {
        eprintln!("❌ No cameras found!");
        return Ok(());
    }

    println!("📹 Available cameras:");
    for (idx, cam) in cameras.iter().enumerate() {
        println!("  [{}] {}", idx, cam.human_name());
    }

    // Use first camera
    let camera_info = &cameras[0];
    println!("\n✓ Using: {}", camera_info.human_name());

    // Request camera format
    let requested_format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestFrameRate(30));

    // Open camera
    let mut camera = Camera::new(camera_info.index().clone(), requested_format)
        .map_err(|e| vidoc::error::Error::Io(std::io::Error::other(e)))?;

    camera
        .open_stream()
        .map_err(|e| vidoc::error::Error::Io(std::io::Error::other(e)))?;

    let camera_format = camera.camera_format();
    let width = camera_format.width() as usize;
    let height = camera_format.height() as usize;
    let dimensions = PixelDimensions { width, height };

    println!(
        "📐 Camera format: {}x{} @ {} FPS",
        width,
        height,
        camera_format.frame_rate()
    );
    println!("\n🔧 Codec settings:");
    println!("   Subsampling: 4:2:0");
    println!("   GOP: I-frame every 10, P-frame every 3 (low latency)");
    println!("   Compression: DCT + Quantization + ANS\n");

    // Configuration
    const MAX_FRAMES: usize = 100; // Capture 100 frames for demo

    // Create channels
    let (capture_tx, capture_rx) = channel();
    let (encode_tx, encode_rx) = channel();

    // GOP ordering - smaller GOP for lower memory usage
    let ordering = Ordering {
        anchor_distance: 3,      // P-frame every 3 frames
        full_image_distance: 10, // I-frame every 10 frames (reduced from 30)
    };

    // Encoder thread
    let encoder_ordering = Ordering {
        anchor_distance: ordering.anchor_distance,
        full_image_distance: ordering.full_image_distance,
    };
    let encode_handle = thread::spawn(move || {
        println!("🔵 Encoder thread started\n");

        let mut encode_stream = Vec::new();
        let gop_writer = GroupOfPicturesWriter::new(
            ChannelFrameReader {
                frame_rx: capture_rx,
            },
            encoder_ordering,
        );

        let encode_start = Instant::now();
        println!("   Encoding frames (expecting up to {})...", MAX_FRAMES);

        let mut stream = BitStreamWriter::new(Cursor::new(&mut encode_stream));
        match gop_writer.write(&mut stream) {
            Ok(_) => {
                let total_time = encode_start.elapsed();
                println!(
                    "✅ Encoding complete: {:.2}s, {} bytes",
                    total_time.as_secs_f64(),
                    encode_stream.len()
                );
                if encode_stream.is_empty() {
                    eprintln!("⚠️  Warning: Encoded stream is empty! No frames were encoded.");
                } else {
                    println!("   Sending encoded data to decoder...");
                    if encode_tx.send(encode_stream).is_err() {
                        eprintln!("❌ Failed to send encoded data to decoder!");
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Encoding error: {}", e);
            }
        }
        println!("🔵 Encoder thread done");
    });

    // Decoder thread
    let decoder_ordering = Ordering {
        anchor_distance: ordering.anchor_distance,
        full_image_distance: ordering.full_image_distance,
    };
    let decoder_dimensions = dimensions;
    let decode_handle = thread::spawn(move || {
        println!("🟢 Decoder thread started\n");

        if let Ok(encoded_data) = encode_rx.recv() {
            println!(
                "📦 Received {} bytes, decoding frame-by-frame...",
                encoded_data.len()
            );

            let decode_start = Instant::now();
            let cursor = Cursor::new(encoded_data);
            let gop_reader = GroupOfPicturesReader::new(cursor, decoder_ordering);

            let mut decoded_frames = Vec::new();
            let mut frame_idx = 0;

            // Decode frames one at a time without buffering GOPs
            for decoded_result in gop_reader {
                if let Ok(decoded_frame) = decoded_result {
                    let blocks_f64 = decoded_frame.data.convert_to::<f64>();
                    decoded_frames.push(blocks_f64);
                    frame_idx += 1;

                    if frame_idx % 10 == 0 {
                        println!("   Decoded {} frames...", frame_idx);
                    }
                }
            }

            println!(
                "✅ Decoded {} frames in {:.2}s\n",
                decoded_frames.len(),
                decode_start.elapsed().as_secs_f64()
            );

            (decoded_frames, decoder_dimensions)
        } else {
            (Vec::new(), decoder_dimensions)
        }
    });

    // Display window
    let mut window = Window::new(
        "Camera Codec - Capturing... Press ESC to finish",
        width,
        height,
        WindowOptions::default(),
    )
    .map_err(|e| {
        vidoc::error::Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{:?}", e),
        ))
    })?;

    window.set_target_fps(30);

    let mut frame_buffer: Vec<u32> = vec![0; width * height];
    let start_time = Instant::now();
    let mut captured_frames = 0;

    println!("▶️  Capturing {} frames...\n", MAX_FRAMES);

    // Capture loop
    while window.is_open() && !window.is_key_down(Key::Escape) && captured_frames < MAX_FRAMES {
        match camera.frame() {
            Ok(frame) => {
                let decoded = frame
                    .decode_image::<RgbFormat>()
                    .map_err(|e| vidoc::error::Error::Io(std::io::Error::other(e)))?;
                let rgb_data = decoded.as_raw().to_vec();

                // Convert to YCbCr and subsample for encoding
                let img = ImageRgb8::new(
                    dimensions,
                    Rgb8::new(rgb_data.clone()),
                    Subsampling::Sample420,
                );
                let subsampled = img.subsample_into_block_ycbcr().convert_to::<i16>();

                // Send to encoder
                let _ = capture_tx.send(subsampled);

                // Display live camera feed
                for (i, rgb_chunk) in rgb_data.chunks(3).enumerate() {
                    if i < frame_buffer.len() && rgb_chunk.len() == 3 {
                        let r = rgb_chunk[0] as u32;
                        let g = rgb_chunk[1] as u32;
                        let b = rgb_chunk[2] as u32;
                        frame_buffer[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    }
                }

                window
                    .update_with_buffer(&frame_buffer, width, height)
                    .map_err(|e| {
                        vidoc::error::Error::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("{:?}", e),
                        ))
                    })?;

                captured_frames += 1;
                if captured_frames % 10 == 0 {
                    println!("📸 Captured {} frames...", captured_frames);
                }
            }
            Err(e) => {
                eprintln!("Camera error: {}", e);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    let capture_time = start_time.elapsed();
    println!(
        "\n⏹  Capture complete: {} frames in {:.2}s ({:.1} fps)\n",
        captured_frames,
        capture_time.as_secs_f64(),
        captured_frames as f64 / capture_time.as_secs_f64()
    );

    // Close capture channel to signal encoder
    println!("📪 Closing capture channel to signal encoder...");
    drop(capture_tx);
    drop(camera);

    // Wait for encoding/decoding
    println!("⏳ Waiting for encoder to finish...");
    let encode_result = encode_handle.join();
    if encode_result.is_err() {
        eprintln!("⚠️  Encoder thread panicked!");
    }

    println!("⏳ Waiting for decoder to finish...");
    let (decoded_frames, dec_dims) = decode_handle.join().unwrap();

    if !decoded_frames.is_empty() {
        println!("🎬 Playing back decoded frames...\n");

        let mut window = Window::new(
            "Decoded Video - Press ESC to exit",
            dec_dims.width,
            dec_dims.height,
            WindowOptions::default(),
        )
        .map_err(|e| {
            vidoc::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("{:?}", e),
            ))
        })?;

        window.set_target_fps(30);

        let mut playback_buffer: Vec<u32> = vec![0; dec_dims.width * dec_dims.height];

        for (idx, decoded_blocks) in decoded_frames.iter().enumerate() {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                break;
            }

            // Reconstruct RGB from decoded YCbCr
            let rgb_flat: Vec<u8> = vidoc::lossy::reconstruct_pixels(
                dec_dims,
                &decoded_blocks.as_ref().y,
                &decoded_blocks.as_ref().cb,
                &decoded_blocks.as_ref().cr,
                None,
                Subsampling::Sample420,
            );

            for (i, rgb_chunk) in rgb_flat.chunks(3).enumerate() {
                if i < playback_buffer.len() && rgb_chunk.len() == 3 {
                    let r = rgb_chunk[0] as u32;
                    let g = rgb_chunk[1] as u32;
                    let b = rgb_chunk[2] as u32;
                    playback_buffer[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
                }
            }

            window
                .update_with_buffer(&playback_buffer, dec_dims.width, dec_dims.height)
                .map_err(|e| {
                    vidoc::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{:?}", e),
                    ))
                })?;

            if (idx + 1) % 10 == 0 {
                println!("   Playing frame {}...", idx + 1);
            }

            thread::sleep(Duration::from_millis(33)); // ~30 fps
        }

        println!("\n✅ Playback complete!");
    }

    println!("\n📊 Demo complete!");
    Ok(())
}
