use std::{
    cell::RefCell,
    collections::VecDeque,
    io::{Cursor, Read, Write},
    rc::Rc,
    sync::Arc,
};

use rayon::prelude::*;

use super::{bframe::BFrame, iframe::IFrame, pframe::PFrame};
use crate::{
    block::Block,
    color::Subsampling,
    dimensions::PixelDimensions,
    lossy::{
        frame::{
            r#macro::{BMacroBlock, PMacroBlock},
            Kind,
        },
        SubSampleBlockGroup,
    },
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Error, Result,
};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

pub struct DecodedFrame {
    pub kind: Kind,
    pub data: SubSampleBlockGroup<i16>,
}

#[derive(Copy, Clone)]
pub struct Ordering {
    /// Distance between anchor (I/P) frames. B-frames fill the gaps.
    pub anchor_distance: usize,
    /// Distance between I-frames (full GOP length).
    pub full_image_distance: usize,
    /// How many GOPs to encode in parallel. Higher values use more memory
    /// (each GOP is ~30 frames × ~500 KB = ~15 MB raw) but better utilise
    /// all cores, especially for short GOPs or high anchor_distance.
    /// 1 = serial (minimum memory); try 4-8 for a typical 8-core machine.
    /// Defaults to 1 if constructed with Ordering { .. } shorthand.
    pub parallel_gops: usize,
}

impl Ordering {
    /// Effective parallelism — always at least 1.
    #[inline]
    pub fn effective_parallel_gops(&self) -> usize {
        self.parallel_gops.max(1)
    }
}

impl Default for Ordering {
    /// parallel_gops defaults to the number of logical CPUs, which is a
    /// reasonable starting point. Override explicitly to tune memory/speed.
    fn default() -> Self {
        Self {
            anchor_distance: 5,
            full_image_distance: 30,
            parallel_gops: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

impl Ordering {
    pub(crate) fn frame_kind(&self, pos: usize) -> Kind {
        let group_pos = pos % self.full_image_distance;
        if group_pos == 0 {
            Kind::I
        } else if group_pos.is_multiple_of(self.anchor_distance) {
            Kind::P
        } else {
            Kind::B
        }
    }
}

pub trait FrameReader<T> {
    fn read_frame(&self) -> Result<Option<SubSampleBlockGroup<T>>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream header
// ─────────────────────────────────────────────────────────────────────────────

/// Output of one iteration of the Cb/Cr parallel encode pass.
struct ChromaEncResult {
    /// Encoded (quantized + zigzagged) Cb block, ready for the bitstream.
    cb_enc: Block<i16>,
    /// Encoded Cr block.
    cr_enc: Block<i16>,
    /// Reconstructed Cb block (dequantize + iDCT), used as the reference frame.
    cb_rec: Block<i16>,
    /// Reconstructed Cr block.
    cr_rec: Block<i16>,
}

pub(crate) enum GroupOfPicturesHeader {
    Frame {
        subsampling: Subsampling,
        dimensions: PixelDimensions,
        kind: Kind,
    },
    End,
}

impl Encodable for GroupOfPicturesHeader {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        match self {
            GroupOfPicturesHeader::Frame {
                subsampling,
                dimensions,
                kind,
            } => {
                stream.write(0u8)?;
                stream.write(u8::from(*subsampling))?;
                stream.write(dimensions.width as u32)?;
                stream.write(dimensions.height as u32)?;
                stream.write(u8::from(*kind))?;
            }
            GroupOfPicturesHeader::End => {
                stream.write(1u8)?;
            }
        }
        Ok(())
    }
}

impl Decodable for GroupOfPicturesHeader {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let header_type = stream
            .read::<u8>()?
            .ok_or(Error::FailedToDecode("header type".to_owned()))?;

        if header_type == 1 {
            Ok(Self::End)
        } else {
            Ok(Self::Frame {
                subsampling: Subsampling::from(
                    stream
                        .read::<u8>()?
                        .ok_or(Error::FailedToDecode("subsampling".to_owned()))?,
                ),
                dimensions: PixelDimensions {
                    width: stream
                        .read::<u32>()?
                        .ok_or(Error::FailedToDecode("width".to_owned()))?
                        as usize,
                    height: stream
                        .read::<u32>()?
                        .ok_or(Error::FailedToDecode("height".to_owned()))?
                        as usize,
                },
                kind: Kind::from(
                    stream
                        .read::<u8>()?
                        .ok_or(Error::FailedToDecode("kind".to_owned()))?,
                ),
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene change detection (shared between writer types)
// ─────────────────────────────────────────────────────────────────────────────

fn detect_scene_change(
    reference: &SubSampleBlockGroup<i16>,
    current: &SubSampleBlockGroup<i16>,
) -> bool {
    if reference.dimensions() != current.dimensions() {
        return true;
    }
    let sad = reference.sum_of_abs_difference(current.clone());
    let total_pixels = (reference.dimensions().width * reference.dimensions().height) as i64;
    let avg_diff = sad / total_pixels;
    avg_diff > 30
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode a single GOP into a byte buffer — no &mut self, safe to call in parallel
// ─────────────────────────────────────────────────────────────────────────────

/// All data produced by encoding a GOP's anchor chain (I + P frames).
/// B-frames can't be encoded until this is complete for their GOP.
struct GopAnchors {
    /// Encoded bytes for each anchor frame, keyed by display position.
    encoded_frames: Vec<(usize, Vec<u8>)>,
    /// Reconstructed anchor frames (what the decoder will see).
    /// B-frames reference these, not the originals.
    reconstructed: Arc<[SubSampleBlockGroup<i16>]>,
    /// The original frame slice so B-frames can be encoded later.
    frames: Arc<[SubSampleBlockGroup<i16>]>,
    ordering: Ordering,
}

/// Phase 1: encode one GOP's I/P anchor chain serially.
/// Returns anchors needed for B-frame encoding.
fn encode_anchor_chain(
    frames: Arc<[SubSampleBlockGroup<i16>]>,
    ordering: Ordering,
) -> Result<GopAnchors> {
    let mut reconstructed_anchors: Vec<SubSampleBlockGroup<i16>> = Vec::new();
    let mut encoded_frames: Vec<(usize, Vec<u8>)> = Vec::new();

    for (idx, frame) in frames.iter().enumerate() {
        let kind = gop_frame_kind(idx, ordering);
        if kind == Kind::B {
            continue;
        }

        let mut frame_data = Vec::new();
        {
            let mut tw = BitStreamWriter::new(Cursor::new(&mut frame_data));
            GroupOfPicturesHeader::Frame {
                subsampling: frame.subsampling(),
                dimensions: frame.dimensions().into(),
                kind,
            }
            .encode(&mut tw)?;

            match kind {
                Kind::I => {
                    let reconstructed = iframe_encode_and_reconstruct(frame, &mut tw)?;
                    reconstructed_anchors.push(reconstructed);
                }
                Kind::P => {
                    let backward_ref = reconstructed_anchors.last().ok_or(Error::InvalidData)?;
                    let pframe = PFrame::new(frame.clone(), backward_ref.clone());
                    let macroblocks = pframe.get_macroblocks();
                    pframe.encode(&mut tw)?;
                    tw.align_to_byte()?;
                    tw.flush()?;
                    let reconstructed = PFrame::reassemble(backward_ref.as_ref(), &macroblocks)?;
                    reconstructed_anchors.push(reconstructed);
                }
                Kind::B => unreachable!(),
            }
        }
        encoded_frames.push((idx, frame_data));
    }

    Ok(GopAnchors {
        encoded_frames,
        reconstructed: reconstructed_anchors.into(),
        frames,
        ordering,
    })
}

/// Phase 2: encode B-frames for one GOP given its completed anchor chain.
/// Safe to call in parallel across multiple GOPs.
fn encode_bframes(anchors: &GopAnchors) -> Result<Vec<(usize, Vec<u8>)>> {
    let frames = &anchors.frames;
    let ordering = anchors.ordering;
    let recon = &anchors.reconstructed;

    frames
        .par_iter()
        .enumerate()
        .filter_map(|(idx, frame)| {
            if gop_frame_kind(idx, ordering) != Kind::B {
                return None;
            }

            // Backward anchor: nearest anchor at or before this B-frame.
            let backward_pos = {
                let mut pos = idx;
                while pos > 0 && gop_frame_kind(pos, ordering) == Kind::B {
                    pos -= 1;
                }
                pos
            };
            // Forward anchor: nearest anchor after this B-frame.
            let forward_pos = {
                let mut pos = idx + 1;
                while pos < frames.len() && gop_frame_kind(pos, ordering) == Kind::B {
                    pos += 1;
                }
                if pos < frames.len() {
                    Some(pos)
                } else {
                    None
                }
            };

            let backward_anchor_idx = backward_pos / ordering.anchor_distance;
            let backward_ref = recon[backward_anchor_idx.min(recon.len() - 1)].clone();
            let forward_ref = forward_pos
                .map(|pos| recon[(pos / ordering.anchor_distance).min(recon.len() - 1)].clone());

            let mut frame_data = Vec::new();
            let result = (|| -> Result<()> {
                let mut tw = BitStreamWriter::new(Cursor::new(&mut frame_data));
                GroupOfPicturesHeader::Frame {
                    subsampling: frame.subsampling(),
                    dimensions: frame.dimensions().into(),
                    kind: Kind::B,
                }
                .encode(&mut tw)?;
                BFrame::new(frame.clone(), forward_ref, backward_ref).encode(&mut tw)?;
                tw.align_to_byte()?;
                tw.flush()
            })();

            Some(result.map(|()| (idx, frame_data)))
        })
        .collect::<Result<Vec<_>>>()
}

/// Write all frames of one GOP to stream in display order.
fn write_gop<W: Write>(
    anchors: GopAnchors,
    bframe_data: Vec<(usize, Vec<u8>)>,
    stream: &mut BitStreamWriter<W>,
) -> Result<()> {
    let mut all_frames = anchors.encoded_frames;
    all_frames.extend(bframe_data);
    all_frames.sort_by_key(|(idx, _)| *idx);
    for (_, data) in all_frames {
        stream.write_aligned_bytes(&data)?;
    }
    Ok(())
}

#[inline]
fn gop_frame_kind(local_idx: usize, ordering: Ordering) -> Kind {
    if local_idx == 0 || local_idx.is_multiple_of(ordering.full_image_distance) {
        Kind::I
    } else if local_idx.is_multiple_of(ordering.anchor_distance) {
        Kind::P
    } else {
        Kind::B
    }
}

/// Encode an I-frame to `stream` and return the reconstructed reference frame.
///
/// Avoids the old encode→decode roundtrip: we run the forward transform for
/// the stream and the inverse transform for the reference in a single pass.
fn iframe_encode_and_reconstruct<W>(
    frame: &SubSampleBlockGroup<i16>,
    stream: &mut BitStreamWriter<W>,
) -> Result<SubSampleBlockGroup<i16>>
where
    W: Write,
{
    use crate::{
        block::{quantization::Quantizor, Block},
        lossy::frame::{build_macro_blocks, r#macro::IMacroBlocks},
    };

    let dimensions = frame.dimensions();
    let subsampling = frame.subsampling();
    let lumi_q = Quantizor::<i16>::video_luminance();
    let chroma_q = Quantizor::<i16>::video_chrominance();

    dimensions.encode(stream)?;
    lumi_q.encode(stream)?;
    chroma_q.encode(stream)?;
    subsampling.encode(stream)?;

    // Y: forward pass for stream, inverse pass for reconstruction
    let (y_enc, y_rec): (Vec<Block<i16>>, Vec<Block<i16>>) = frame
        .y()
        .par_iter()
        .map(|block| {
            let quantized = lumi_q.quantize(block.dct());
            let zigzagged = quantized.zigzag();
            let reconstructed: Block<i16> = lumi_q.dequantize(quantized).idct();
            (zigzagged, reconstructed)
        })
        .unzip();

    IMacroBlocks::new(build_macro_blocks(&y_enc, dimensions)).encode(stream)?;

    let chroma_dimensions = dimensions.subsample(subsampling);

    // Cb/Cr: single parallel pass, both channels together
    let chroma_results: Vec<ChromaEncResult> = frame
        .cb()
        .par_iter()
        .zip(frame.cr().par_iter())
        .map(|(cb_block, cr_block)| {
            let cb_q = chroma_q.quantize(cb_block.dct());
            let cr_q = chroma_q.quantize(cr_block.dct());
            let cb_rec: Block<i16> = chroma_q.dequantize(cb_q).idct();
            let cr_rec: Block<i16> = chroma_q.dequantize(cr_q).idct();
            ChromaEncResult {
                cb_enc: cb_q.zigzag(),
                cr_enc: cr_q.zigzag(),
                cb_rec,
                cr_rec,
            }
        })
        .collect();

    let cap = chroma_results.len();
    let (mut cb_enc, mut cr_enc, mut cb_rec, mut cr_rec) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    for r in chroma_results {
        cb_enc.push(r.cb_enc);
        cr_enc.push(r.cr_enc);
        cb_rec.push(r.cb_rec);
        cr_rec.push(r.cr_rec);
    }

    IMacroBlocks::new(build_macro_blocks(&cb_enc, chroma_dimensions)).encode(stream)?;
    IMacroBlocks::new(build_macro_blocks(&cr_enc, chroma_dimensions)).encode(stream)?;
    stream.flush()?;

    Ok(SubSampleBlockGroup::new(
        dimensions,
        subsampling,
        y_rec,
        cb_rec,
        cr_rec,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// GroupOfPicturesWriter — parallel GOP encode
// ─────────────────────────────────────────────────────────────────────────────

pub struct GroupOfPicturesWriter<FR, T>(Rc<RefCell<GroupOfPicturesWriterInner<FR, T>>>)
where
    FR: FrameReader<T>;

impl<FR> GroupOfPicturesWriter<FR, i16>
where
    FR: FrameReader<i16>,
{
    pub fn new(content: FR, ordering: Ordering) -> Self {
        Self(Rc::new(RefCell::new(GroupOfPicturesWriterInner::new(
            content, ordering,
        ))))
    }

    pub fn write<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.0.borrow_mut().encode(stream)
    }
}

struct GroupOfPicturesWriterInner<FR, T> {
    ordering: Ordering,
    content: FR,
    buffered_frame: Option<SubSampleBlockGroup<i16>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<FR> GroupOfPicturesWriterInner<FR, i16>
where
    FR: FrameReader<i16>,
{
    fn new(content: FR, ordering: Ordering) -> Self {
        Self {
            ordering,
            content,
            buffered_frame: None,
            _phantom: std::marker::PhantomData,
        }
    }

    fn encode<W>(&mut self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        // Collect `parallel_gops` GOPs at a time, encode their anchor chains
        // in parallel (rayon par_iter), then encode all B-frames across the
        // batch in one parallel sweep, then write in order and free memory.
        //
        // parallel_gops=1 → one GOP at a time, minimum memory (~15 MB/GOP).
        // parallel_gops=N → N×15 MB peak, better core utilisation.
        // Sweet spot on an 8-core machine is typically 4-8.

        let ordering = self.ordering;
        let parallel_gops = ordering.effective_parallel_gops();

        loop {
            // Collect up to `parallel_gops` GOPs serially.
            let mut batch: Vec<Arc<[SubSampleBlockGroup<i16>]>> = Vec::with_capacity(parallel_gops);

            for _ in 0..parallel_gops {
                let frames = self.collect_gop_frames()?;
                if frames.is_empty() {
                    break;
                }
                batch.push(frames.into());
            }

            if batch.is_empty() {
                break;
            }

            // Encode all anchor chains in parallel across the batch.
            let anchor_batch: Vec<GopAnchors> = batch
                .into_par_iter()
                .map(|frames| encode_anchor_chain(Arc::clone(&frames), ordering))
                .collect::<Result<Vec<_>>>()?;

            // Encode all B-frames across the entire batch in one parallel sweep.
            // B-frames only reference anchors within their own GOP so this is safe.
            let bframe_batch: Vec<Vec<(usize, Vec<u8>)>> = anchor_batch
                .par_iter()
                .map(encode_bframes)
                .collect::<Result<Vec<_>>>()?;

            // Write each GOP in order, freeing memory as we go.
            for (anchors, bframes) in anchor_batch.into_iter().zip(bframe_batch) {
                write_gop(anchors, bframes, stream)?;
            }
        }

        GroupOfPicturesHeader::End.encode(stream)?;
        stream.flush()
    }

    fn collect_gop_frames(&mut self) -> Result<Vec<SubSampleBlockGroup<i16>>> {
        let mut frames = Vec::new();

        if let Some(buffered) = self.buffered_frame.take() {
            frames.push(buffered);
        }

        let first_iteration = frames.is_empty();

        for local_pos in 0..self.ordering.full_image_distance {
            if let Some(frame) = self.content.read_frame()? {
                if !first_iteration {
                    if self.ordering.frame_kind(local_pos + frames.len()) == Kind::I {
                        self.buffered_frame = Some(frame);
                        break;
                    }
                    if let Some(ref prev) = frames.first().cloned() {
                        if detect_scene_change(prev, &frame) {
                            self.buffered_frame = Some(frame);
                            break;
                        }
                    }
                }
                frames.push(frame);
            } else {
                break;
            }
        }

        Ok(frames)
    }
}

impl<FR> Encodable for GroupOfPicturesWriter<FR, i16>
where
    FR: FrameReader<i16>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.0.borrow_mut().encode(stream)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GroupOfPicturesReader — full GOP buffering, correct B-frame display order
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::enum_variant_names)]
enum DecodedFrameData {
    IFrame(SubSampleBlockGroup<f32>),
    PFrame(Vec<PMacroBlock<i16>>),
    BFrame(Vec<BMacroBlock<i16>>),
}

pub struct GroupOfPicturesReader<R>
where
    R: Read,
{
    ordering: Ordering,
    source: BitStreamReader<R>,
    decoded_frames: VecDeque<(Kind, SubSampleBlockGroup<i16>)>,
    last_iframe: Option<SubSampleBlockGroup<i16>>,
    eof: bool,
}

impl<R> GroupOfPicturesReader<R>
where
    R: Read,
{
    pub fn new(source: R, ordering: Ordering) -> Self {
        Self {
            ordering,
            source: BitStreamReader::new(source),
            decoded_frames: VecDeque::new(),
            last_iframe: None,
            eof: false,
        }
    }

    fn decode_gop(&mut self) -> Result<Vec<(Kind, SubSampleBlockGroup<i16>)>> {
        let mut gop_data: Vec<(Kind, DecodedFrameData)> = Vec::new();
        let mut has_seen_iframe = false;

        loop {
            let header = GroupOfPicturesHeader::decode(&mut self.source)?;
            match header {
                GroupOfPicturesHeader::Frame { kind, .. } => {
                    if has_seen_iframe && kind == Kind::I {
                        let next = self.decode_frame(kind)?;
                        if let DecodedFrameData::IFrame(iframe) = next {
                            self.last_iframe = Some(iframe.into());
                        }
                        break;
                    }
                    if kind == Kind::I {
                        has_seen_iframe = true;
                    }
                    gop_data.push((kind, self.decode_frame(kind)?));
                    if gop_data.len() >= self.ordering.full_image_distance {
                        break;
                    }
                }
                GroupOfPicturesHeader::End => {
                    self.eof = true;
                    break;
                }
            }
        }

        self.reassemble_gop(gop_data)
    }

    fn decode_frame(&mut self, kind: Kind) -> Result<DecodedFrameData> {
        let data = match kind {
            Kind::I => DecodedFrameData::IFrame(IFrame::<i16>::decode(&mut self.source)?),
            Kind::P => DecodedFrameData::PFrame(PFrame::decode(&mut self.source)?.into_inner()),
            Kind::B => DecodedFrameData::BFrame(BFrame::decode(&mut self.source)?.into_inner()),
        };
        self.source.align_to_byte()?;
        Ok(data)
    }

    fn reassemble_gop(
        &mut self,
        gop_data: Vec<(Kind, DecodedFrameData)>,
    ) -> Result<Vec<(Kind, SubSampleBlockGroup<i16>)>> {
        if gop_data.is_empty() {
            return Ok(Vec::new());
        }
        if gop_data[0].0 != Kind::I {
            return Err(Error::InvalidData);
        }

        let gop_len = gop_data.len();
        let frame_kinds: Vec<Kind> = gop_data.iter().map(|(k, _)| *k).collect();
        let mut all_decoded: Vec<Option<SubSampleBlockGroup<i16>>> = vec![None; gop_len];

        // Pass 1: decode anchors serially (I and P frames)
        for (idx, (kind, data)) in gop_data.iter().enumerate() {
            match (kind, data) {
                (Kind::I, DecodedFrameData::IFrame(iframe)) => {
                    all_decoded[idx] = Some(iframe.clone().into());
                }
                (Kind::P, DecodedFrameData::PFrame(pmbs)) => {
                    let backward_pos = idx.saturating_sub(self.ordering.anchor_distance);
                    let backward = all_decoded[backward_pos]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;
                    all_decoded[idx] = Some(PFrame::reassemble(backward.as_ref(), pmbs)?);
                }
                _ => {}
            }
        }

        // Pass 2: decode B-frames in parallel — both anchors are now available
        let b_results: Vec<(usize, SubSampleBlockGroup<i16>)> = gop_data
            .par_iter()
            .enumerate()
            .filter_map(|(idx, (kind, data))| {
                if let (Kind::B, DecodedFrameData::BFrame(bmbs)) = (kind, data) {
                    Some((idx, bmbs))
                } else {
                    None
                }
            })
            .map(|(idx, bmbs)| {
                let backward_pos = idx - (idx % self.ordering.anchor_distance);
                let forward_pos =
                    idx + (self.ordering.anchor_distance - (idx % self.ordering.anchor_distance));

                let backward = all_decoded[backward_pos]
                    .clone()
                    .ok_or(Error::InvalidData)?;
                let forward = if forward_pos < gop_len {
                    all_decoded[forward_pos].clone()
                } else {
                    None
                };

                let frame = BFrame::reassemble(
                    forward.as_ref().map(|f| f.as_ref()),
                    backward.as_ref(),
                    bmbs,
                )?;
                Ok((idx, frame))
            })
            .collect::<Result<Vec<_>>>()?;

        for (idx, frame) in b_results {
            all_decoded[idx] = Some(frame);
        }

        // Collect in display order
        let mut result = Vec::with_capacity(gop_len);
        for (idx, kind) in frame_kinds.into_iter().enumerate() {
            let frame = all_decoded[idx].take().ok_or(Error::InvalidData)?;
            result.push((kind, frame));
        }

        if let Some((_, first)) = result.first() {
            self.last_iframe = Some(first.clone());
        }

        Ok(result)
    }
}

impl<R> Iterator for GroupOfPicturesReader<R>
where
    R: Read,
{
    type Item = Result<DecodedFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((kind, data)) = self.decoded_frames.pop_front() {
            return Some(Ok(DecodedFrame { kind, data }));
        }
        if self.eof {
            return None;
        }

        match self.decode_gop() {
            Ok(frames) => {
                if frames.is_empty() {
                    return None;
                }
                for frame in frames {
                    self.decoded_frames.push_back(frame);
                }
                self.decoded_frames
                    .pop_front()
                    .map(|(kind, data)| Ok(DecodedFrame { kind, data }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FramesReader — streaming decoder with correct B-frame handling
//
// Design: B-frames need both a forward and backward anchor. In streaming, the
// forward anchor hasn't been decoded yet when the B-frames arrive. The solution
// is to buffer B-frames until the next anchor (P-frame) is decoded, then
// decode all buffered B-frames and emit them in display order BEFORE the
// P-frame.
//
// Display order:  I  B  B  B  P  B  B  B  P  ...
// Decode order:   I  B  B  B  P  B  B  B  P  ...  (same for streaming)
// Emit order:     I  B  B  B  P  B  B  B  P  ...  (correct)
//
// The latency is exactly anchor_distance frames — we buffer B-frames until
// their forward anchor arrives, then flush them all at once. This is the
// theoretical minimum for any B-frame streaming decoder.
// ─────────────────────────────────────────────────────────────────────────────

pub struct FramesReader<R>
where
    R: Read,
{
    source: BitStreamReader<R>,
    /// Frames ready to return to the caller, in display order.
    ready: VecDeque<DecodedFrame>,
    /// B-frames buffered waiting for their forward anchor.
    pending_bframes: Vec<(usize, Vec<BMacroBlock<i16>>)>,
    last_anchor: Option<SubSampleBlockGroup<i16>>,
    last_iframe: Option<SubSampleBlockGroup<i16>>,
    frame_pos: usize,
    eof: bool,
}

impl<R> FramesReader<R>
where
    R: Read,
{
    pub fn new(source: R) -> Self {
        Self {
            source: BitStreamReader::new(source),
            ready: VecDeque::new(),
            pending_bframes: Vec::new(),
            last_anchor: None,
            last_iframe: None,
            frame_pos: 0,
            eof: false,
        }
    }

    /// Flush pending B-frames now that both anchors are known.
    ///
    /// `backward` = the anchor before the B-frames.
    /// `forward`  = the anchor just decoded (the P-frame or next I-frame).
    /// B-frames are decoded in parallel then inserted into `ready` in display
    /// order BEFORE `forward_frame`.
    fn flush_bframes(
        &mut self,
        backward: &SubSampleBlockGroup<i16>,
        forward: &SubSampleBlockGroup<i16>,
        forward_frame: DecodedFrame,
    ) -> Result<()> {
        if self.pending_bframes.is_empty() {
            self.ready.push_back(forward_frame);
            return Ok(());
        }

        // Decode pending B-frames in parallel — both anchors are now available.
        let pending = std::mem::take(&mut self.pending_bframes);

        let decoded_bframes: Vec<(usize, SubSampleBlockGroup<i16>)> = pending
            .par_iter()
            .map(|(display_pos, bmbs)| {
                let frame = BFrame::reassemble(Some(forward.as_ref()), backward.as_ref(), bmbs)?;
                Ok((*display_pos, frame))
            })
            .collect::<Result<Vec<_>>>()?;

        // Sort by display position and emit before the forward anchor.
        let mut sorted = decoded_bframes;
        sorted.sort_by_key(|(pos, _)| *pos);

        for (_, frame) in sorted {
            self.ready.push_back(DecodedFrame {
                kind: Kind::B,
                data: frame,
            });
        }
        self.ready.push_back(forward_frame);

        Ok(())
    }

    /// Decode one frame from the stream and process it.
    /// Returns false when the stream ends.
    fn decode_next(&mut self) -> Result<bool> {
        let header = GroupOfPicturesHeader::decode(&mut self.source)?;

        match header {
            GroupOfPicturesHeader::End => {
                self.eof = true;
                // Drain any pending B-frames using the last anchor as both references.
                // These are at the end of the stream with no forward anchor —
                // we fall back to using the backward anchor as a substitute.
                if !self.pending_bframes.is_empty() {
                    if let Some(backward) = self.last_anchor.clone().or(self.last_iframe.clone()) {
                        let pending = std::mem::take(&mut self.pending_bframes);
                        let backward_clone = backward.clone();
                        let decoded: Vec<(usize, SubSampleBlockGroup<i16>)> = pending
                            .par_iter()
                            .map(|(pos, bmbs)| {
                                // No true forward anchor — use backward as fallback.
                                // Quality degrades slightly but output is correct.
                                let frame =
                                    BFrame::reassemble(None, backward_clone.as_ref(), bmbs)?;
                                Ok((*pos, frame))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let mut sorted = decoded;
                        sorted.sort_by_key(|(pos, _)| *pos);
                        for (_, frame) in sorted {
                            self.ready.push_back(DecodedFrame {
                                kind: Kind::B,
                                data: frame,
                            });
                        }
                    }
                }
                Ok(false)
            }

            GroupOfPicturesHeader::Frame { kind, .. } => {
                let frame_data = match kind {
                    Kind::I => DecodedFrameData::IFrame(IFrame::<i16>::decode(&mut self.source)?),
                    Kind::P => {
                        DecodedFrameData::PFrame(PFrame::decode(&mut self.source)?.into_inner())
                    }
                    Kind::B => {
                        DecodedFrameData::BFrame(BFrame::decode(&mut self.source)?.into_inner())
                    }
                };
                self.source.align_to_byte()?;

                let display_pos = self.frame_pos;
                self.frame_pos += 1;

                match (kind, frame_data) {
                    (Kind::I, DecodedFrameData::IFrame(iframe)) => {
                        let converted: SubSampleBlockGroup<i16> = iframe.into();

                        // An I-frame can also be the forward anchor for pending B-frames.
                        // This happens at a GOP boundary when B-frames from the previous
                        // GOP haven't been flushed yet.
                        if !self.pending_bframes.is_empty() {
                            if let Some(backward) =
                                self.last_anchor.clone().or(self.last_iframe.clone())
                            {
                                let forward_frame = DecodedFrame {
                                    kind: Kind::I,
                                    data: converted.clone(),
                                };
                                self.flush_bframes(&backward, &converted, forward_frame)?;
                            } else {
                                // No backward anchor — drop pending B-frames
                                self.pending_bframes.clear();
                                self.ready.push_back(DecodedFrame {
                                    kind: Kind::I,
                                    data: converted.clone(),
                                });
                            }
                        } else {
                            self.ready.push_back(DecodedFrame {
                                kind: Kind::I,
                                data: converted.clone(),
                            });
                        }

                        self.last_iframe = Some(converted.clone());
                        self.last_anchor = Some(converted);
                    }

                    (Kind::P, DecodedFrameData::PFrame(pmbs)) => {
                        let backward = self
                            .last_anchor
                            .as_ref()
                            .or(self.last_iframe.as_ref())
                            .ok_or(Error::InvalidData)?;

                        let reconstructed = PFrame::reassemble(backward.as_ref(), &pmbs)?;

                        // P-frame is the forward anchor for buffered B-frames.
                        if !self.pending_bframes.is_empty() {
                            let backward_clone = backward.clone();
                            let p_frame = DecodedFrame {
                                kind: Kind::P,
                                data: reconstructed.clone(),
                            };
                            self.flush_bframes(&backward_clone, &reconstructed, p_frame)?;
                        } else {
                            self.ready.push_back(DecodedFrame {
                                kind: Kind::P,
                                data: reconstructed.clone(),
                            });
                        }

                        self.last_anchor = Some(reconstructed);
                    }

                    (Kind::B, DecodedFrameData::BFrame(bmbs)) => {
                        // Buffer this B-frame — we don't have the forward anchor yet.
                        self.pending_bframes.push((display_pos, bmbs));
                    }

                    _ => return Err(Error::InvalidData),
                }

                Ok(true)
            }
        }
    }
}

impl<R> Iterator for FramesReader<R>
where
    R: Read,
{
    type Item = Result<DecodedFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return any frames already ready.
        if let Some(frame) = self.ready.pop_front() {
            return Some(Ok(frame));
        }

        if self.eof {
            return None;
        }

        // Keep reading until we have a frame ready or the stream ends.
        loop {
            match self.decode_next() {
                Err(e) => return Some(Err(e)),
                Ok(false) => {
                    // EOF — return any remaining ready frames.
                    return self.ready.pop_front().map(Ok);
                }
                Ok(true) => {
                    if let Some(frame) = self.ready.pop_front() {
                        return Some(Ok(frame));
                    }
                    // No frame ready yet (B-frame buffered) — keep reading.
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FramesWriter — streaming encoder (no B-frames, minimal latency)
// ─────────────────────────────────────────────────────────────────────────────

pub struct FramesWriter<FR, T>
where
    FR: FrameReader<T>,
{
    ordering: Ordering,
    content: FR,
    _phantom: std::marker::PhantomData<T>,
}

impl<FR> FramesWriter<FR, i16>
where
    FR: FrameReader<i16>,
{
    pub fn new(content: FR, ordering: Ordering) -> Self {
        Self {
            ordering,
            content,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn write<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.encode(stream)
    }
}

impl<FR> Encodable for FramesWriter<FR, i16>
where
    FR: FrameReader<i16>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let mut frame_pos = 0;
        let mut last_anchor: Option<SubSampleBlockGroup<i16>> = None;
        let mut last_iframe: Option<SubSampleBlockGroup<i16>> = None;
        let mut last_for_scene: Option<SubSampleBlockGroup<i16>> = None;

        while let Some(frame) = self.content.read_frame()? {
            if let Some(ref prev) = last_for_scene {
                if detect_scene_change(prev, &frame) {
                    frame_pos = 0;
                }
            }

            // B-frames become P-frames in streaming (no future references).
            let kind = match self.ordering.frame_kind(frame_pos) {
                Kind::B => Kind::P,
                other => other,
            };

            GroupOfPicturesHeader::Frame {
                subsampling: frame.subsampling(),
                dimensions: frame.dimensions().into(),
                kind,
            }
            .encode(stream)?;

            match kind {
                Kind::I => {
                    IFrame::new(frame.clone()).encode(stream)?;
                    stream.align_to_byte()?;
                    last_iframe = Some(frame.clone());
                    last_anchor = Some(frame.clone());
                    last_for_scene = Some(frame);
                }
                Kind::P => {
                    let reference = last_anchor
                        .as_ref()
                        .or(last_iframe.as_ref())
                        .ok_or(Error::InvalidData)?;
                    let pframe = PFrame::new(frame, reference.clone());
                    let mbs = pframe.get_macroblocks();
                    pframe.encode(stream)?;
                    stream.align_to_byte()?;
                    let reconstructed = PFrame::reassemble(reference.as_ref(), &mbs)?;
                    last_for_scene = Some(reconstructed.clone());
                    last_anchor = Some(reconstructed);
                }
                Kind::B => unreachable!(),
            }

            stream.flush()?;
            frame_pos += 1;
        }

        GroupOfPicturesHeader::End.encode(stream)?;
        stream.flush()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (unchanged from original, kept for regression coverage)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{block::Block, dimensions::BlockDimensions};

    struct VecFrameReader(Rc<RefCell<VecFrameReaderInner>>);
    struct VecFrameReaderInner {
        frames: Vec<SubSampleBlockGroup<i16>>,
        index: usize,
    }

    impl VecFrameReader {
        fn new(frames: Vec<SubSampleBlockGroup<i16>>) -> Self {
            Self(Rc::new(RefCell::new(VecFrameReaderInner {
                frames,
                index: 0,
            })))
        }
    }

    impl FrameReader<i16> for VecFrameReader {
        fn read_frame(&self) -> Result<Option<SubSampleBlockGroup<i16>>> {
            let mut inner = self.0.borrow_mut();
            if inner.index < inner.frames.len() {
                let frame = inner.frames[inner.index].clone();
                inner.index += 1;
                Ok(Some(frame))
            } else {
                Ok(None)
            }
        }
    }

    fn make_frame(width: usize, height: usize, luma: i16) -> SubSampleBlockGroup<i16> {
        let mut block = Block::<i16>::default();
        for r in 0..8 {
            for c in 0..8 {
                block.set(r, c, luma);
            }
        }
        SubSampleBlockGroup::new(
            BlockDimensions { width, height },
            Subsampling::Sample420,
            vec![block; width * height],
            vec![Block::<i16>::default(); width * height / 4],
            vec![Block::<i16>::default(); width * height / 4],
        )
    }

    fn make_gradient_frame(width: usize, height: usize, base: i16) -> SubSampleBlockGroup<i16> {
        let y: Vec<Block<i16>> = (0..width * height)
            .map(|i| {
                let mut b = Block::<i16>::default();
                let (brow, bcol) = (i / width, i % width);
                for r in 0..8 {
                    for c in 0..8 {
                        b.set(
                            r,
                            c,
                            base + ((brow * 8 + r) as i16 % 50) + ((bcol * 8 + c) as i16 % 50),
                        );
                    }
                }
                b
            })
            .collect();
        SubSampleBlockGroup::new(
            BlockDimensions { width, height },
            Subsampling::Sample420,
            y,
            vec![Block::<i16>::default(); width * height / 4],
            vec![Block::<i16>::default(); width * height / 4],
        )
    }

    fn mse(a: &SubSampleBlockGroup<i16>, b: &SubSampleBlockGroup<i16>) -> f64 {
        let mut sq = 0i64;
        let mut n = 0i64;
        for (ab, bb) in a.y().iter().zip(b.y().iter()) {
            for r in 0..8 {
                for c in 0..8 {
                    let d = ab.get(r, c) as i64 - bb.get(r, c) as i64;
                    sq += d * d;
                    n += 1;
                }
            }
        }
        sq as f64 / n as f64
    }

    fn psnr(mse: f64) -> f64 {
        if mse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * (255.0_f64 / mse.sqrt()).log10()
        }
    }

    fn encode_decode(
        frames: Vec<SubSampleBlockGroup<i16>>,
        ordering: Ordering,
    ) -> Vec<DecodedFrame> {
        let mut buf = Vec::new();
        {
            let reader = VecFrameReader::new(frames);
            let writer = GroupOfPicturesWriter::new(reader, ordering);
            let mut stream = BitStreamWriter::new(Cursor::new(&mut buf));
            writer.encode(&mut stream).unwrap();
        }
        let reader = GroupOfPicturesReader::new(Cursor::new(&buf), ordering);
        reader.map(|f| f.unwrap()).collect()
    }

    #[test]
    fn test_gop_frame_pattern() {
        let ordering = Ordering {
            anchor_distance: 3,
            full_image_distance: 12,
            ..Default::default()
        };
        let expected = [
            Kind::I,
            Kind::B,
            Kind::B,
            Kind::P,
            Kind::B,
            Kind::B,
            Kind::P,
            Kind::B,
            Kind::B,
            Kind::P,
            Kind::B,
            Kind::B,
        ];
        for (idx, &exp) in expected.iter().enumerate() {
            assert_eq!(ordering.frame_kind(idx), exp, "frame {idx}");
        }
        assert_eq!(ordering.frame_kind(12), Kind::I);
    }

    #[test]
    fn test_identical_frames() {
        let ordering = Ordering {
            anchor_distance: 3,
            full_image_distance: 12,
            ..Default::default()
        };
        let frames: Vec<_> = (0..12).map(|_| make_frame(8, 8, 128)).collect();
        let original = frames.clone();
        let decoded = encode_decode(frames, ordering);
        assert_eq!(decoded.len(), original.len());
        for (i, (orig, dec)) in original.iter().zip(&decoded).enumerate() {
            let p = psnr(mse(orig, &dec.data));
            assert!(p > 40.0 || p.is_infinite(), "frame {i}: PSNR {p:.2} dB");
        }
    }

    #[test]
    fn test_gradient_frames_quality() {
        let ordering = Ordering {
            anchor_distance: 3,
            full_image_distance: 12,
            ..Default::default()
        };
        let frames: Vec<_> = (0..24)
            .map(|i| make_gradient_frame(16, 12, 100 + i as i16 * 10))
            .collect();
        let original = frames.clone();
        let decoded = encode_decode(frames, ordering);
        assert_eq!(decoded.len(), original.len());

        let mut i_psnrs = vec![];
        let mut p_psnrs = vec![];
        let mut b_psnrs = vec![];

        for (orig, dec) in original.iter().zip(&decoded) {
            let p = psnr(mse(orig, &dec.data));
            match dec.kind {
                Kind::I => i_psnrs.push(p),
                Kind::P => p_psnrs.push(p),
                Kind::B => b_psnrs.push(p),
            }
        }

        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            avg(&i_psnrs) > 35.0,
            "I-frame avg PSNR: {:.2}",
            avg(&i_psnrs)
        );
        assert!(
            avg(&p_psnrs) > 30.0,
            "P-frame avg PSNR: {:.2}",
            avg(&p_psnrs)
        );
        assert!(
            avg(&b_psnrs) > 20.0,
            "B-frame avg PSNR: {:.2}",
            avg(&b_psnrs)
        );
    }

    #[test]
    fn test_streaming_reader_bframe_order() {
        // FramesReader must emit frames in display order even when B-frames
        // are buffered pending their forward anchor.
        let ordering = Ordering {
            anchor_distance: 3,
            full_image_distance: 12,
            ..Default::default()
        };
        let frames: Vec<_> = (0..12)
            .map(|i| make_gradient_frame(8, 8, 80 + i as i16 * 5))
            .collect();

        // Encode with GroupOfPicturesWriter (produces real B-frames)
        let mut buf = Vec::new();
        {
            let reader = VecFrameReader::new(frames.clone());
            let writer = GroupOfPicturesWriter::new(reader, ordering);
            let mut stream = BitStreamWriter::new(Cursor::new(&mut buf));
            writer.encode(&mut stream).unwrap();
        }

        // Decode with FramesReader — should emit all 12 frames in display order
        let reader = FramesReader::new(Cursor::new(&buf));
        let decoded: Vec<DecodedFrame> = reader.map(|r| r.unwrap()).collect();

        assert_eq!(decoded.len(), 12, "should decode all 12 frames");

        // Verify display order matches the original GOP pattern
        let expected_kinds = [
            Kind::I,
            Kind::B,
            Kind::B,
            Kind::P,
            Kind::B,
            Kind::B,
            Kind::P,
            Kind::B,
            Kind::B,
            Kind::P,
            Kind::B,
            Kind::B,
        ];
        for (i, (dec, &exp)) in decoded.iter().zip(expected_kinds.iter()).enumerate() {
            assert_eq!(dec.kind, exp, "frame {i} kind mismatch");
        }
    }
}
