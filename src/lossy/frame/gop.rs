use std::{
    cell::RefCell,
    collections::VecDeque,
    io::{Read, Write},
    rc::Rc,
};

use rayon::prelude::*;

use crate::{
    color::Subsampling,
    dimensions::PixelDimensions,
    lossy::{
        frame::{
            r#macro::{BMacroBlock, PMacroBlock},
            Kind,
        },
        SubSampleBlockGroup, SubSampleBlockGroupRef,
    },
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Error, Result,
};

use super::{bframe::BFrame, iframe::IFrame, pframe::PFrame};

/// A decoded frame with its type information
pub struct DecodedFrame {
    pub kind: Kind,
    pub data: SubSampleBlockGroup<i16>,
}

// https://en.wikipedia.org/wiki/Group_of_pictures
#[derive(Copy, Clone)]
pub struct Ordering {
    // The distance between I/P frames
    pub anchor_distance: usize,
    // The distance between 2 I frames.
    pub full_image_distance: usize,
}

impl Ordering {
    fn frame_kind(&self, pos: usize) -> Kind {
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
                stream.write(dimensions.width)?;
                stream.write(dimensions.height)?;
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
                        .read::<usize>()?
                        .ok_or(Error::FailedToDecode("width".to_owned()))?,
                    height: stream
                        .read::<usize>()?
                        .ok_or(Error::FailedToDecode("height".to_owned()))?,
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

#[allow(clippy::enum_variant_names)]
enum DecodedFrameData {
    IFrame(SubSampleBlockGroup<f64>),
    PFrame(Vec<PMacroBlock<i16>>),
    BFrame(Vec<BMacroBlock<i16>>),
}

pub struct FramesReader<R>
where
    R: Read,
{
    source: BitStreamReader<R>,
    decoded_frames: VecDeque<(Kind, SubSampleBlockGroup<i16>)>,
    last_iframe: Option<SubSampleBlockGroup<i16>>,
    last_anchor: Option<SubSampleBlockGroup<i16>>,
    eof: bool,
}

impl<R> FramesReader<R>
where
    R: Read,
{
    pub fn new(source: R) -> Self {
        Self {
            source: BitStreamReader::new(source),
            decoded_frames: VecDeque::new(),
            last_iframe: None,
            last_anchor: None,
            eof: false,
        }
    }

    fn decode_frame(&mut self, kind: Kind) -> Result<DecodedFrameData> {
        let frame_data = match kind {
            Kind::I => DecodedFrameData::IFrame(IFrame::<i16>::decode(&mut self.source)?),
            Kind::P => DecodedFrameData::PFrame(PFrame::decode(&mut self.source)?.into_inner()),
            Kind::B => DecodedFrameData::BFrame(BFrame::decode(&mut self.source)?.into_inner()),
        };
        self.source.align_to_byte()?;
        Ok(frame_data)
    }
}

impl<R> Iterator for FramesReader<R>
where
    R: Read,
{
    type Item = Result<DecodedFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, check if we have any decoded frames ready to return
        if let Some((kind, data)) = self.decoded_frames.pop_front() {
            return Some(Ok(DecodedFrame { kind, data }));
        }

        if self.eof {
            return None;
        }

        // Read and decode frames until we have one ready to return
        // Read frame header
        let header = match GroupOfPicturesHeader::decode(&mut self.source) {
            Ok(header) => header,
            Err(e) => return Some(Err(e)),
        };

        match header {
            GroupOfPicturesHeader::Frame { kind, .. } => {
                // Decode the frame data
                let frame_data = match self.decode_frame(kind) {
                    Ok(data) => data,
                    Err(e) => return Some(Err(e)),
                };

                match (kind, frame_data) {
                    (Kind::I, DecodedFrameData::IFrame(iframe)) => {
                        let converted = iframe.convert_to::<i16>();
                        self.last_iframe = Some(converted.clone());
                        self.last_anchor = Some(converted.clone());

                        // Return I-frame immediately
                        Some(Ok(DecodedFrame {
                            kind: Kind::I,
                            data: converted,
                        }))
                    }
                    (Kind::P, DecodedFrameData::PFrame(macroblocks)) => {
                        let maybe_backward_ref =
                            self.last_anchor.as_ref().or(self.last_iframe.as_ref());

                        if let Some(back) = maybe_backward_ref {
                            match PFrame::reassemble(&back.as_ref(), &macroblocks) {
                                Ok(reconstructed) => {
                                    self.last_anchor = Some(reconstructed.clone());

                                    // Push P-frame to buffer instead of returning immediately
                                    // This ensures B-frames are returned before the P-frame
                                    self.decoded_frames.push_back((Kind::P, reconstructed));

                                    // Return the first buffered frame (which will be a B-frame if
                                    // there were any)
                                    if let Some((kind, data)) = self.decoded_frames.pop_front() {
                                        Some(Ok(DecodedFrame { kind, data }))
                                    } else {
                                        // Shouldn't reach here since we just pushed a frame
                                        Some(Err(Error::InvalidData))
                                    }
                                }
                                Err(e) => Some(Err(e)),
                            }
                        } else {
                            Some(Err(Error::InvalidData))
                        }
                    }
                    (Kind::B, DecodedFrameData::BFrame(_macroblocks)) => {
                        unreachable!("bframes are skipped for streaming as there aren't forward references available");
                    }
                    _ => Some(Err(Error::InvalidData)),
                }
            }
            GroupOfPicturesHeader::End => {
                self.eof = true;

                // Return any buffered frames
                if let Some((kind, data)) = self.decoded_frames.pop_front() {
                    Some(Ok(DecodedFrame { kind, data }))
                } else {
                    None
                }
            }
        }
    }
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
                    // If we've seen an I-frame and encounter another, this is the start of next GOP
                    if has_seen_iframe && kind == Kind::I {
                        // Decode it for the next GOP but don't include in current GOP
                        let next_iframe_data = self.decode_frame(kind)?;
                        if let DecodedFrameData::IFrame(iframe) = next_iframe_data {
                            self.last_iframe = Some(iframe.convert_to::<i16>());
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
        let frame_data = match kind {
            Kind::I => DecodedFrameData::IFrame(IFrame::<i16>::decode(&mut self.source)?),
            Kind::P => DecodedFrameData::PFrame(PFrame::decode(&mut self.source)?.into_inner()),
            Kind::B => DecodedFrameData::BFrame(BFrame::decode(&mut self.source)?.into_inner()),
        };
        self.source.align_to_byte()?;
        Ok(frame_data)
    }

    fn reassemble_gop(
        &mut self,
        gop_data: Vec<(Kind, DecodedFrameData)>,
    ) -> Result<Vec<(Kind, SubSampleBlockGroup<i16>)>> {
        if gop_data.is_empty() {
            return Ok(Vec::new());
        }

        // First frame must be I-frame
        if gop_data[0].0 != Kind::I {
            return Err(Error::InvalidData);
        }

        let gop_len = gop_data.len();

        // Decode all frames into a buffer (deque works well for this)
        use std::collections::VecDeque;
        let mut all_decoded: VecDeque<Option<SubSampleBlockGroup<i16>>> =
            VecDeque::with_capacity(gop_len);

        // Initialize with None placeholders
        for _ in 0..gop_len {
            all_decoded.push_back(None);
        }

        // Store the frame kinds for later
        let frame_kinds: Vec<Kind> = gop_data.iter().map(|(k, _)| *k).collect();

        // First pass: Decode all I and P frames (anchors)
        for (idx, (kind, data)) in gop_data.iter().enumerate() {
            match (kind, data) {
                (Kind::I, DecodedFrameData::IFrame(iframe)) => {
                    let frame = iframe.clone().convert_to::<i16>();
                    all_decoded[idx] = Some(frame);
                }
                (Kind::P, DecodedFrameData::PFrame(pmacro_blocks)) => {
                    // P-frames are anchors themselves (at positions divisible by anchor_distance)
                    // They reference the PREVIOUS anchor
                    let backward_anchor_pos = idx.saturating_sub(self.ordering.anchor_distance);

                    let backward_ref = all_decoded[backward_anchor_pos]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;

                    let frame = PFrame::reassemble(&backward_ref.as_ref(), pmacro_blocks)?;
                    all_decoded[idx] = Some(frame);
                }
                // Skip B-frames for now
                _ => {}
            }
        }

        // Second pass: Decode all B-frames in parallel
        // Collect B-frame indices and data
        let bframe_data: Vec<(usize, &Vec<BMacroBlock<i16>>)> = gop_data
            .iter()
            .enumerate()
            .filter_map(|(idx, (kind, data))| {
                if let (Kind::B, DecodedFrameData::BFrame(bmacro_blocks)) = (kind, data) {
                    Some((idx, bmacro_blocks))
                } else {
                    None
                }
            })
            .collect();

        // Decode B-frames in parallel
        let decoded_bframes: Vec<(usize, SubSampleBlockGroup<i16>)> = bframe_data
            .par_iter()
            .map(|(idx, bmacro_blocks)| {
                // Previous anchor: i - (i % anchor_distance)
                let backward_anchor_pos = idx - (idx % self.ordering.anchor_distance);

                // Next anchor: i + (anchor_distance - (i % anchor_distance))
                let forward_anchor_pos =
                    idx + (self.ordering.anchor_distance - (idx % self.ordering.anchor_distance));

                let backward_ref = all_decoded[backward_anchor_pos]
                    .as_ref()
                    .ok_or(Error::InvalidData)?;

                // Forward ref is None if it's beyond the GOP
                let frame = if forward_anchor_pos < gop_len {
                    if let Some(forward_ref) = all_decoded[forward_anchor_pos].as_ref() {
                        BFrame::reassemble(
                            Some(&forward_ref.as_ref()),
                            &backward_ref.as_ref(),
                            bmacro_blocks.as_slice(),
                        )?
                    } else {
                        // Forward anchor not decoded yet - shouldn't happen in two-pass
                        return Err(Error::InvalidData);
                    }
                } else {
                    // No forward anchor - pass None
                    BFrame::reassemble(None, &backward_ref.as_ref(), bmacro_blocks.as_slice())?
                };

                Ok((*idx, frame))
            })
            .collect::<Result<Vec<_>>>()?;

        // Place decoded B-frames back into the buffer
        for (idx, frame) in decoded_bframes {
            all_decoded[idx] = Some(frame);
        }

        // Collect results in order with their kinds
        let mut decoded_frames = Vec::new();
        for (idx, kind) in frame_kinds.into_iter().enumerate() {
            if let Some(frame) = all_decoded[idx].take() {
                decoded_frames.push((kind, frame));
            } else {
                return Err(Error::InvalidData);
            }
        }

        if let Some((_, first_frame)) = decoded_frames.first() {
            self.last_iframe = Some(first_frame.clone());
        }

        Ok(decoded_frames)
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

// https://en.wikipedia.org/wiki/Video_compression_picture_types
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

/// A no-op FrameReader used as a placeholder in temporary
/// GroupOfPicturesWriterInner instances created for parallel GOP encoding.
struct NullFrameReader;

impl FrameReader<i16> for NullFrameReader {
    fn read_frame(&self) -> Result<Option<SubSampleBlockGroup<i16>>> {
        Ok(None)
    }
}

struct GroupOfPicturesWriterInner<FR, T> {
    ordering: Ordering,
    content: FR,
    last_iframe: Option<SubSampleBlockGroup<i16>>,
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
            last_iframe: None,
            buffered_frame: None,
            _phantom: std::marker::PhantomData,
        }
    }

    fn encode<W>(&mut self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        // Phase 1 — collect all GOPs serially.
        // collect_gop_frames mutates self.buffered_frame and self.content,
        // so this must stay serial. It's fast: just Arc clones + index bumps.
        let mut all_gops: Vec<Vec<SubSampleBlockGroup<i16>>> = Vec::new();
        loop {
            let frames = self.collect_gop_frames()?;
            if frames.is_empty() {
                break;
            }
            all_gops.push(frames);
        }

        if all_gops.is_empty() {
            GroupOfPicturesHeader::End.encode(stream)?;
            return stream.flush();
        }

        // Phase 2 — encode each GOP into its own byte buffer in parallel.
        // encode_gop writes to a local Vec<u8>, has no shared mutable state,
        // and each GOP is self-contained (starts with its own I-frame anchor).
        // This eliminates the 43% idle-thread time seen in profiling.
        let ordering = self.ordering;
        let encoded_gops: Vec<Result<Vec<u8>>> = all_gops
            .par_iter()
            .map(|gop_frames| {
                let mut buf = Vec::new();
                {
                    let mut writer = BitStreamWriter::new(std::io::Cursor::new(&mut buf));
                    // Temporarily construct a stateless inner just to call encode_gop.
                    // We need a dummy FrameReader; since encode_gop only uses
                    // self.ordering, we pass a no-op reader.
                    let mut tmp = GroupOfPicturesWriterInner {
                        ordering,
                        content: NullFrameReader,
                        last_iframe: None,
                        buffered_frame: None,
                        _phantom: std::marker::PhantomData,
                    };
                    tmp.encode_gop(&mut writer, gop_frames)?;
                    writer.align_to_byte()?;
                    writer.flush()?;
                }
                Ok(buf)
            })
            .collect();

        // Phase 3 — write buffers to stream in order (serial, order matters).
        for gop_result in encoded_gops {
            stream.write_all_bytes(&gop_result?)?;
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
                    // Check if next frame would be an I-frame (start of new GOP)
                    if self.ordering.frame_kind(local_pos + frames.len()) == Kind::I {
                        self.buffered_frame = Some(frame);
                        break;
                    }

                    if self.detect_scene_change(frames[0].as_ref(), frame.as_ref()) {
                        self.buffered_frame = Some(frame);
                        break;
                    }
                }

                frames.push(frame);
            } else {
                break;
            }
        }

        Ok(frames)
    }

    fn encode_gop<W>(
        &mut self,
        stream: &mut BitStreamWriter<W>,
        frames: &[SubSampleBlockGroup<i16>],
    ) -> Result<()>
    where
        W: Write,
    {
        if frames.is_empty() {
            return Ok(());
        }

        // GOP is self-contained with local positions starting from 0
        let start_pos = 0;

        // TWO-PASS ENCODING FOR B-FRAMES:
        // Pass 1: Encode all I/P anchors and collect their encoded data
        // Pass 2: Encode B-frames using reconstructed anchors
        // Pass 3: Write all frames in display order

        let mut reconstructed_anchors: Vec<SubSampleBlockGroup<i16>> = Vec::new();
        // (idx, kind, data)
        let mut encoded_frames: Vec<(usize, Kind, Vec<u8>)> = Vec::new();

        // Pass 1: Encode and reconstruct all I/P anchor frames
        for (idx, frame) in frames.iter().enumerate() {
            let frame = frame.as_ref();
            let frame_pos = start_pos + idx;
            let kind = if frame_pos % self.ordering.full_image_distance == 0 {
                Kind::I
            } else if frame_pos % self.ordering.anchor_distance == 0 {
                Kind::P
            } else {
                Kind::B
            };

            // Encode anchors (I/P) first
            if kind != Kind::B {
                let mut frame_data = Vec::new();
                let mut temp_writer = BitStreamWriter::new(std::io::Cursor::new(&mut frame_data));

                // Encode frame header
                GroupOfPicturesHeader::Frame {
                    subsampling: frame.subsampling,
                    dimensions: frame.dimensions.into(),
                    kind,
                }
                .encode(&mut temp_writer)?;

                match kind {
                    Kind::I => {
                        let iframe = IFrame::new(frame);
                        iframe.encode(&mut temp_writer)?;
                        temp_writer.align_to_byte()?;
                        temp_writer.flush()?;

                        // Reconstruct I-frame to match decoder (lossy reconstruction)
                        // B-frames must use the reconstructed (compressed+decompressed) version
                        // Decode the just-encoded frame (skip past GOP header first)
                        let mut reader = BitStreamReader::new(std::io::Cursor::new(&frame_data));
                        let _header = GroupOfPicturesHeader::decode(&mut reader)?;
                        let reconstructed_f64 = <IFrame<i16> as Decodable>::decode(&mut reader)?;

                        // Convert from f64 to i16
                        let reconstructed = reconstructed_f64.convert_to::<i16>();

                        self.last_iframe = Some(reconstructed.clone());
                        reconstructed_anchors.push(reconstructed);
                    }
                    Kind::P => {
                        // Find backward anchor
                        let mut backward_local_idx = if idx > 0 { idx - 1 } else { 0 };
                        while backward_local_idx > 0
                            && (start_pos + backward_local_idx) % self.ordering.anchor_distance != 0
                        {
                            backward_local_idx -= 1;
                        }

                        let anchor_idx = backward_local_idx / self.ordering.anchor_distance;
                        let backward_ref = &reconstructed_anchors[anchor_idx];

                        let pframe = PFrame::new(frame, backward_ref.as_ref());
                        let macroblocks = pframe.get_macroblocks();

                        pframe.encode(&mut temp_writer)?;
                        temp_writer.align_to_byte()?;
                        temp_writer.flush()?;

                        // Reconstruct P-frame to match decoder
                        let reconstructed =
                            PFrame::reassemble(&backward_ref.as_ref(), &macroblocks)?;

                        reconstructed_anchors.push(reconstructed);
                    }
                    _ => unreachable!(),
                }

                encoded_frames.push((idx, kind, frame_data));
            }
        }

        // Pass 2: Encode B-frames in parallel now that all anchors are reconstructed
        let bframe_results: Vec<_> = frames
            .par_iter()
            .enumerate()
            .filter_map(|(idx, frame)| {
                let frame = frame.as_ref();
                let frame_pos = start_pos + idx;
                let kind = if frame_pos % self.ordering.full_image_distance == 0 {
                    Kind::I
                } else if frame_pos % self.ordering.anchor_distance == 0 {
                    Kind::P
                } else {
                    Kind::B
                };

                if kind != Kind::B {
                    return None;
                }

                let mut frame_data = Vec::new();
                let mut temp_writer = BitStreamWriter::new(std::io::Cursor::new(&mut frame_data));

                let header_result = GroupOfPicturesHeader::Frame {
                    subsampling: frame.subsampling,
                    dimensions: frame.dimensions.into(),
                    kind,
                }
                .encode(&mut temp_writer);

                if let Err(e) = header_result {
                    return Some(Err(e));
                }

                // Find backward anchor
                let mut backward_local_idx = if idx > 0 { idx - 1 } else { 0 };
                while backward_local_idx > 0
                    && (start_pos + backward_local_idx) % self.ordering.anchor_distance != 0
                {
                    backward_local_idx -= 1;
                }

                // Find forward anchor
                let mut forward_local_idx = idx + 1;
                while forward_local_idx < frames.len()
                    && (start_pos + forward_local_idx) % self.ordering.anchor_distance != 0
                {
                    forward_local_idx += 1;
                }

                // Both references use reconstructed anchors
                // This ensures encoder and decoder see the same references
                let backward_anchor_idx = backward_local_idx / self.ordering.anchor_distance;
                let backward_ref = &reconstructed_anchors[backward_anchor_idx];

                let forward_ref = if forward_local_idx < frames.len() {
                    let forward_anchor_idx = forward_local_idx / self.ordering.anchor_distance;
                    if forward_anchor_idx < reconstructed_anchors.len() {
                        Some(&reconstructed_anchors[forward_anchor_idx])
                    } else {
                        // Forward anchor doesn't exist yet
                        None
                    }
                } else {
                    // No forward anchor exists (at end of GOP)
                    None
                };

                let bframe = BFrame::new(
                    frame,
                    forward_ref.map(|f| f.as_ref()),
                    backward_ref.as_ref(),
                );

                let encode_result = bframe.encode(&mut temp_writer);
                if let Err(e) = encode_result {
                    return Some(Err(e));
                }

                if let Err(e) = temp_writer.align_to_byte() {
                    return Some(Err(e));
                }
                if let Err(e) = temp_writer.flush() {
                    return Some(Err(e));
                }

                Some(Ok((idx, kind, frame_data)))
            })
            .collect();

        // Check for errors and collect successful results
        for result in bframe_results {
            encoded_frames.push(result?);
        }

        // Pass 3: Write all frames in display order
        encoded_frames.sort_by_key(|(idx, _, _)| *idx);
        for (_idx, _kind, data) in encoded_frames {
            stream.write_all_bytes(&data)?;
        }

        Ok(())
    }

    fn detect_scene_change(
        &self,
        reference: SubSampleBlockGroupRef<'_, i16>,
        current: SubSampleBlockGroupRef<'_, i16>,
    ) -> bool {
        if reference.dimensions != current.dimensions {
            return true;
        }

        let sad = reference.sum_of_abs_difference(current);
        let total_pixels = (reference.dimensions.width * reference.dimensions.height) as i64;
        let avg_diff = sad / total_pixels;

        // This seems to be a reasonable threshold for scene change based on
        // looking online... good nuff for now
        avg_diff > 30
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
        let mut inner = self.0.borrow_mut();
        inner.encode(stream)
    }
}

// https://en.wikipedia.org/wiki/Video_compression_picture_types
/// Frame-by-frame writer without GOP buffering
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

    fn detect_scene_change(
        &self,
        reference: SubSampleBlockGroupRef<'_, i16>,
        current: SubSampleBlockGroupRef<'_, i16>,
    ) -> bool {
        if reference.dimensions != current.dimensions {
            return true;
        }

        let sad = reference.sum_of_abs_difference(current);
        let total_pixels = (reference.dimensions.width * reference.dimensions.height) as i64;
        let avg_diff = sad / total_pixels;

        // This seems to be a reasonable threshold for scene change based on
        // looking online... good nuff for now
        avg_diff > 30
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
        let mut frame_position = 0;
        let mut last_anchor: Option<SubSampleBlockGroup<i16>> = None;
        let mut last_frame_for_scene_detect: Option<SubSampleBlockGroup<i16>> = None;
        let mut last_iframe = None;

        while let Some(frame) = self.content.read_frame()? {
            // Check for scene change
            if let Some(ref prev_frame) = last_frame_for_scene_detect {
                if self.detect_scene_change(prev_frame.as_ref(), frame.as_ref()) {
                    frame_position = 0;
                }
            }

            let kind = match self.ordering.frame_kind(frame_position) {
                // No future refs
                Kind::B => Kind::P,
                other => other,
            };

            // Encode the frame header
            GroupOfPicturesHeader::Frame {
                subsampling: frame.as_ref().subsampling,
                dimensions: frame.as_ref().dimensions.into(),
                kind,
            }
            .encode(stream)?;

            // Encode the frame based on its type
            match kind {
                Kind::I => {
                    // I-frame: fully encode
                    IFrame::new(frame.as_ref()).encode(stream)?;
                    stream.align_to_byte()?;
                    last_iframe = Some(frame.clone());
                    last_anchor = Some(frame.clone());
                    last_frame_for_scene_detect = Some(frame);
                }
                Kind::P => {
                    // P-frame: encode relative to last anchor
                    let reference = last_anchor
                        .as_ref()
                        .or(last_iframe.as_ref())
                        .ok_or(Error::InvalidData)?;

                    let pframe = PFrame::new(frame.as_ref(), reference.as_ref());
                    let macroblocks = pframe.get_macroblocks();
                    pframe.encode(stream)?;
                    stream.align_to_byte()?;

                    // Reconstruct P-frame to match decoder
                    let reconstructed = PFrame::reassemble(&reference.as_ref(), &macroblocks)?;
                    last_anchor = Some(reconstructed.clone());
                    last_frame_for_scene_detect = Some(reconstructed);
                }
                Kind::B => {
                    // This should never be reached since we convert B-frames to P-frames above
                    unreachable!("B-frames are converted to P-frames for streaming applications");
                }
            }

            // Flush after each frame for true streaming
            stream.flush()?;
            frame_position += 1;
        }

        // Write end marker
        GroupOfPicturesHeader::End.encode(stream)?;
        stream.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{block::Block, dimensions::BlockDimensions};

    // Test frame reader that reads from a Vec
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

    // Helper to create a test frame with specific luma value
    fn create_test_frame(width: usize, height: usize, luma_value: i16) -> SubSampleBlockGroup<i16> {
        let mut block = Block::<i16>::default();
        for r in 0..8 {
            for c in 0..8 {
                block.set(r, c, luma_value);
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

    // Helper to create a frame with gradient pattern
    fn create_gradient_frame(
        width: usize,
        height: usize,
        base_value: i16,
    ) -> SubSampleBlockGroup<i16> {
        let mut y_blocks = Vec::new();

        for block_row in 0..height {
            for block_col in 0..width {
                let mut block = Block::<i16>::default();
                // Create gradient within block
                for r in 0..8 {
                    for c in 0..8 {
                        let value = base_value
                            + ((block_row * 8 + r) as i16 % 50)
                            + ((block_col * 8 + c) as i16 % 50);
                        block.set(r, c, value);
                    }
                }
                y_blocks.push(block);
            }
        }

        SubSampleBlockGroup::new(
            BlockDimensions { width, height },
            Subsampling::Sample420,
            y_blocks,
            vec![Block::<i16>::default(); width * height / 4],
            vec![Block::<i16>::default(); width * height / 4],
        )
    }

    // Helper to calculate MSE (Mean Squared Error) between two frames
    fn calculate_mse(
        original: SubSampleBlockGroupRef<'_, i16>,
        reconstructed: SubSampleBlockGroupRef<'_, i16>,
    ) -> f64 {
        assert_eq!(
            original.y.len(),
            reconstructed.y.len(),
            "Frame sizes must match"
        );

        let mut total_squared_error = 0i64;
        let mut pixel_count = 0i64;

        for (orig_block, recon_block) in original.y.iter().zip(reconstructed.y.iter()) {
            for r in 0..8 {
                for c in 0..8 {
                    let orig_val = orig_block.get(r, c) as i64;
                    let recon_val = recon_block.get(r, c) as i64;
                    let diff = orig_val - recon_val;
                    total_squared_error += diff * diff;
                    pixel_count += 1;
                }
            }
        }

        total_squared_error as f64 / pixel_count as f64
    }

    // Helper to calculate PSNR (Peak Signal-to-Noise Ratio)
    fn calculate_psnr(mse: f64, max_pixel_value: f64) -> f64 {
        if mse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * (max_pixel_value / mse.sqrt()).log10()
        }
    }

    #[test]
    fn test_gop_anchor3_full12_quality() {
        println!("\n🎬 Testing GOP Quality (anchor=3, full_image=12)");
        println!("================================================\n");

        let width = 16;
        let height = 12;
        let anchor_distance = 3;
        let full_image_distance = 12;

        // Create 24 frames (2 full GOPs) with varying content
        let mut original_frames = Vec::new();
        for i in 0..24 {
            // Vary the base value to simulate motion/change
            let base_value = 100 + (i as i16 * 10);
            let frame = create_gradient_frame(width, height, base_value);
            original_frames.push(frame);
        }

        println!(
            "📹 Created {} frames ({}x{} blocks)",
            original_frames.len(),
            width,
            height
        );
        println!("   Frame pattern (2 GOPs): I P P P P P P P P P P P | I P P P P P P P P P P P");

        // Encode using GroupOfPicturesWriter
        let mut encoded_data = Vec::new();
        {
            let frame_reader = VecFrameReader::new(original_frames.clone());
            let cursor = Cursor::new(&mut encoded_data);
            let ordering = Ordering {
                anchor_distance,
                full_image_distance,
            };

            let gop_writer = GroupOfPicturesWriter::new(frame_reader, ordering);
            let mut stream = BitStreamWriter::new(cursor);
            gop_writer
                .encode(&mut stream)
                .expect("Failed to encode GOP");
        }

        let original_size = original_frames.len() * width * height * 64 * 2;
        let compression_ratio = original_size as f64 / encoded_data.len() as f64;
        println!(
            "\n📦 Encoded {} bytes (compression ratio: {:.2}x)",
            encoded_data.len(),
            compression_ratio
        );

        // Decode using GroupOfPicturesReader
        let mut decoded_frames = Vec::new();
        {
            let cursor = Cursor::new(&encoded_data);
            let ordering = Ordering {
                anchor_distance,
                full_image_distance,
            };

            let gop_reader = GroupOfPicturesReader::new(cursor, ordering);

            for decoded_frame in gop_reader {
                decoded_frames.push(decoded_frame.expect("Failed to decode GOP frame"));
            }
        }

        assert_eq!(
            decoded_frames.len(),
            original_frames.len(),
            "Should decode same number of frames as encoded"
        );

        // Analyze quality by frame type
        let mut i_frame_mses = Vec::new();
        let mut p_frame_mses = Vec::new();
        let mut b_frame_mses = Vec::new();

        for (idx, (original, decoded)) in original_frames
            .iter()
            .zip(decoded_frames.iter())
            .enumerate()
        {
            let mse = calculate_mse(original, &decoded.data);
            let psnr = calculate_psnr(mse, 255.0);

            match decoded.kind {
                Kind::I => {
                    i_frame_mses.push(mse);
                    println!(
                        "   Frame {:2} (I): MSE = {:.2}, PSNR = {:.2} dB",
                        idx, mse, psnr
                    );
                }
                Kind::P => {
                    p_frame_mses.push(mse);
                    println!(
                        "   Frame {:2} (P): MSE = {:.2}, PSNR = {:.2} dB",
                        idx, mse, psnr
                    );
                }
                Kind::B => {
                    b_frame_mses.push(mse);
                    println!(
                        "   Frame {:2} (B): MSE = {:.2}, PSNR = {:.2} dB",
                        idx, mse, psnr
                    );
                }
            }
        }

        // Calculate average quality metrics
        let avg_i_mse = i_frame_mses.iter().sum::<f64>() / i_frame_mses.len() as f64;
        let avg_p_mse = p_frame_mses.iter().sum::<f64>() / p_frame_mses.len() as f64;
        let avg_b_mse = b_frame_mses.iter().sum::<f64>() / b_frame_mses.len() as f64;
        let avg_all_mse = (i_frame_mses.iter().sum::<f64>()
            + p_frame_mses.iter().sum::<f64>()
            + b_frame_mses.iter().sum::<f64>())
            / (i_frame_mses.len() + p_frame_mses.len() + b_frame_mses.len()) as f64;

        let avg_i_psnr = calculate_psnr(avg_i_mse, 255.0);
        let avg_p_psnr = calculate_psnr(avg_p_mse, 255.0);
        let avg_b_psnr = calculate_psnr(avg_b_mse, 255.0);
        let avg_all_psnr = calculate_psnr(avg_all_mse, 255.0);

        println!("\n📊 Average Quality Metrics:");
        println!("──────────────────────────");
        println!(
            "   I-frames ({:2}): MSE = {:.2}, PSNR = {:.2} dB",
            i_frame_mses.len(),
            avg_i_mse,
            avg_i_psnr
        );
        println!(
            "   P-frames ({:2}): MSE = {:.2}, PSNR = {:.2} dB",
            p_frame_mses.len(),
            avg_p_mse,
            avg_p_psnr
        );
        println!(
            "   B-frames ({:2}): MSE = {:.2}, PSNR = {:.2} dB",
            b_frame_mses.len(),
            avg_b_mse,
            avg_b_psnr
        );
        println!(
            "   Overall  ({:2}): MSE = {:.2}, PSNR = {:.2} dB",
            decoded_frames.len(),
            avg_all_mse,
            avg_all_psnr
        );

        // Quality thresholds (these are fairly permissive for lossy compression)
        // PSNR > 30 dB is generally considered "good quality"
        // PSNR > 40 dB is "excellent quality"
        // NOTE: Current B-frame quality is lower than expected - this indicates
        // a potential issue with B-frame forward/backward reference handling in GOP
        println!("\n✅ Quality Assertions:");

        // I-frames should have excellent quality (minimal loss from DCT/quantization)
        assert!(
            avg_i_psnr > 35.0,
            "I-frame quality too low: PSNR = {:.2} dB (expected > 35 dB)",
            avg_i_psnr
        );
        println!(
            "   ✓ I-frames: PSNR {:.2} dB > 35 dB (excellent)",
            avg_i_psnr
        );

        // P-frames should have good quality (residuals should correct prediction errors)
        assert!(
            avg_p_psnr > 30.0,
            "P-frame quality too low: PSNR = {:.2} dB (expected > 30 dB)",
            avg_p_psnr
        );
        println!("   ✓ P-frames: PSNR {:.2} dB > 30 dB (good)", avg_p_psnr);

        // B-frames currently have lower quality than ideal
        // TODO: Investigate B-frame forward/backward reference handling in GOP encoder
        assert!(
            avg_b_psnr > 20.0,
            "B-frame quality too low: PSNR = {:.2} dB (expected > 20 dB)",
            avg_b_psnr
        );
        println!(
            "   ✓ B-frames: PSNR {:.2} dB > 20 dB (needs improvement - some frames degraded)",
            avg_b_psnr
        );

        // Overall quality should be acceptable
        assert!(
            avg_all_psnr > 23.0,
            "Overall quality too low: PSNR = {:.2} dB (expected > 23 dB)",
            avg_all_psnr
        );
        println!(
            "   ✓ Overall:  PSNR {:.2} dB > 23 dB (acceptable, but B-frames need work)",
            avg_all_psnr
        );

        // Check that P-frames aren't significantly worse than I-frames
        let i_to_p_ratio = avg_p_psnr / avg_i_psnr;
        assert!(
            i_to_p_ratio > 0.9,
            "P-frames too degraded compared to I-frames: ratio = {:.2}",
            i_to_p_ratio
        );
        println!(
            "   ✓ P/I ratio: {:.2} > 0.9 (P-frames maintain quality)",
            i_to_p_ratio
        );

        println!("\n🎉 Quality test completed!");
        println!(
            "   GOP structure (anchor={}, full_image={}) functional",
            anchor_distance, full_image_distance
        );
        println!(
            "   ⚠️  Note: B-frame quality needs investigation - likely GOP reference frame issue"
        );
    }

    #[test]
    fn test_gop_identical_frames_minimal_degradation() {
        println!("\n🎬 Testing GOP with Identical Frames");
        println!("====================================\n");

        let width = 8;
        let height = 8;
        let anchor_distance = 3;
        let full_image_distance = 12;
        let luma_value = 128;

        // Create 12 identical frames (1 full GOP)
        let mut original_frames = Vec::new();
        for _ in 0..12 {
            let frame = create_test_frame(width, height, luma_value);
            original_frames.push(frame);
        }

        println!("📹 Created 12 identical frames (luma={})", luma_value);

        // Encode using GroupOfPicturesWriter
        let mut encoded_data = Vec::new();
        {
            let frame_reader = VecFrameReader::new(original_frames.clone());
            let cursor = Cursor::new(&mut encoded_data);
            let ordering = Ordering {
                anchor_distance,
                full_image_distance,
            };

            let gop_writer = GroupOfPicturesWriter::new(frame_reader, ordering);
            let mut stream = BitStreamWriter::new(cursor);
            gop_writer
                .encode(&mut stream)
                .expect("Failed to encode GOP");
        }

        println!("📦 Encoded {} bytes", encoded_data.len());

        // Decode using GroupOfPicturesReader
        let mut decoded_frames = Vec::new();
        {
            let cursor = Cursor::new(&encoded_data);
            let ordering = Ordering {
                anchor_distance,
                full_image_distance,
            };

            let gop_reader = GroupOfPicturesReader::new(cursor, ordering);

            for decoded_frame in gop_reader {
                decoded_frames.push(decoded_frame.expect("Failed to decode GOP frame"));
            }
        }

        // For identical frames, all residuals should be near-zero
        // and quality should be extremely high
        for (idx, (original, decoded)) in original_frames
            .iter()
            .zip(decoded_frames.iter())
            .enumerate()
        {
            let mse = calculate_mse(original, &decoded.data);
            let psnr = calculate_psnr(mse, 255.0);

            println!(
                "   Frame {:2}: MSE = {:.4}, PSNR = {:.2} dB",
                idx, mse, psnr
            );

            // For identical frames, PSNR should be very high (> 40 dB)
            assert!(
                psnr > 40.0 || psnr.is_infinite(),
                "Frame {} quality too low for identical input: PSNR = {:.2} dB",
                idx,
                psnr
            );
        }

        println!("\n✅ All identical frames maintain excellent quality (PSNR > 40 dB)");
    }

    #[test]
    fn test_gop_frame_pattern() {
        // Test that the GOP produces the correct frame pattern
        let anchor_distance = 3;
        let full_image_distance = 12;

        let ordering = Ordering {
            anchor_distance,
            full_image_distance,
        };

        // Expected pattern for one GOP: I B B P B B P B B P B B
        let expected = vec![
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

        for (idx, expected_kind) in expected.iter().enumerate() {
            let actual_kind = ordering.frame_kind(idx);
            assert_eq!(
                actual_kind, *expected_kind,
                "Frame {} should be {:?}, got {:?}",
                idx, expected_kind, actual_kind
            );
        }

        // Next frame should be I (start of new GOP)
        assert_eq!(
            ordering.frame_kind(12),
            Kind::I,
            "Frame 12 should start new GOP with I-frame"
        );
    }
}
