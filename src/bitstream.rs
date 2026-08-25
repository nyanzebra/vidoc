use std::{
    io::{Read, Write},
    mem::size_of,
};

use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Unsigned};

use crate::{Error, Result};

/// Number of bytes read from the underlying stream in one refill.
///
/// 64 KiB is large enough to amortize Read::read calls while remaining small
/// enough to be friendly to codec pipelines and cache behavior.
const INPUT_BUFFER_SIZE: usize = 64 * 1024;

/// Number of bytes accumulated before the writer talks to the underlying
/// writer.
///
/// This is intentionally independent of the bit accumulator. The bit
/// accumulator only contains the currently incomplete byte.
const OUTPUT_BUFFER_SIZE: usize = 64 * 1024;

const BYTE_BITS: usize = 8;
const MAX_BITS: usize = u128::BITS as usize;

/// A buffered MSB-first bitstream reader.
///
/// Bit numbering is codec-style/network-style:
///
/// ```text
/// write_bits(0b101, 3)
///             ^^^
///             written first -> 1 0 1
/// ```
///
/// Bytes are therefore emitted/read in big-endian bit order.
///
/// The reader never silently returns a partially decoded field. If a field
/// begins successfully but the stream ends before all requested bits arrive,
/// `Error::UnexpectedEndOfStream` is returned.
pub struct BitStreamReader<R>
where
    R: Read,
{
    stream: R,

    /// Input byte buffer.
    input: [u8; INPUT_BUFFER_SIZE],

    /// Next unread byte in `input`.
    input_pos: usize,

    /// One-past-last valid byte in `input`.
    input_end: usize,

    /// Accumulated bits waiting to be consumed.
    ///
    /// The valid bits occupy the least-significant `bit_count` bits, with the
    /// oldest bit at the highest position within that range.
    bit_buffer: u128,

    /// Number of valid bits in `bit_buffer`.
    bit_count: usize,

    /// True after the underlying reader has returned EOF.
    eof: bool,
}

impl<R> BitStreamReader<R>
where
    R: Read,
{
    pub fn new(stream: R) -> Self {
        Self {
            stream,
            input: [0; INPUT_BUFFER_SIZE],
            input_pos: 0,
            input_end: 0,
            bit_buffer: 0,
            bit_count: 0,
            eof: false,
        }
    }

    /// Creates a reader and performs an initial input-buffer refill.
    ///
    /// This method is retained for compatibility with the existing codec.
    /// Unlike the old implementation, it does not attempt to load the entire
    /// stream into a BitVec.
    pub fn new_with_data(mut stream: R) -> Result<Self> {
        let mut reader = Self::new(stream);
        reader.fill_input_buffer()?;
        stream = reader.stream;
        reader.stream = stream;
        Ok(reader)
    }

    /// Reads exactly `bytes` codec bytes.
    ///
    /// If the stream ends part way through the request, an error is returned
    /// rather than returning a shorter vector.
    pub fn read_slice(&mut self, bytes: usize) -> Result<Vec<u8>> {
        if bytes == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(bytes);

        // Fast path when the bitstream is byte aligned.
        if self.bit_count == 0 {
            result.resize(bytes, 0);
            self.read_aligned_exact(&mut result)?;
            return Ok(result);
        }

        // Unaligned byte reads intentionally cross byte boundaries.
        for byte in &mut result {
            *byte = self.read::<u8>()?.ok_or(Error::UnexpectedEndOfStream)?;
        }

        Ok(result)
    }

    /// Reads exactly `N` codec bytes without heap allocation.
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0u8; N];

        if N == 0 {
            return Ok(result);
        }

        if self.bit_count == 0 {
            self.read_aligned_exact(&mut result)?;
            return Ok(result);
        }

        for byte in &mut result {
            *byte = self.read::<u8>()?.ok_or(Error::UnexpectedEndOfStream)?;
        }

        Ok(result)
    }

    /// Reads an unsigned integer using exactly its native width.
    ///
    /// This method is retained for compatibility. For values that are part of
    /// the on-disk codec format, prefer the explicit-width methods such as
    /// `read_u32` and `read_u64`.
    pub fn read<T>(&mut self) -> Result<Option<T>>
    where
        T: PrimInt + FromPrimitive + Unsigned,
    {
        let bits = size_of::<T>()
            .checked_mul(BYTE_BITS)
            .ok_or(Error::InvalidData)?;

        let value = match self.read_bits(bits)? {
            Some(value) => value,
            None => return Ok(None),
        };

        T::from_u128(value).ok_or(Error::InvalidData).map(Some)
    }

    #[inline]
    pub fn read_u8(&mut self) -> Result<Option<u8>> {
        self.read::<u8>()
    }

    #[inline]
    pub fn read_u16(&mut self) -> Result<Option<u16>> {
        self.read::<u16>()
    }

    #[inline]
    pub fn read_u32(&mut self) -> Result<Option<u32>> {
        self.read::<u32>()
    }

    #[inline]
    pub fn read_u64(&mut self) -> Result<Option<u64>> {
        self.read::<u64>()
    }

    #[inline]
    pub fn read_u128(&mut self) -> Result<Option<u128>> {
        self.read::<u128>()
    }

    pub fn read_bit(&mut self) -> Result<Option<bool>> {
        Ok(self.read_bits(1)?.map(|value| value != 0))
    }

    /// Reads `len` MSB-first bits.
    ///
    /// `None` means the stream was already at EOF before any bit belonging to
    /// this field was available.
    ///
    /// If some bits were available but the complete field could not be read,
    /// `UnexpectedEndOfStream` is returned. This is important for a codec:
    /// silently returning a partial Rice coefficient or frame header can turn
    /// corruption into plausible-but-wrong decoded video.
    #[inline]
    pub(crate) fn read_bits(&mut self, len: usize) -> Result<Option<u128>> {
        if len == 0 {
            return Ok(None);
        }

        if len > MAX_BITS {
            return Err(Error::InvalidData);
        }

        while self.bit_count < len {
            match self.read_input_byte()? {
                Some(byte) => {
                    self.bit_buffer = (self.bit_buffer << BYTE_BITS) | u128::from(byte);
                    self.bit_count += BYTE_BITS;
                }
                None => {
                    if self.bit_count == 0 {
                        return Ok(None);
                    }

                    // We consumed no part of this field. The partial bits are
                    // retained so the reader remains in a well-defined state.
                    return Err(Error::UnexpectedEndOfStream);
                }
            }
        }

        let shift = self.bit_count - len;

        let value = if len == MAX_BITS {
            self.bit_buffer
        } else {
            (self.bit_buffer >> shift) & mask_u128(len)
        };

        self.bit_count -= len;

        if self.bit_count == 0 {
            self.bit_buffer = 0;
        } else {
            self.bit_buffer &= mask_u128(self.bit_count);
        }

        Ok(Some(value))
    }

    /// Reads exactly `len` bits.
    ///
    /// Unlike `read_bits`, this is useful in code where EOF is always an
    /// error, including codec headers.
    pub(crate) fn read_bits_exact(&mut self, len: usize) -> Result<u128> {
        self.read_bits(len)?.ok_or(Error::UnexpectedEndOfStream)
    }

    /// Discards the remaining bits in the current byte.
    ///
    /// This is intentionally destructive and should only be used at explicit
    /// codec byte-alignment points.
    pub fn align_to_byte(&mut self) -> Result<()> {
        let remainder = self.bit_count % BYTE_BITS;

        if remainder != 0 {
            let _ = self.read_bits_exact(remainder)?;
        }

        Ok(())
    }

    /// Counts zero bits until the first one bit.
    ///
    /// The terminating one is consumed.
    pub fn count_leading_zeros(&mut self) -> Result<usize> {
        let mut count = 0usize;

        loop {
            match self.read_bit()? {
                Some(true) => return Ok(count),
                Some(false) => {
                    count = count.checked_add(1).ok_or(Error::InvalidData)?;
                }
                None => {
                    // A unary code with no terminating one is malformed.
                    return Err(Error::UnexpectedEndOfStream);
                }
            }
        }
    }

    /// Returns the underlying reader.
    ///
    /// Any bits already buffered have necessarily been read from `stream`.
    /// Callers that need to continue consuming the underlying object should
    /// therefore continue through this BitStreamReader instead of using the
    /// returned reader directly.
    pub fn into_inner(self) -> R {
        self.stream
    }

    /// Returns the number of unread buffered bits.
    #[inline]
    pub fn buffered_bits(&self) -> usize {
        self.bit_count
    }

    /// Returns whether the reader currently sits on a byte boundary.
    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_count % BYTE_BITS == 0
    }

    fn read_input_byte(&mut self) -> Result<Option<u8>> {
        if self.input_pos < self.input_end {
            let byte = self.input[self.input_pos];
            self.input_pos += 1;
            return Ok(Some(byte));
        }

        if self.eof {
            return Ok(None);
        }

        self.fill_input_buffer()?;

        if self.input_pos < self.input_end {
            let byte = self.input[self.input_pos];
            self.input_pos += 1;
            Ok(Some(byte))
        } else {
            Ok(None)
        }
    }

    fn fill_input_buffer(&mut self) -> Result<()> {
        self.input_pos = 0;
        self.input_end = 0;

        loop {
            match self.stream.read(&mut self.input)? {
                0 => {
                    self.eof = true;
                    return Ok(());
                }
                count => {
                    self.input_end = count;
                    return Ok(());
                }
            }
        }
    }

    fn read_aligned_exact(&mut self, output: &mut [u8]) -> Result<()> {
        debug_assert_eq!(self.bit_count, 0);

        let mut written = 0usize;

        while written < output.len() {
            if self.input_pos < self.input_end {
                let available = self.input_end - self.input_pos;
                let wanted = output.len() - written;
                let count = available.min(wanted);

                output[written..written + count]
                    .copy_from_slice(&self.input[self.input_pos..self.input_pos + count]);

                self.input_pos += count;
                written += count;
                continue;
            }

            if self.eof {
                return Err(Error::UnexpectedEndOfStream);
            }

            self.fill_input_buffer()?;

            if self.input_pos == self.input_end && self.eof {
                return Err(Error::UnexpectedEndOfStream);
            }
        }

        Ok(())
    }
}

/// A buffered MSB-first bitstream writer.
///
/// The writer keeps only one incomplete byte in the bit accumulator and
/// accumulates complete bytes in a large output buffer before writing them to
/// the underlying stream.
pub struct BitStreamWriter<W>
where
    W: Write,
{
    stream: W,

    /// Complete bytes waiting to be written.
    output: Vec<u8>,

    /// Bits belonging to the incomplete current byte.
    pending_byte: u8,

    /// Number of valid bits in `pending_byte`.
    ///
    /// The valid bits occupy the high bits of the byte.
    pending_bits: usize,
}

impl<W> BitStreamWriter<W>
where
    W: Write,
{
    pub fn new(stream: W) -> Self {
        Self {
            stream,
            output: Vec::with_capacity(OUTPUT_BUFFER_SIZE),
            pending_byte: 0,
            pending_bits: 0,
        }
    }

    /// Writes `count` zero bits.
    ///
    /// This is the hot path for Rice coding, so it avoids invoking the generic
    /// bit writer once for every zero.
    pub fn write_zeros(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        // Finish a partially-filled byte first.
        if self.pending_bits != 0 {
            let available = BYTE_BITS - self.pending_bits;
            let take = count.min(available);

            self.pending_byte <<= take;
            self.pending_bits += take;

            if self.pending_bits == BYTE_BITS {
                self.push_output_byte(self.pending_byte)?;
                self.pending_byte = 0;
                self.pending_bits = 0;
            }

            if take == count {
                return Ok(());
            }
        }

        let mut remaining = count
            - if self.pending_bits == 0 {
                0
            } else {
                // The branch above can only leave us here if the entire count was
                // consumed, so this is unreachable.
                0
            };

        // At this point the writer is byte-aligned.
        //
        // Avoid a huge allocation for a massive unary run by emitting in
        // bounded chunks.
        let zero_chunk = [0u8; 4096];

        while remaining >= zero_chunk.len() * BYTE_BITS {
            self.output.extend_from_slice(&zero_chunk);
            self.flush_output_if_needed()?;
            remaining -= zero_chunk.len() * BYTE_BITS;
        }

        if remaining > 0 {
            let full_bytes = remaining / BYTE_BITS;

            if full_bytes > 0 {
                self.output.extend_from_slice(&zero_chunk[..full_bytes]);
                self.flush_output_if_needed()?;
                remaining -= full_bytes * BYTE_BITS;
            }

            if remaining > 0 {
                self.pending_byte = 0;
                self.pending_bits = remaining;
            }
        }

        Ok(())
    }

    /// Pads with zero bits until the next byte boundary.
    pub fn align_to_byte(&mut self) -> Result<()> {
        if self.pending_bits == 0 {
            return Ok(());
        }

        let padding = BYTE_BITS - self.pending_bits;

        self.pending_byte <<= padding;
        self.push_output_byte(self.pending_byte)?;

        self.pending_byte = 0;
        self.pending_bits = 0;

        Ok(())
    }

    pub fn write_bit(&mut self, bit: bool) -> Result<()> {
        self.write_bits(if bit { 1u128 } else { 0u128 }, 1)
    }

    /// Writes an unsigned integer using exactly its native width.
    ///
    /// This is retained for compatibility. Codec structures should prefer
    /// explicit-width methods (`write_u32`, `write_u64`, etc.).
    pub fn write<T>(&mut self, value: T) -> Result<()>
    where
        T: PrimInt + ToPrimitive + Unsigned,
    {
        let bits = size_of::<T>()
            .checked_mul(BYTE_BITS)
            .ok_or(Error::InvalidData)?;

        let value = value.to_u128().ok_or(Error::InvalidData)?;

        self.write_bits(value, bits)
    }

    #[inline]
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.write::<u8>(value)
    }

    #[inline]
    pub fn write_u16(&mut self, value: u16) -> Result<()> {
        self.write::<u16>(value)
    }

    #[inline]
    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        self.write::<u32>(value)
    }

    #[inline]
    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        self.write::<u64>(value)
    }

    #[inline]
    pub fn write_u128(&mut self, value: u128) -> Result<()> {
        self.write::<u128>(value)
    }

    pub fn write_slice(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }

        if self.pending_bits == 0 {
            self.write_aligned_bytes(bytes)
        } else {
            for &byte in bytes {
                self.write_bits(u128::from(byte), BYTE_BITS)?;
            }

            Ok(())
        }
    }

    /// Writes a value containing between 1 and 128 bits.
    ///
    /// Values are always interpreted MSB-first. Bits above `len` are ignored.
    ///
    /// A request larger than 128 bits is rejected rather than pretending that
    /// a u128 contains more data than it actually does.
    #[inline]
    pub(crate) fn write_bits<T>(&mut self, value: T, len: usize) -> Result<()>
    where
        T: Into<u128>,
    {
        if len == 0 {
            return Ok(());
        }

        if len > MAX_BITS {
            return Err(Error::InvalidData);
        }

        let mut remaining = len;
        let mut value = value.into();

        // Write from the most-significant requested bits toward the least
        // significant bits. Chunks of 56 avoid edge cases around shifting a
        // u128 by 128.
        while remaining > 0 {
            let chunk = remaining.min(56);
            let shift = remaining - chunk;

            let part = (value >> shift) & mask_u128(chunk);

            self.append_bits(part, chunk)?;

            remaining -= chunk;

            if shift == 0 {
                break;
            }

            value &= mask_u128(shift);
        }

        Ok(())
    }

    /// Writes bytes directly when byte-aligned.
    ///
    /// Unlike the previous implementation this still goes through the
    /// BitStreamWriter's output buffer, so ordering relative to buffered bits
    /// is guaranteed.
    pub fn write_all_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.align_to_byte()?;
        self.write_aligned_bytes(bytes)
    }

    /// Flushes all pending bits.
    ///
    /// If the stream is not byte-aligned, the final byte is padded with zero
    /// bits. This is the codec's canonical padding representation.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending_bits != 0 {
            let padding = BYTE_BITS - self.pending_bits;

            let byte = self.pending_byte << padding;

            self.push_output_byte(byte)?;

            self.pending_byte = 0;
            self.pending_bits = 0;
        }

        self.flush_output()?;
        self.stream.flush()?;

        Ok(())
    }

    /// Finishes the writer and returns the underlying stream.
    ///
    /// Unlike `into_inner`, this guarantees that all encoded data has reached
    /// the underlying writer.
    pub fn finish(mut self) -> Result<W> {
        self.flush()?;
        Ok(self.stream)
    }

    /// Returns the underlying writer without flushing.
    ///
    /// This method is retained for compatibility with the existing API.
    /// Prefer `finish()` for codec output.
    pub fn into_inner(self) -> W {
        self.stream
    }

    #[inline]
    pub fn pending_bits(&self) -> usize {
        self.pending_bits
    }

    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.pending_bits == 0
    }

    fn append_bits(&mut self, value: u128, count: usize) -> Result<()> {
        debug_assert!(count > 0);
        debug_assert!(count <= 56);

        let value = value & mask_u128(count);

        if self.pending_bits == 0 && count == BYTE_BITS {
            self.push_output_byte(value as u8)?;
            return Ok(());
        }

        let available = BYTE_BITS - self.pending_bits;

        if count <= available {
            self.pending_byte = (self.pending_byte << count) | value as u8;
            self.pending_bits += count;

            if self.pending_bits == BYTE_BITS {
                self.push_output_byte(self.pending_byte)?;
                self.pending_byte = 0;
                self.pending_bits = 0;
            }

            return Ok(());
        }

        // Fill the current byte with the high part of `value`.
        let high_count = available;
        let low_count = count - high_count;

        let high = value >> low_count;

        self.pending_byte = (self.pending_byte << high_count) | high as u8;

        self.push_output_byte(self.pending_byte)?;

        self.pending_byte = (value & mask_u128(low_count)) as u8;
        self.pending_bits = low_count;

        Ok(())
    }

    fn write_aligned_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        debug_assert_eq!(self.pending_bits, 0);

        let mut remaining = bytes;

        while !remaining.is_empty() {
            let available = OUTPUT_BUFFER_SIZE - self.output.len();

            if remaining.len() <= available {
                self.output.extend_from_slice(remaining);
                break;
            }

            if available > 0 {
                self.output.extend_from_slice(&remaining[..available]);
                remaining = &remaining[available..];
            }

            self.flush_output()?;
        }

        self.flush_output_if_needed()
    }

    #[inline]
    fn push_output_byte(&mut self, byte: u8) -> Result<()> {
        self.output.push(byte);
        self.flush_output_if_needed()
    }

    #[inline]
    fn flush_output_if_needed(&mut self) -> Result<()> {
        if self.output.len() >= OUTPUT_BUFFER_SIZE {
            self.flush_output()?;
        }

        Ok(())
    }

    fn flush_output(&mut self) -> Result<()> {
        if self.output.is_empty() {
            return Ok(());
        }

        self.stream.write_all(&self.output)?;
        self.output.clear();

        Ok(())
    }
}

#[inline]
fn mask_u128(bits: usize) -> u128 {
    debug_assert!(bits <= MAX_BITS);

    if bits == 0 {
        0
    } else if bits == MAX_BITS {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_single_bits() {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            writer.write_bit(true).unwrap();
            writer.write_bit(false).unwrap();
            writer.write_bit(true).unwrap();
            writer.write_bit(true).unwrap();
            writer.write_bit(false).unwrap();

            writer.flush().unwrap();
        }

        assert_eq!(buffer, vec![0b10110_000]);

        let mut reader = BitStreamReader::new(buffer.as_slice());

        assert_eq!(reader.read_bit().unwrap(), Some(true));
        assert_eq!(reader.read_bit().unwrap(), Some(false));
        assert_eq!(reader.read_bit().unwrap(), Some(true));
        assert_eq!(reader.read_bit().unwrap(), Some(true));
        assert_eq!(reader.read_bit().unwrap(), Some(false));
    }

    #[test]
    fn roundtrip_various_widths() {
        let cases = [
            (0b1u128, 1usize),
            (0b10, 2),
            (0b101, 3),
            (0b101101, 6),
            (0xabu128, 8),
            (0x1234, 16),
            (0x12345678, 32),
            (0x123456789abcdef0, 64),
            (u128::MAX, 128),
        ];

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            for &(value, bits) in &cases {
                writer.write_bits(value, bits).unwrap();
            }

            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &(expected, bits) in &cases {
            assert_eq!(reader.read_bits(bits).unwrap(), Some(expected));
        }
    }

    #[test]
    fn crosses_every_possible_byte_boundary() {
        for width in 1..=128 {
            let value = if width == 128 {
                u128::MAX
            } else {
                mask_u128(width) ^ (mask_u128(width) >> 1)
            };

            let mut buffer = Vec::new();

            {
                let mut writer = BitStreamWriter::new(&mut buffer);
                writer.write_bits(value, width).unwrap();
                writer.flush().unwrap();
            }

            let mut reader = BitStreamReader::new(buffer.as_slice());

            assert_eq!(
                reader.read_bits(width).unwrap(),
                Some(value),
                "failed at width {width}"
            );
        }
    }

    #[test]
    fn mixed_width_roundtrip() {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            writer.write_bits(0b101, 3).unwrap();
            writer.write_u8(0xab).unwrap();
            writer.write_bits(0b110011, 6).unwrap();
            writer.write_u16(0xcdef).unwrap();
            writer.write_bits(0b1, 1).unwrap();

            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        assert_eq!(reader.read_bits(3).unwrap(), Some(0b101));
        assert_eq!(reader.read_u8().unwrap(), Some(0xab));
        assert_eq!(reader.read_bits(6).unwrap(), Some(0b110011));
        assert_eq!(reader.read_u16().unwrap(), Some(0xcdef));
        assert_eq!(reader.read_bit().unwrap(), Some(true));
    }

    #[test]
    fn aligned_bulk_write_and_read() {
        let data: Vec<u8> = (0..=255).collect();
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            writer.write_all_bytes(&data).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer, data);

        let mut reader = BitStreamReader::new(buffer.as_slice());

        let decoded = reader.read_slice(data.len()).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn unaligned_bulk_write_and_read() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            writer.write_bits(0b101, 3).unwrap();
            writer.write_slice(&data).unwrap();
            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        assert_eq!(reader.read_bits(3).unwrap(), Some(0b101));
        assert_eq!(reader.read_slice(data.len()).unwrap(), data);
    }

    #[test]
    fn zero_runs() {
        for count in 0..=100_000 {
            let mut buffer = Vec::new();

            {
                let mut writer = BitStreamWriter::new(&mut buffer);

                writer.write_zeros(count).unwrap();
                writer.write_bit(true).unwrap();
                writer.flush().unwrap();
            }

            let mut reader = BitStreamReader::new(buffer.as_slice());

            assert_eq!(
                reader.count_leading_zeros().unwrap(),
                count,
                "failed at zero count {count}"
            );
        }
    }

    #[test]
    fn partial_field_is_an_error() {
        let data = [0b1010_0000];

        let mut reader = BitStreamReader::new(data.as_slice());

        assert_eq!(reader.read_bits(3).unwrap(), Some(0b101));

        let error = reader.read_bits(8).unwrap_err();

        assert!(matches!(error, Error::UnexpectedEndOfStream));
    }

    #[test]
    fn eof_is_none_when_no_bits_remain() {
        let mut reader = BitStreamReader::new(&[][..]);

        assert_eq!(reader.read_bit().unwrap(), None);
        assert_eq!(reader.read_bits(32).unwrap(), None);
    }

    #[test]
    fn empty_slice_is_valid() {
        let mut writer = BitStreamWriter::new(Vec::<u8>::new());
        writer.write_slice(&[]).unwrap();
        writer.flush().unwrap();

        let mut reader = BitStreamReader::new(&[][..]);

        assert_eq!(reader.read_slice(0).unwrap(), Vec::<u8>::new());
        assert_eq!(reader.read_array::<0>().unwrap(), []);
    }

    #[test]
    fn byte_alignment_discards_padding() {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            writer.write_bits(0b101, 3).unwrap();
            writer.align_to_byte().unwrap();
            writer.write_u8(0xab).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer, vec![0b1010_0000, 0xab]);

        let mut reader = BitStreamReader::new(buffer.as_slice());

        assert_eq!(reader.read_bits(3).unwrap(), Some(0b101));
        reader.align_to_byte().unwrap();
        assert_eq!(reader.read_u8().unwrap(), Some(0xab));
    }

    #[test]
    fn max_width_roundtrip() {
        let value = 0x123456789abcdef00123456789abcdefu128;

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            writer.write_u128(value).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer.len(), 16);

        let mut reader = BitStreamReader::new(buffer.as_slice());

        assert_eq!(reader.read_u128().unwrap(), Some(value));
        assert_eq!(reader.read_bit().unwrap(), None);
    }

    #[test]
    fn writer_rejects_more_than_128_bits() {
        let mut writer = BitStreamWriter::new(Vec::<u8>::new());

        assert!(matches!(
            writer.write_bits(0u128, 129),
            Err(Error::InvalidData)
        ));
    }

    #[test]
    fn writer_finish_flushes() {
        let writer = BitStreamWriter::new(Vec::<u8>::new());

        let mut writer = writer;
        writer.write_u32(0x12345678).unwrap();

        let data = writer.finish().unwrap();

        assert_eq!(data, 0x12345678u32.to_be_bytes());
    }

    #[test]
    fn big_stream_roundtrip() {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            for i in 0..1_000_000u32 {
                writer.write_u32(i).unwrap();
            }

            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for i in 0..1_000_000u32 {
            assert_eq!(reader.read_u32().unwrap(), Some(i));
        }

        assert_eq!(reader.read_u8().unwrap(), None);
    }

    #[test]
    fn reader_handles_short_reads() {
        struct ShortReader {
            data: Vec<u8>,
            pos: usize,
        }

        impl Read for ShortReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.pos >= self.data.len() {
                    return Ok(0);
                }

                let count = 1.min(buf.len());

                buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);

                self.pos += count;

                Ok(count)
            }
        }

        let input = vec![0x12, 0x34, 0x56, 0x78];

        let mut reader = BitStreamReader::new(ShortReader {
            data: input,
            pos: 0,
        });

        assert_eq!(reader.read_u32().unwrap(), Some(0x12345678));
        assert_eq!(reader.read_u8().unwrap(), None);
    }

    #[test]
    fn reader_does_not_lose_partial_bits() {
        let data = [0b1010_1010, 0b1100_0000];

        let mut reader = BitStreamReader::new(data.as_slice());

        assert_eq!(reader.read_bits(3).unwrap(), Some(0b101));
        assert_eq!(reader.read_bits(5).unwrap(), Some(0b01010));
        assert_eq!(reader.read_bits(2).unwrap(), Some(0b11));
    }

    #[test]
    fn all_byte_values() {
        let data: Vec<u8> = (0..=255).collect();
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            writer.write_slice(&data).unwrap();
            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for expected in data {
            assert_eq!(reader.read_u8().unwrap(), Some(expected));
        }
    }
}
