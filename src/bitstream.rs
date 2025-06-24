use std::{
    io::{Read, Write},
    mem::size_of,
};

use num_traits::{FromPrimitive, PrimInt, ToPrimitive, Unsigned};

use crate::{bitvec::BitVec, Error, Result};

const BYTE_SIZE: usize = u8::BITS as usize;

pub struct BitStreamReader<R>
where
    R: Read,
{
    stream: R,
    bitvec: BitVec,
}

impl<R> BitStreamReader<R>
where
    R: Read,
{
    pub fn new(stream: R) -> Self {
        Self {
            stream,
            bitvec: BitVec::new(),
        }
    }

    pub fn new_with_data(mut stream: R) -> Result<Self> {
        let mut bitvec = BitVec::new();
        bitvec.read_from_stream(&mut stream)?;
        Ok(Self { stream, bitvec })
    }

    pub fn read_slice(&mut self, bytes: usize) -> Result<Vec<u8>> {
        let mut res = vec![];
        for _ in 0..bytes {
            if let Some(byte) = self.read::<u8>()? {
                res.push(byte);
            } else {
                return Err(Error::UnexpectedEndOfStream);
            }
        }
        Ok(res)
    }

    pub fn read<T>(&mut self) -> Result<Option<T>>
    where
        T: PrimInt + FromPrimitive + Unsigned,
    {
        match self.read_bits(size_of::<T>() * BYTE_SIZE)? {
            Some(value) => Ok(T::from_u128(value)),
            None => Ok(None),
        }
    }

    pub fn read_bit(&mut self) -> Result<Option<bool>> {
        self.read_bits(1).map(|opt| opt.map(|x| (x & 1) == 1))
    }

    #[inline]
    pub(crate) fn read_bits(&mut self, len: usize) -> Result<Option<u128>> {
        if len == 0 {
            return Ok(None);
        }

        assert!(len <= u128::BITS as usize, "len is too large");

        // Ensure we have enough data
        if self.bitvec.len() >= len {
            Ok(self.bitvec.pop_bits(len as u8))
        } else {
            // We need more data than available in buffer
            let available_bits = self.bitvec.len();

            // First, get whatever bits we have in the current buffer
            let partial_bits = if available_bits > 0 {
                self.bitvec.pop_bits(available_bits as u8).unwrap_or(0)
            } else {
                0
            };

            // Clear and refill the buffer
            self.bitvec.clear();
            self.bitvec.read_from_stream(&mut self.stream)?;

            // Calculate how many more bits we need
            let remaining_bits_needed = len - available_bits;

            if self.bitvec.len() >= remaining_bits_needed {
                // Get the remaining bits from the new buffer
                let remaining_bits = self
                    .bitvec
                    .pop_bits(remaining_bits_needed as u8)
                    .unwrap_or(0);

                // Combine: partial_bits were the high-order bits we read first,
                // remaining_bits are the low-order bits we need to append
                // Handle overflow: if remaining_bits_needed >= 128, just return remaining_bits
                let combined = if remaining_bits_needed >= u128::BITS as usize {
                    remaining_bits
                } else {
                    (partial_bits << remaining_bits_needed) | remaining_bits
                };

                Ok(Some(combined))
            } else {
                // Still not enough data even after refill - return what we have

                if available_bits > 0 {
                    Ok(Some(partial_bits))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub fn into_inner(self) -> R {
        self.stream
    }

    /// Align the reader to the next byte boundary by discarding any remaining bits in the current
    /// byte
    pub fn align_to_byte(&mut self) -> Result<()> {
        let current_bit_pos = self.bitvec.bit_position_in_byte();
        if current_bit_pos != 0 {
            let bits_to_skip = BYTE_SIZE - current_bit_pos;
            self.read_bits(bits_to_skip)?;
        }
        Ok(())
    }

    /// Efficiently count leading zeros until finding a 1 bit (for Rice decoding)
    /// Returns the count of leading zeros found before a 1 bit
    pub fn count_leading_zeros(&mut self) -> Result<usize> {
        let mut count = 0;

        loop {
            if self.bitvec.is_empty() {
                self.bitvec.read_from_stream(&mut self.stream)?;
            }

            if let Some(pos) = self.bitvec.first_one() {
                count += pos;
                break;
            } else {
                count += self.bitvec.len();
                self.bitvec.clear();
            }
        }

        Ok(count)
    }
}

pub struct BitStreamWriter<W>
where
    W: Write,
{
    stream: W,
    bitvec: BitVec,
}

impl<W> BitStreamWriter<W>
where
    W: Write,
{
    pub fn new(stream: W) -> Self {
        Self {
            stream,
            bitvec: BitVec::new(),
        }
    }

    /// Write zeros efficiently using buffer fill and flush strategy
    pub fn write_zeros(&mut self, count: usize) -> Result<()> {
        let mut remaining = count;

        while remaining > 0 {
            // Write zeros in chunks of up to 128 bits
            let chunk_size = remaining.min(128);

            // Use the existing write_bits mechanism which handles overflow automatically
            let (remaining_value, unwritten_bits) = self.bitvec.push_bits(0u128, chunk_size as u8);
            if unwritten_bits > 0 {
                // Buffer was full, flush and write remainder
                self.bitvec.flush_to_stream(&mut self.stream)?;
                self.bitvec.push_bits(remaining_value, unwritten_bits);
            }

            remaining -= chunk_size;
        }
        Ok(())
    }

    pub fn align_to_byte(&mut self) -> Result<()> {
        let current_bit_pos = self.bitvec.bit_position_in_byte();
        if current_bit_pos != 0 {
            let bits_to_pad = BYTE_SIZE - current_bit_pos;
            self.write_zeros(bits_to_pad)?;
        }
        Ok(())
    }

    pub fn write_bit(&mut self, bit: bool) -> Result<()> {
        self.write_bits(bit, 1)
    }

    pub fn write<T>(&mut self, val: T) -> Result<()>
    where
        T: PrimInt + ToPrimitive + Unsigned,
    {
        let bits_needed = size_of::<T>() * 8;
        let val_as_u128 = val.to_u128().ok_or(Error::InvalidData)?;
        self.write_bits(val_as_u128, bits_needed)
    }

    pub fn write_slice(&mut self, bytes: &[u8]) -> Result<()> {
        for byte in bytes {
            self.write(*byte)?;
        }
        Ok(())
    }

    // Write raw bits with a specific length
    #[inline]
    pub(crate) fn write_bits<T>(&mut self, value: T, len: usize) -> Result<()>
    where
        T: Into<u128>,
    {
        let value_u128 = value.into();

        if len == 0 {
            return Ok(());
        }

        // For zeros, use the optimized write_zeros method
        if value_u128 == 0 && len > 128 {
            return self.write_zeros(len);
        }

        // For lengths > 128, we need to split the write into chunks
        if len > 128 {
            // For large non-zero writes, write in chunks
            let mut remaining_len = len;
            let mut current_value = value_u128;

            while remaining_len > 0 {
                let chunk_size = std::cmp::min(remaining_len, 128);

                // Extract the lowest chunk_size bits
                let chunk_value = current_value & ((1u128 << chunk_size) - 1);

                // Write this chunk
                let (remaining_value, unwritten_bits) =
                    self.bitvec.push_bits(chunk_value, chunk_size as u8);
                if unwritten_bits > 0 {
                    self.bitvec.flush_to_stream(&mut self.stream)?;
                    self.bitvec.push_bits(remaining_value, unwritten_bits);
                }

                // Move to next chunk
                current_value >>= chunk_size;
                remaining_len -= chunk_size;
            }

            return Ok(());
        }

        // Write the bits for <= 128 bit values
        let (remaining_value, unwritten_bits) = self.bitvec.push_bits(value_u128, len as u8);
        if unwritten_bits > 0 {
            self.bitvec.flush_to_stream(&mut self.stream)?;
            self.bitvec.push_bits(remaining_value, unwritten_bits);
        }

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.bitvec.flush(&mut self.stream)
    }

    pub fn into_inner(self) -> W {
        self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_write_small() {
        let stream = Vec::new();
        let mut writer = BitStreamWriter::new(stream);
        let value = 0b10110111u32;
        writer.write_bits(value, 8).expect("write_bits");
        writer.flush().expect("flush");

        let inner = writer.into_inner();
        assert!(!inner.is_empty());

        let mut reader = BitStreamReader::new(inner.as_slice());
        let read_bits = reader
            .read_bits(8)
            .expect("read bits")
            .expect("should have data");
        assert_eq!(value as u128, read_bits);
    }

    #[test]
    fn read_write_large() {
        let stream = Vec::new();
        let mut writer = BitStreamWriter::new(stream);
        let value = 0b101101110001110101u32;
        writer.write_bits(value, 18).expect("write_bits");
        writer.flush().expect("flush");

        let inner = writer.into_inner();
        assert!(!inner.is_empty());

        let mut reader = BitStreamReader::new(inner.as_slice());
        let read_bits = reader
            .read_bits(18)
            .expect("read bits")
            .expect("should have data");
        assert_eq!(value as u128, read_bits);
    }

    #[test]
    fn read_write_large2() {
        let stream = Vec::new();
        let mut writer = BitStreamWriter::new(stream);
        let value = 0b0000000000000000001u32;
        writer.write_bits(value, 19).expect("write_bits");
        writer.flush().expect("flush");

        let inner = writer.into_inner();
        assert!(!inner.is_empty());

        let mut reader = BitStreamReader::new(inner.as_slice());
        let read_bits = reader
            .read_bits(19)
            .expect("read bits")
            .expect("should have data");
        assert_eq!(value as u128, read_bits);
    }

    #[test]
    fn read_write_really_large() {
        let stream = Vec::new();
        let mut writer = BitStreamWriter::new(stream);
        writer.write_bits(1u32, 127).expect("write_bits");
        writer.flush().expect("flush");

        let inner = writer.into_inner();

        let mut reader = BitStreamReader::new(inner.as_slice());
        let mut bit_count = 0;
        while let Some(bit) = reader.read_bit().unwrap() {
            if !bit {
                bit_count += 1;
            } else {
                break;
            }
        }
        assert_eq!(bit_count, 127 - 1);
    }

    #[test]
    fn test_write_read_single_bit() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        writer.write_bit(true).expect("write_bit");
        writer.write_bit(false).expect("write_bit");
        writer.write_bit(true).expect("write_bit");
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        assert_eq!(reader.read_bit().expect("read bit"), Some(true));
        assert_eq!(reader.read_bit().expect("read bit"), Some(false));
        assert_eq!(reader.read_bit().expect("read bit"), Some(true));
    }

    #[test]
    fn test_write_read_u8() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_values = [0x00, 0xFF, 0xAA, 0x55, 0x12, 0x34, 0x56, 0x78];

        for &value in &test_values {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &test_values {
            assert_eq!(reader.read::<u8>().expect("read u8"), Some(expected));
        }
    }

    #[test]
    fn test_write_read_u16() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_values = [
            0x0000, 0xFFFF, 0xAAAA, 0x5555, 0x1234, 0x5678, 0x9ABC, 0xDEF0,
        ];

        for &value in &test_values {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &test_values {
            assert_eq!(reader.read::<u16>().expect("read u16"), Some(expected));
        }
    }

    #[test]
    fn test_write_read_u32() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_values = [
            0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x12345678, 0x9ABCDEF0,
        ];

        for &value in &test_values {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &test_values {
            assert_eq!(reader.read::<u32>().expect("read u32"), Some(expected));
        }
    }

    #[test]
    fn test_write_read_u64() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_values = [
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0x123456789ABCDEF0,
        ];

        for &value in &test_values {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &test_values {
            assert_eq!(reader.read::<u64>().expect("read u64"), Some(expected));
        }
    }

    #[test]
    fn test_write_read_u128() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_values = [
            0x00000000000000000000000000000000,
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
            0x55555555555555555555555555555555,
            0x123456789ABCDEF0123456789ABCDEF0,
        ];

        for &value in &test_values {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &test_values {
            assert_eq!(reader.read::<u128>().expect("read u128"), Some(expected));
        }
    }

    #[test]
    fn test_write_read_usize() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_values = [0, usize::MAX, 12345, 67890, 0xABCDEF];

        for &value in &test_values {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &test_values {
            assert_eq!(reader.read::<usize>().expect("read usize"), Some(expected));
        }
    }

    #[test]
    fn test_write_read_signed_integers() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        // Test i8
        let i8_values = [i8::MIN, -1, 0, 1, i8::MAX];
        for &value in &i8_values {
            writer.write(value as u8).expect("write");
        }

        // Test i16
        let i16_values = [i16::MIN, -1000, 0, 1000, i16::MAX];
        for &value in &i16_values {
            writer.write(value as u16).expect("write");
        }

        // Test i32
        let i32_values = [i32::MIN, -100000, 0, 100000, i32::MAX];
        for &value in &i32_values {
            writer.write(value as u32).expect("write");
        }

        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Read back i8 values
        for &expected in &i8_values {
            assert_eq!(reader.read::<u8>().expect("read i8"), Some(expected as u8));
        }

        // Read back i16 values
        for &expected in &i16_values {
            assert_eq!(
                reader.read::<u16>().expect("read i16"),
                Some(expected as u16)
            );
        }

        // Read back i32 values
        for &expected in &i32_values {
            assert_eq!(
                reader.read::<u32>().expect("read i32"),
                Some(expected as u32)
            );
        }
    }

    #[test]
    fn test_write_read_slice() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        writer.write_slice(&test_data).expect("write_slice");
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        let read_data = reader.read_slice(test_data.len()).unwrap();
        assert_eq!(read_data, test_data);
    }

    #[test]
    fn test_mixed_bit_byte_operations() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        writer.write_bit(true).expect("write_bit");
        writer.write_bit(false).expect("write_bit");
        writer.write_bit(true).expect("write_bit");
        writer.write(0xABu8).expect("write");
        writer.write_bit(false).expect("write_bit");
        writer.write_bit(true).expect("write_bit");

        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Read back the bits
        assert_eq!(reader.read_bit().expect("read bit"), Some(true));
        assert_eq!(reader.read_bit().expect("read bit"), Some(false));
        assert_eq!(reader.read_bit().expect("read bit"), Some(true));

        // Read back the byte
        assert_eq!(reader.read::<u8>().expect("read u8"), Some(0xAB));

        // Read back the remaining bits
        assert_eq!(reader.read_bit().expect("read bit"), Some(false));
        assert_eq!(reader.read_bit().expect("read bit"), Some(true));
    }

    #[test]
    fn test_read_bits_various_lengths() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        // Write a known pattern
        writer
            .write(0b11010011100010110101110010101001u32)
            .expect("write");
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Read different lengths of bits
        let bits_1 = reader.read_bits(1).expect("read 1 bit");
        let bits_3 = reader.read_bits(3).expect("read 3 bits");
        let bits_8 = reader.read_bits(8).expect("read 8 bits");
        let bits_4 = reader.read_bits(4).expect("read 4 bits");

        // Verify we get valid u128 values
        assert!(bits_1.unwrap() <= 1); // 1 bit: 0 or 1
        assert!(bits_3.unwrap() <= 7); // 3 bits: 0-7
        assert!(bits_8.unwrap() <= 255); // 8 bits: 0-255
        assert!(bits_4.unwrap() <= 15); // 4 bits: 0-15
    }

    #[test]
    fn test_empty_buffer_reads() {
        let buffer = Vec::new();
        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Reading from empty buffer should gracefully indicate end of stream
        assert_eq!(reader.read_bit().expect("read_bit"), None);
        assert_eq!(reader.read::<u8>().expect("read"), None);
        assert_eq!(reader.read::<u16>().expect("read"), None);
    }

    #[test]
    fn test_partial_byte_flush() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        // Write less than a full byte
        writer.write_bit(true).expect("write_bit");
        writer.write_bit(false).expect("write_bit");
        writer.write_bit(true).expect("write_bit");

        writer.flush().expect("flush should succeed");

        // Buffer should contain at least one byte (with padding)
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_large_data_throughput() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        // Write a large amount of data
        let test_data: Vec<u32> = (0..1000).collect();
        for &value in &test_data {
            writer.write(value).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Read back and verify
        for &expected in &test_data {
            assert_eq!(reader.read::<u32>().expect("read u32"), Some(expected));
        }
    }

    #[test]
    fn test_writer_into_inner() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        writer.write(0x42u8).expect("write");
        writer.flush().expect("flush should succeed");

        let inner = writer.into_inner();
        // Should get back the original buffer
        assert!(!inner.is_empty());
    }

    #[test]
    fn test_reader_into_inner() {
        let buffer = vec![0x42];
        let reader = BitStreamReader::new(buffer.as_slice());
        let inner = reader.into_inner();
        // Should get back the original buffer reference
        assert!(!inner.is_empty());
    }

    #[test]
    fn test_edge_case_zero_length_read() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        writer.write(0x42u8).expect("write");
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Reading zero bytes should succeed and return empty vec
        let empty_slice = reader.read_slice(0).expect("empty");
        assert_eq!(empty_slice.len(), 0);
    }

    #[test]
    fn test_bit_alignment_across_byte_boundaries() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        // Write 7 bits (not byte-aligned)
        for i in 0..7 {
            writer.write_bit(i % 2 == 0).expect("write_bit");
        }

        // Write a full byte
        writer.write(0xFFu8).expect("write");

        // Write 3 more bits
        writer.write_bit(true).expect("write_bit");
        writer.write_bit(false).expect("write_bit");
        writer.write_bit(true).expect("write_bit");

        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Read back the 7 bits
        for i in 0..7 {
            let expected = i % 2 == 0;
            assert_eq!(reader.read_bit().expect("read bit"), Some(expected));
        }

        // Read back the byte
        assert_eq!(reader.read::<u8>().expect("read u8"), Some(0xFF));

        // Read back the 3 bits
        assert_eq!(reader.read_bit().expect("read bit"), Some(true));
        assert_eq!(reader.read_bit().expect("read bit"), Some(false));
        assert_eq!(reader.read_bit().expect("read bit"), Some(true));
    }

    #[test]
    fn test_roundtrip_all_byte_values() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        // Write all possible byte values
        for i in 0..=255u8 {
            writer.write::<u8>(i).expect("write");
        }
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());

        // Read back and verify all values
        for expected in 0..=255u8 {
            assert_eq!(reader.read::<u8>().expect("read u8"), Some(expected));
        }
    }

    #[test]
    fn test_endianness_consistency() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        let test_value = 0x12345678u32;
        writer.write(test_value).expect("write");
        writer.flush().expect("flush should succeed");

        let mut reader = BitStreamReader::new(buffer.as_slice());
        let read_value = reader.read::<u32>().expect("read u32");

        assert_eq!(read_value, Some(test_value));

        // Also test that individual bytes match big-endian expectation
        let mut buffer2 = Vec::new();
        let mut writer2 = BitStreamWriter::new(&mut buffer2);
        writer2.write(0x12345678u32).expect("write");
        writer2.flush().expect("flush should succeed");

        let mut reader2 = BitStreamReader::new(buffer2.as_slice());

        // Should read back as big-endian: 0x12, 0x34, 0x56, 0x78
        assert_eq!(reader2.read::<u8>().expect("read byte 0"), Some(0x12));
        assert_eq!(reader2.read::<u8>().expect("read byte 1"), Some(0x34));
        assert_eq!(reader2.read::<u8>().expect("read byte 2"), Some(0x56));
        assert_eq!(reader2.read::<u8>().expect("read byte 3"), Some(0x78));
    }

    #[test]
    fn test_streaming_behavior() {
        let mut stream = std::io::Cursor::new(Vec::new());

        // Create a writer that will need to flush multiple times
        let mut writer = BitStreamWriter::new(&mut stream);

        // Write enough data to force multiple flushes
        for i in 0..1000u32 {
            writer.write(i).expect("write");
        }
        writer.flush().expect("final flush");

        // Now read it back
        stream.set_position(0);
        let mut reader = BitStreamReader::new(stream);

        // Read back all the data
        for i in 0..1000u32 {
            let value = reader.read::<u32>().expect("read should succeed");
            assert_eq!(value, Some(i), "Value at position {} should match", i);
        }

        // Should be at end of stream now
        assert_eq!(reader.read::<u32>().expect("end read"), None);
    }

    #[test]
    fn test_automatic_buffer_management() {
        let mut stream = std::io::Cursor::new(Vec::new());
        let mut writer = BitStreamWriter::new(&mut stream);

        // Write a mix of different sized data
        writer.write(0xABu8).expect("write");
        writer.write(0xCDEFu16).expect("write");
        writer.write(0x12345678u32).expect("write");
        writer.write(0x9ABCDEF123456789u64).expect("write");

        // Write some bits that don't align to byte boundaries
        writer.write_bits(0b101010u32, 6).expect("write_bits");
        writer.write_bits(0b11001100u32, 8).expect("write_bits");

        writer.flush().expect("flush");

        // Read it all back
        stream.set_position(0);
        let mut reader = BitStreamReader::new(stream);

        assert_eq!(reader.read::<u8>().expect("read u8"), Some(0xAB));
        assert_eq!(reader.read::<u16>().expect("read u16"), Some(0xCDEF));
        assert_eq!(reader.read::<u32>().expect("read u32"), Some(0x12345678));
        assert_eq!(
            reader.read::<u64>().expect("read u64"),
            Some(0x9ABCDEF123456789)
        );

        let bits6 = reader.read_bits(6).expect("read 6 bits");
        assert_eq!(bits6, Some(0b101010));

        let bits8 = reader.read_bits(8).expect("read 8 bits");
        assert_eq!(bits8, Some(0b11001100));
    }
}
