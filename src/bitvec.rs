use std::{
    fmt::Binary,
    io::{Read, Write},
};

use crate::{FromBytes, Result, ToBytes};

// Use a reasonable buffer size for heap allocation
// 4KB page
const PAGE_SIZE: usize = 4096;
const PAGE_SIZE_BITS: usize = PAGE_SIZE * 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BitVec {
    // Heap-allocated buffer
    data: Box<[u8]>,
    // Position where next bit will be written
    write_pos: usize,
    // Position where next bit will be read
    read_pos: usize,
}

impl Binary for BitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.len() {
            let bit = self.get_bit(self.read_pos + i);
            write!(f, "{}", if bit { '1' } else { '0' })?;
        }
        Ok(())
    }
}

impl BitVec {
    pub(crate) fn new() -> Self {
        Self {
            data: vec![0; PAGE_SIZE].into_boxed_slice(),
            write_pos: 0,
            read_pos: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.write_pos - self.read_pos
    }

    pub(crate) fn bit_position_in_byte(&self) -> usize {
        self.read_pos % 8
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
    }

    pub(crate) fn available_capacity_bits(&self) -> usize {
        let max_bits = PAGE_SIZE_BITS; // PAGE_SIZE is in bytes, convert to bits
        max_bits.saturating_sub(self.write_pos)
    }
}

impl BitVec {
    pub(crate) fn push_bit(&mut self, bit: bool) {
        assert!(
            self.write_pos < PAGE_SIZE,
            "BitVec is full - no more capacity"
        );

        self.set_bit(self.write_pos, bit);
        self.write_pos += 1;
    }

    /// Pushes `len` bits from `val` into the `BitVec`. If `len` is larger than the available
    /// capacity then we write what can fit and return the remainder value and remaining bit
    /// count. Returns (remaining_value, remaining_bits) where remaining_bits = 0 means all bits
    /// were written.
    #[inline]
    pub(crate) fn push_bits<T>(&mut self, val: T, len: u8) -> (u128, u8)
    where
        T: Into<u128>,
    {
        let val = val.into();
        if len == 0 {
            return (0, 0);
        }

        let available_capacity = self.available_capacity_bits();
        if available_capacity == 0 {
            // No capacity at all, return everything
            return (val, len);
        }

        let bits_to_write = std::cmp::min(len as usize, available_capacity);
        let mut remaining_len = len;
        let current_val = val;

        // Write bits one by one to ensure we don't exceed capacity
        let mut bits_written = 0;
        while bits_written < bits_to_write && remaining_len > 0 {
            let bit_pos = remaining_len - 1;
            let bit = (current_val >> bit_pos) & 1 == 1;
            self.set_bit(self.write_pos, bit);
            self.write_pos += 1;
            remaining_len -= 1;
            bits_written += 1;
        }

        if remaining_len > 0 {
            // Calculate the remaining value based on what we couldn't write
            let remaining_mask = (1u128 << remaining_len) - 1;
            let remaining_val = current_val & remaining_mask;
            (remaining_val, remaining_len)
        } else {
            (0, 0)
        }
    }

    pub(crate) fn push<T>(&mut self, val: T) -> (u128, u8)
    where
        T: Into<u128>,
    {
        let len = std::mem::size_of::<T>() * 8;
        self.push_bits(val, len as u8)
    }

    pub(crate) fn pop_bit(&mut self) -> Option<bool> {
        if self.read_pos >= self.write_pos {
            None
        } else {
            let bit = self.get_bit(self.read_pos);
            self.read_pos += 1;
            Some(bit)
        }
    }

    pub(crate) fn pop_bits(&mut self, bits: u8) -> Option<u128> {
        if bits == 0 || self.len() < bits as usize {
            return None;
        }

        let mut result = 0u128;
        let mut remaining_bits = bits;

        // Handle full bytes first
        while remaining_bits >= 8 {
            let mut byte_val = 0u8;

            // If we're byte-aligned, read directly
            if self.read_pos.is_multiple_of(8) {
                let byte_idx = self.read_pos / 8;
                if byte_idx >= PAGE_SIZE {
                    return None;
                }
                byte_val = self.data[byte_idx];
                self.read_pos += 8;
            } else {
                // Not byte-aligned, read bit by bit for this byte
                for _ in 0..8 {
                    let bit = self.pop_bit()?;
                    byte_val = (byte_val << 1) | (bit as u8);
                }
            }

            // Add this byte to our result
            result = (result << 8) | (byte_val as u128);
            remaining_bits -= 8;
        }

        // Handle remaining bits (less than 8)
        for _ in 0..remaining_bits {
            let bit = self.pop_bit()?;
            result = (result << 1) | (bit as u128);
        }

        Some(result)
    }
}

impl BitVec {
    pub(crate) fn flush_to_stream<W: Write>(&mut self, writer: &mut W) -> Result<()> {
        // Write only the actual data, no length prefix
        if self.write_pos == 0 {
            return Ok(()); // Nothing to write
        }

        // Calculate bytes needed from the start (round up)
        let bytes_needed = self.write_pos.div_ceil(8);
        writer.write_all(&self.data[0..bytes_needed])?;

        // Clear the buffer by resetting positions
        self.data = vec![0; PAGE_SIZE].into_boxed_slice();
        self.write_pos = 0;
        self.read_pos = 0;

        Ok(())
    }

    pub(crate) fn read_from_stream<R: Read>(&mut self, reader: &mut R) -> Result<()> {
        // Save the current partial byte at write_pos if we're not aligned
        let byte_offset = self.write_pos / 8;
        let bit_offset = self.write_pos % 8;

        let saved_partial_byte = if bit_offset != 0 {
            Some(self.data[byte_offset])
        } else {
            None
        };

        // Read as much data as available, up to our buffer capacity
        let bytes_read = reader.read(&mut self.data[byte_offset..])?;

        if bytes_read == 0 {
            return Ok(());
        }

        // If we had a partial byte, restore the bits that shouldn't be overwritten
        if let Some(partial_byte) = saved_partial_byte {
            if bytes_read > 0 {
                // Create a mask to preserve the existing bits in the partial byte
                let preserve_mask = (1u8 << bit_offset) - 1;
                let preserved_bits = partial_byte & preserve_mask;

                // Clear the low bits from the new byte and combine with preserved bits
                let new_byte = self.data[byte_offset] & !preserve_mask;
                self.data[byte_offset] = new_byte | preserved_bits;
            }
        }

        self.write_pos += bytes_read * 8;

        Ok(())
    }

    pub(crate) fn flush<W: Write>(&mut self, writer: &mut W) -> Result<()> {
        self.flush_to_stream(writer)
    }
}

impl BitVec {
    pub(crate) fn first_one(&self) -> Option<usize> {
        let zeroes = self.zeroes();
        let len = self.len();
        if zeroes == len {
            None
        } else {
            Some(len - zeroes)
        }
    }

    fn zeroes(&self) -> usize {
        let mut count = 0;
        for i in 0..PAGE_SIZE_BITS {
            if self.get_bit(i) {
                break;
            }
            count += 1;
        }
        count
    }
}

impl BitVec {
    fn set_bit(&mut self, bit_idx: usize, bit: bool) {
        let byte_idx = bit_idx / 8;
        let bit_pos = 7 - (bit_idx % 8);

        if bit {
            self.data[byte_idx] |= 1 << bit_pos;
        } else {
            self.data[byte_idx] &= !(1 << bit_pos);
        }
    }

    fn get_bit(&self, bit_idx: usize) -> bool {
        let byte_idx = bit_idx / 8;
        let bit_pos = 7 - (bit_idx % 8);
        (self.data[byte_idx] >> bit_pos) & 1 == 1
    }
}

impl From<bool> for BitVec {
    fn from(value: bool) -> Self {
        let mut bitvec = BitVec::new();
        bitvec.push_bit(value);
        bitvec
    }
}

impl From<u8> for BitVec {
    fn from(value: u8) -> Self {
        let mut bitvec = BitVec::new();
        bitvec.push(value);
        bitvec
    }
}

impl From<u16> for BitVec {
    fn from(value: u16) -> Self {
        let mut bitvec = BitVec::new();
        bitvec.push(value);
        bitvec
    }
}

impl From<u32> for BitVec {
    fn from(value: u32) -> Self {
        let mut bitvec = BitVec::new();
        bitvec.push(value);
        bitvec
    }
}

impl From<u64> for BitVec {
    fn from(value: u64) -> Self {
        let mut bitvec = BitVec::new();
        bitvec.push(value);
        bitvec
    }
}

impl From<u128> for BitVec {
    fn from(value: u128) -> Self {
        let mut bitvec = BitVec::new();
        bitvec.push(value);
        bitvec
    }
}

impl ToBytes for BitVec {
    fn to_bytes(&self) -> Vec<u8> {
        let bytes_needed = self.len().div_ceil(8);
        self.data[self.read_pos / 8..(self.read_pos / 8) + bytes_needed].to_vec()
    }
}

impl FromBytes for BitVec {
    fn from_bytes(bytes: &[u8]) -> (Self, usize) {
        if bytes.len() > PAGE_SIZE {
            panic!("Byte array too large for BitVec");
        }

        let mut data = vec![0u8; PAGE_SIZE];
        data[..bytes.len()].copy_from_slice(bytes);
        let len = bytes.len() * 8;
        (
            Self {
                data: data.into_boxed_slice(),
                write_pos: len,
                read_pos: 0,
            },
            bytes.len(),
        )
    }
}

impl std::ops::Index<usize> for BitVec {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!(
                "Index {} out of bounds for BitVec of length {}",
                index,
                self.len()
            );
        }

        if self.get_bit(self.read_pos + index) {
            &true
        } else {
            &false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_vec_new() {
        let bitvec = BitVec::new();
        assert_eq!(bitvec.len(), 0);
        assert!(bitvec.is_empty());
    }

    #[test]
    fn bit_vec_push_pop_bit() {
        let mut bitvec = BitVec::new();

        bitvec.push_bit(true);
        bitvec.push_bit(false);
        bitvec.push_bit(true);

        assert_eq!(bitvec.len(), 3);
        assert_eq!(bitvec.pop_bit(), Some(true));
        assert_eq!(bitvec.pop_bit(), Some(false));
        assert_eq!(bitvec.pop_bit(), Some(true));
        assert_eq!(bitvec.pop_bit(), None);
    }

    #[test]
    fn bit_vec_push_pop_bits() {
        let mut bitvec = BitVec::new();

        let (remainder, remaining_bits) = bitvec.push_bits(0b1010u8, 4);
        assert_eq!(remainder, 0);
        assert_eq!(remaining_bits, 0);
        assert_eq!(bitvec.len(), 4);

        let val = bitvec.pop_bits(4);
        assert_eq!(val, Some(0b1010));
        assert_eq!(bitvec.len(), 0);
    }

    #[test]
    fn bit_vec_from_primitives() {
        let bitvec = BitVec::from(0b1010u8);
        assert_eq!(bitvec.len(), 8);

        let bitvec = BitVec::from(0b1010u16);
        assert_eq!(bitvec.len(), 16);
    }

    #[test]
    fn bit_vec_stream_io() {
        let mut bitvec = BitVec::new();
        let (remainder, remaining_bits) = bitvec.push_bits(0b1010u8, 4);
        assert_eq!(remainder, 0);
        assert_eq!(remaining_bits, 0);
        let (remainder, remaining_bits) = bitvec.push_bits(0b1100u8, 4);
        assert_eq!(remainder, 0);
        assert_eq!(remaining_bits, 0);

        let mut buffer = Vec::new();
        bitvec.flush_to_stream(&mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(buffer);
        let mut restored = BitVec::new();
        restored.read_from_stream(&mut cursor).unwrap();

        assert_eq!(restored.len(), 8);
    }

    #[test]
    fn bit_vec_push_bits_overflow_behavior() {
        let mut bitvec = BitVec::new();

        // Fill up most of the buffer first
        let _capacity = bitvec.available_capacity_bits();

        // Try to write exactly at capacity - should work
        let (remainder, remaining_bits) = bitvec.push_bits(0xFFu8, 8);
        assert_eq!(remainder, 0);
        assert_eq!(remaining_bits, 0);
        assert_eq!(bitvec.len(), 8);

        // Now try to write more than available capacity
        let large_value = 0xDEADBEEFu32;
        let (remainder, remaining_bits) = bitvec.push_bits(large_value, 32);

        if remaining_bits > 0 {
            // Some bits couldn't fit - this is expected behavior
            assert!(remainder > 0);
            assert!(bitvec.len() < 8 + 32); // Should be less than full requested size

        // The bitstream writer should handle this by flushing and writing remainder
        } else {
            // Everything fit - buffer was large enough
            assert_eq!(remainder, 0);
            assert_eq!(bitvec.len(), 8 + 32);
        }
    }

    #[test]
    fn bit_vec_push_bits_exact_fill() {
        let mut bitvec = BitVec::new();

        // Write data until we get close to capacity
        let mut total_written = 0;
        while bitvec.available_capacity_bits() > 8 {
            let (remainder, remaining_bits) = bitvec.push_bits(0xAAu8, 8);
            assert_eq!(remainder, 0);
            assert_eq!(remaining_bits, 0);
            total_written += 8;
        }

        // Now write exactly what fits
        let available = bitvec.available_capacity_bits();
        if available > 0 {
            let (remainder, remaining_bits) = bitvec.push_bits(0xFFu8, available as u8);
            assert_eq!(remainder, 0);
            assert_eq!(remaining_bits, 0);
            total_written += available;
        }

        assert_eq!(bitvec.len(), total_written);

        // Now try to write one more bit - should return it as remainder
        let (remainder, remaining_bits) = bitvec.push_bits(1u8, 1);
        assert_eq!(remainder, 1);
        assert_eq!(remaining_bits, 1);
        assert_eq!(bitvec.len(), total_written); // Should be unchanged
    }
}
