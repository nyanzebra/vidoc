//! Bitstream read/write layer, built on `bitstream-io`.
//!
//! # Why bitstream-io instead of the hand-rolled BitVec?
//!
//! The old implementation routed every write — including whole `u32`s — through
//! a bit-by-bit insertion loop into a 4 KB heap buffer. This had two problems:
//!
//! 1. **Performance**: writing a `u32` cost 32 loop iterations. Profiling showed ~38% of total
//!    encode time was inside `write_bits`/`Range::spec_next`.
//!
//! 2. **Correctness hazards**: `BitVec` maintained two independent positions (`write_pos`,
//!    `read_pos`) with no way to safely bypass it. Every attempt to write raw bytes while the
//!    bitvec held unflushed data corrupted the stream.
//!
//! `bitstream-io` uses a single `u8` accumulator. There is no separate buffer —
//! whole-byte writes go directly to the underlying `Write`. `aligned_writer()` /
//! `aligned_reader()` return `&mut W` / `&mut R` only when the accumulator is
//! provably empty, making raw byte I/O safe by construction.
//!
//! # Endianness
//!
//! Big-endian (MSB first) throughout, matching the existing stream format.
//!
//! # API summary
//!
//! ## Writer
//! | Method | Description |
//! |--------|-------------|
//! | `write_val(v: T)` | Write `T` using its full bit-width |
//! | `write_val_bits(bits, v: T)` | Write `T` using `bits` bits (runtime) |
//! | `write_bit(b)` | Write a single bit |
//! | `write_bytes(&[u8])` | Write bytes through the bit layer |
//! | `write_aligned_bytes(&[u8])` | Align then write raw bytes (fast path) |
//! | `align_to_byte()` | Pad to next byte boundary |
//! | `flush()` | Flush underlying writer |
//!
//! ## Reader
//! | Method | Description |
//! |--------|-------------|
//! | `read_val::<T>()` | Read `T` using its full bit-width, `None` at EOF |
//! | `read_val_bits::<T>(bits)` | Read `bits` bits into `T` (runtime) |
//! | `read_bit()` | Read a single bit, `None` at EOF |
//! | `read_vec(n)` | Read `n` bytes into a `Vec` |
//! | `read_array::<N>()` | Read `N` bytes into a stack array |
//! | `read_raw_bytes::<N>()` | Align then read `N` raw bytes (fast path) |
//! | `align_to_byte()` | Discard bits to next byte boundary |
//! | `count_leading_zeros()` | Count leading zero bits (for Rice decoding) |

use std::io::{self, Read, Write};

use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};

use crate::{Error, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Writer
// ─────────────────────────────────────────────────────────────────────────────

pub struct BitStreamWriter<W: Write> {
    inner: BitWriter<W, BigEndian>,
}

impl<W: Write> BitStreamWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            inner: BitWriter::endian(writer, BigEndian),
        }
    }

    /// Write `T` using its full bit-width (e.g. `u32` → 32 bits, BE).
    ///
    /// Replaces the old `stream.write(val)`. For signed integers cast to
    /// the unsigned equivalent first (`val as u32`), as the existing codebase does.
    #[inline]
    pub fn write_val<T>(&mut self, val: T) -> Result<()>
    where
        T: bitstream_io::Integer,
    {
        let bits = (std::mem::size_of::<T>() * 8) as u32;
        self.inner.write_var(bits, val).map_err(Error::from)
    }

    /// Write `T` using a runtime-specified number of bits.
    ///
    /// Replaces the old `write_bits(value, len)` (note: arg order is now
    /// `bits` first, `value` second to match bitstream-io convention).
    #[inline]
    pub fn write_val_bits<T>(&mut self, bits: u32, val: T) -> Result<()>
    where
        T: bitstream_io::Integer,
    {
        self.inner.write_var(bits, val).map_err(Error::from)
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) -> Result<()> {
        self.inner.write_bit(bit).map_err(Error::from)
    }

    /// Write a byte slice through the bit layer.
    ///
    /// Handles non-aligned position correctly. Replaces the old `write_slice`.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner.write_bytes(bytes).map_err(Error::from)
    }

    /// Align to the next byte boundary, then write raw bytes directly to the
    /// underlying `Write`, bypassing the bit accumulator.
    ///
    /// Replaces the old `write_all_bytes`. This is always safe: `aligned_writer()`
    /// guarantees the accumulator is empty before returning the raw writer,
    /// so no buffered bits can be corrupted — the root cause of the old ANS bugs.
    #[inline]
    pub fn write_aligned_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner
            .aligned_writer()
            .and_then(|w| w.write_all(bytes))
            .map_err(Error::from)
    }

    /// Pad to the next byte boundary with zero bits.
    #[inline]
    pub fn align_to_byte(&mut self) -> Result<()> {
        self.inner.byte_align().map_err(Error::from)
    }

    /// Flush the underlying writer. Does NOT flush partial bits — call
    /// `align_to_byte()` first if needed.
    #[inline]
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush().map_err(Error::from)
    }

    /// Consume the writer and return the underlying `Write`.
    #[inline]
    pub fn into_inner(self) -> W {
        self.inner.into_writer()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reader
// ─────────────────────────────────────────────────────────────────────────────

pub struct BitStreamReader<R: Read> {
    inner: BitReader<R, BigEndian>,
}

impl<R: Read> BitStreamReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            inner: BitReader::endian(reader, BigEndian),
        }
    }

    /// Read `T` using its full bit-width. Returns `None` at EOF.
    ///
    /// Replaces the old `stream.read::<T>()`.
    #[inline]
    pub fn read_val<T>(&mut self) -> Result<Option<T>>
    where
        T: bitstream_io::Integer,
    {
        let bits = (std::mem::size_of::<T>() * 8) as u32;
        match self.inner.read_var(bits) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(Error::from(e)),
        }
    }

    /// Read `bits` bits into `T` (runtime width). Returns `None` at EOF.
    ///
    /// Replaces the old `read_bits(len)`.
    #[inline]
    pub fn read_val_bits<T>(&mut self, bits: u32) -> Result<Option<T>>
    where
        T: bitstream_io::Integer,
    {
        match self.inner.read_var(bits) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(Error::from(e)),
        }
    }

    /// Read a single bit. Returns `None` at EOF.
    #[inline]
    pub fn read_bit(&mut self) -> Result<Option<bool>> {
        match self.inner.read_bit() {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(Error::from(e)),
        }
    }

    /// Read `n` bytes through the bit layer into a `Vec`.
    ///
    /// Replaces `read_slice`.
    #[inline]
    pub fn read_vec(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner.read_bytes(&mut buf).map_err(Error::from)?;
        Ok(buf)
    }

    /// Read exactly `N` bytes through the bit layer into a stack array.
    ///
    /// Replaces `read_array`. Correct at any alignment — uses
    /// `BitRead::read_bytes` internally.
    #[inline]
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.inner.read_bytes(&mut buf).map_err(Error::from)?;
        Ok(buf)
    }

    /// Read exactly `N` raw bytes from the underlying reader after aligning.
    ///
    /// Must only be called after `align_to_byte()`. Uses `aligned_reader()`
    /// which guarantees the accumulator is empty, then calls `read_exact` —
    /// no bit loop, no buffer confusion. This is the safe version of the old
    /// `read_raw_bytes` which had a bitvec look-ahead bug.
    #[inline]
    pub fn read_raw_bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.inner
            .aligned_reader()
            .read_exact(&mut buf)
            .map_err(Error::from)?;
        Ok(buf)
    }

    /// Discard bits up to the next byte boundary.
    #[inline]
    pub fn align_to_byte(&mut self) -> Result<()> {
        self.inner.byte_align();
        Ok(())
    }

    /// Count leading zero bits until a `1` bit (for Rice/unary decoding).
    ///
    /// `read_unary::<1>()` counts zeros until it sees a 1 (the stop bit).
    #[inline]
    pub fn count_leading_zeros(&mut self) -> Result<usize> {
        self.inner
            .read_unary::<1>()
            .map(|n| n as usize)
            .map_err(Error::from)
    }

    /// Consume the reader and return the underlying `Read`.
    #[inline]
    pub fn into_inner(self) -> R {
        self.inner.into_reader()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backwards-compat shims — keep old names compiling during migration.
// Each is a thin delegation. Remove once all call sites are updated.
// ─────────────────────────────────────────────────────────────────────────────

impl<W: Write> BitStreamWriter<W> {
    #[inline]
    #[deprecated(note = "rename to write_val")]
    pub fn write<T: bitstream_io::Integer>(&mut self, val: T) -> Result<()> {
        self.write_val(val)
    }

    #[inline]
    #[deprecated(note = "rename to write_bytes")]
    pub fn write_slice(&mut self, bytes: &[u8]) -> Result<()> {
        self.write_bytes(bytes)
    }

    #[inline]
    #[deprecated(note = "rename to write_aligned_bytes")]
    pub fn write_all_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.write_aligned_bytes(bytes)
    }

    /// Old write_bits had (value, len) arg order. New write_val_bits has (bits, value).
    /// This shim preserves the old order to ease migration.
    #[inline]
    #[deprecated(note = "use write_val_bits(bits, value) — note reversed arg order")]
    pub fn write_bits_compat<T: bitstream_io::Integer>(
        &mut self,
        value: T,
        bits: u32,
    ) -> Result<()> {
        self.write_val_bits(bits, value)
    }
}

impl<R: Read> BitStreamReader<R> {
    #[inline]
    #[deprecated(note = "rename to read_val")]
    pub fn read<T: bitstream_io::Integer>(&mut self) -> Result<Option<T>> {
        self.read_val()
    }

    #[inline]
    #[deprecated(note = "rename to read_vec")]
    pub fn read_slice(&mut self, n: usize) -> Result<Vec<u8>> {
        self.read_vec(n)
    }

    /// Old read_bits returned Option<u128>. This shim keeps that shape.
    #[inline]
    #[deprecated(note = "use read_val_bits::<T>(bits) with explicit type")]
    pub fn read_bits(&mut self, bits: u32) -> Result<Option<u128>> {
        self.read_val_bits::<u128>(bits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn roundtrip_u32() {
        for val in [0u32, 1, 0x12345678, u32::MAX] {
            let mut buf = Vec::new();
            let mut w = BitStreamWriter::new(&mut buf);
            w.write_val(val).unwrap();
            w.flush().unwrap();
            drop(w);
            let mut r = BitStreamReader::new(Cursor::new(&buf));
            assert_eq!(r.read_val::<u32>().unwrap(), Some(val));
        }
    }

    #[test]
    fn bit_and_byte_interleaved() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.write_val(0xABu8).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.flush().unwrap();
        drop(w);

        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_bit().unwrap(), Some(true));
        assert_eq!(r.read_bit().unwrap(), Some(false));
        assert_eq!(r.read_bit().unwrap(), Some(true));
        assert_eq!(r.read_val::<u8>().unwrap(), Some(0xAB));
        assert_eq!(r.read_bit().unwrap(), Some(false));
        assert_eq!(r.read_bit().unwrap(), Some(true));
    }

    #[test]
    fn aligned_raw_bytes_roundtrip() {
        let payload = b"hello world 12345678";
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        // write_aligned_bytes pads the 2 bits then writes raw
        w.write_aligned_bytes(payload).unwrap();
        w.flush().unwrap();
        drop(w);

        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_bit().unwrap(), Some(true));
        assert_eq!(r.read_bit().unwrap(), Some(false));
        r.align_to_byte().unwrap();
        // read_raw_bytes after align_to_byte is safe by construction
        let got = r.read_raw_bytes::<20>().unwrap();
        assert_eq!(&got, payload);
    }

    #[test]
    fn ans_encode_decode_pattern() {
        // The exact pattern that was broken in the old implementation:
        // header via write_val, then align + raw bytes for state and words.
        let data_len: u32 = 42;
        let state: u32 = 0xDEAD_BEEF;
        let words: &[u16] = &[0x1234, 0x5678, 0xABCD];

        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_val(data_len).unwrap();
        w.align_to_byte().unwrap();
        w.write_aligned_bytes(&state.to_le_bytes()).unwrap();
        let mut word_bytes = Vec::new();
        for &wrd in words {
            word_bytes.extend_from_slice(&wrd.to_le_bytes());
        }
        w.write_aligned_bytes(&word_bytes).unwrap();
        w.flush().unwrap();
        drop(w);

        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_val::<u32>().unwrap(), Some(data_len));
        r.align_to_byte().unwrap();
        let got_state = u32::from_le_bytes(r.read_raw_bytes::<4>().unwrap());
        assert_eq!(got_state, state);
        for &expected in words {
            let got = u16::from_le_bytes(r.read_raw_bytes::<2>().unwrap());
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn endianness_big_endian() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_val(0x12345678u32).unwrap();
        w.flush().unwrap();
        drop(w);
        // BE: MSB first
        assert_eq!(buf, [0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn leading_zeros() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        for _ in 0..5 {
            w.write_bit(false).unwrap();
        }
        w.write_bit(true).unwrap();
        w.flush().unwrap();
        drop(w);
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.count_leading_zeros().unwrap(), 5);
    }

    #[test]
    fn eof_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_val::<u8>().unwrap(), None);
        assert_eq!(r.read_bit().unwrap(), None);
    }

    #[test]
    fn large_roundtrip() {
        let vals: Vec<u32> = (0..2000).collect();
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        for &v in &vals {
            w.write_val(v).unwrap();
        }
        w.flush().unwrap();
        drop(w);
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        for &expected in &vals {
            assert_eq!(r.read_val::<u32>().unwrap(), Some(expected));
        }
    }

    #[test]
    fn write_val_bits_and_read_val_bits() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_val_bits(12u32, 0b1010_0011_1100u16).unwrap();
        w.flush().unwrap();
        drop(w);
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_val_bits::<u16>(12).unwrap(), Some(0b1010_0011_1100));
    }
}
