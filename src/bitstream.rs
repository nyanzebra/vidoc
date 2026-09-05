use std::io::{self, Read, Write};

use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};

use crate::{Error, Result};

pub struct BitStreamWriter<W>
where
    W: Write,
{
    inner: BitWriter<W, BigEndian>,
}

impl<W> BitStreamWriter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        Self {
            inner: BitWriter::endian(writer, BigEndian),
        }
    }

    /// Write `T` using its full bit-width (e.g. `u32` → 32 bits, BE).
    #[inline]
    pub fn write<T>(&mut self, val: T) -> Result<()>
    where
        T: bitstream_io::Integer,
    {
        let bits = (std::mem::size_of::<T>() * 8) as u32;
        self.inner.write_var(bits, val).map_err(Error::from)
    }

    /// Write `T` using a runtime-specified number of bits.
    #[inline]
    pub fn write_bits<T>(&mut self, bits: u32, val: T) -> Result<()>
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
}

pub struct BitStreamReader<R>
where
    R: Read,
{
    inner: BitReader<R, BigEndian>,
}

impl<R> BitStreamReader<R>
where
    R: Read,
{
    pub fn new(reader: R) -> Self {
        Self {
            inner: BitReader::endian(reader, BigEndian),
        }
    }

    /// Read `T` using its full bit-width. Returns `None` at EOF.
    #[inline]
    pub fn read<T>(&mut self) -> Result<Option<T>>
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
    #[inline]
    pub fn read_bits<T>(&mut self, bits: u32) -> Result<Option<T>>
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
    #[inline]
    pub fn read_to_vec(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner.read_bytes(&mut buf).map_err(Error::from)?;
        Ok(buf)
    }

    /// Read exactly `N` bytes through the bit layer into a stack array.
    #[inline]
    pub fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
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
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    impl<W> BitStreamWriter<W>
    where
        W: Write,
    {
        /// Consume the writer and return the underlying `Write`.
        #[inline]
        pub fn into_inner(self) -> W {
            self.inner.into_writer()
        }
    }

    impl<R> BitStreamReader<R>
    where
        R: Read,
    {
        /// Consume the reader and return the underlying `Read`.
        #[inline]
        pub fn into_inner(self) -> R {
            self.inner.into_reader()
        }
    }

    #[test]
    fn roundtrip_u32() {
        for val in [0u32, 1, 0x12345678, u32::MAX] {
            let mut buf = Vec::new();
            let mut w = BitStreamWriter::new(&mut buf);
            w.write(val).unwrap();
            w.flush().unwrap();
            let mut r = BitStreamReader::new(Cursor::new(&buf));
            assert_eq!(r.read::<u32>().unwrap(), Some(val));
        }
    }

    #[test]
    fn bit_and_byte_interleaved() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.write(0xABu8).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.flush().unwrap();

        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_bit().unwrap(), Some(true));
        assert_eq!(r.read_bit().unwrap(), Some(false));
        assert_eq!(r.read_bit().unwrap(), Some(true));
        assert_eq!(r.read::<u8>().unwrap(), Some(0xAB));
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
        w.write(data_len).unwrap();
        w.align_to_byte().unwrap();
        w.write_aligned_bytes(&state.to_le_bytes()).unwrap();
        let mut word_bytes = Vec::new();
        for &wrd in words {
            word_bytes.extend_from_slice(&wrd.to_le_bytes());
        }
        w.write_aligned_bytes(&word_bytes).unwrap();
        w.flush().unwrap();

        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read::<u32>().unwrap(), Some(data_len));
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
        w.write(0x12345678u32).unwrap();
        w.flush().unwrap();
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
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.count_leading_zeros().unwrap(), 5);
    }

    #[test]
    fn eof_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read::<u8>().unwrap(), None);
        assert_eq!(r.read_bit().unwrap(), None);
    }

    #[test]
    fn large_roundtrip() {
        let vals: Vec<u32> = (0..2000).collect();
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        for &v in &vals {
            w.write(v).unwrap();
        }
        w.flush().unwrap();
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        for &expected in &vals {
            assert_eq!(r.read::<u32>().unwrap(), Some(expected));
        }
    }

    #[test]
    fn write_val_bits_and_read_bits() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        w.write_bits(12u32, 0b1010_0011_1100u16).unwrap();
        w.flush().unwrap();
        let mut r = BitStreamReader::new(Cursor::new(&buf));
        assert_eq!(r.read_bits::<u16>(12).unwrap(), Some(0b1010_0011_1100));
    }
}
