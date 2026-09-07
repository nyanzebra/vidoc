//! Entropy coding via lz4_flex.
//!
//! Drop-in replacement for the hand-rolled ANS implementation.
//! The public API is identical so all call sites are unchanged.
//!
//! # Why lz4_flex?
//!
//! - Pure Rust, no unsafe dependencies
//! - LZ4 is extremely fast to compress and decompress
//! - The block format (lz4_flex::compress_prepend_size / decompress_size_prepended) prepends the
//!   original length so the decompressor knows how much to allocate — no separate length field
//!   needed
//! - DCT coefficient blocks have significant byte-level redundancy (many zeros, repeated small
//!   values) that LZ4's LZ77 pass exploits before entropy coding, which ANS alone cannot see
//!
//! # Stream format (per encode_raw call)
//!
//! ```text
//! [compressed_len: u32 LE, raw bytes via write_aligned_bytes]
//! [lz4 block:      compressed_len bytes, raw]
//! ```
//!
//! The original data length is embedded in the lz4 block header by
//! compress_prepend_size, so we only need to store the compressed byte count.

use std::{
    cell::RefCell,
    io::{Read, Write},
};

use lz4_flex::{compress_prepend_size, decompress_size_prepended};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    error::{Error, Result},
};

// ─────────────────────────────────────────────────────────────────────────────
// Core helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compress `bytes` with LZ4 and write to `stream`.
fn write_compressed<W: Write>(bytes: &[u8], stream: &mut BitStreamWriter<W>) -> Result<()> {
    let compressed = compress_prepend_size(bytes);
    // Write the compressed length so the reader knows how many bytes to read.
    stream.flush()?;
    stream.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes())?;
    stream.write_aligned_bytes(&compressed)?;
    Ok(())
}

/// Read a compressed block from `stream` and decompress it.
fn read_compressed<R: Read>(stream: &mut BitStreamReader<R>) -> Result<Vec<u8>> {
    let len = u32::from_le_bytes(stream.read_raw_bytes::<4>()?) as usize;
    decompress_size_prepended(&stream.read_to_vec(len)?)
        .map_err(|e| Error::FailedToDecode(format!("lz4: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API — identical signatures to the old ANS module
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a slice of `T` by serialising to bytes then LZ4-compressing.
pub(crate) fn encode<T, W>(data: &[T], stream: &mut BitStreamWriter<W>) -> Result<()>
where
    T: crate::ToBytes,
    W: Write,
{
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_bytes()).collect();
    write_compressed(&bytes, stream)
}

/// Decode a slice of `T` by LZ4-decompressing then deserialising.
pub(crate) fn decode<T, R>(stream: &mut BitStreamReader<R>) -> Result<Vec<T>>
where
    T: crate::FromBytes,
    R: Read,
{
    let raw = read_compressed(stream)?;
    let item_size = std::mem::size_of::<T>();
    let mut res = Vec::with_capacity(raw.len().checked_div(item_size).unwrap_or(0));
    let mut start = 0;
    while start < raw.len() {
        let (item, consumed) = T::from_bytes(&raw[start..]);
        if consumed == 0 {
            break;
        }
        res.push(item);
        start += consumed;
    }
    Ok(res)
}

thread_local! {
    static SCRATCH: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(4096));
}

/// Encode a slice of `T` (fixed-size numeric type) by casting to bytes then
/// LZ4-compressing. Replaces the old `encode_raw::<N, T, W>`.
pub(crate) fn encode_raw<const N: usize, T, W>(
    data: &[T],
    stream: &mut BitStreamWriter<W>,
) -> Result<()>
where
    T: num_traits::ToBytes<Bytes = [u8; N]> + Copy,
    W: Write,
{
    if data.is_empty() {
        return Err(Error::InvalidData);
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(data.len() * N);
    for item in data {
        bytes.extend_from_slice(&item.to_le_bytes());
    }
    write_compressed(&bytes, stream)
}

/// Decode a slice of `T` from a LZ4 block. Replaces the old `decode_raw::<N, T, R>`.
pub(crate) fn decode_raw<const N: usize, T, R>(stream: &mut BitStreamReader<R>) -> Result<Vec<T>>
where
    T: num_traits::FromBytes<Bytes = [u8; N]> + Copy,
    R: Read,
{
    let raw = read_compressed(stream)?;
    if raw.len() % N != 0 {
        return Err(Error::FailedToDecode(format!(
            "decompressed length {} not a multiple of element size {N}",
            raw.len()
        )));
    }
    let mut result = Vec::with_capacity(raw.len() / N);
    for chunk in raw.chunks_exact(N) {
        let arr: [u8; N] = chunk.try_into().unwrap();
        result.push(T::from_le_bytes(&arr));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{BitStreamReader, BitStreamWriter};

    fn roundtrip_raw(data: &[i16]) -> Vec<i16> {
        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            encode_raw::<2, i16, _>(data, &mut w).unwrap();
            w.flush().unwrap();
        }
        decode_raw::<2, i16, _>(&mut BitStreamReader::new(Cursor::new(buf))).unwrap()
    }

    #[test]
    fn roundtrip_single() {
        let d = vec![42i16; 100];
        assert_eq!(roundtrip_raw(&d), d);
    }

    #[test]
    fn roundtrip_mixed() {
        let d: Vec<i16> = (-128..128).collect();
        assert_eq!(roundtrip_raw(&d), d);
    }

    #[test]
    fn roundtrip_dct_like() {
        let mut d = vec![0i16; 200];
        d.extend([-3i16, -1, 0, 1, 3, -2, 2, 0, 0, 1]);
        assert_eq!(roundtrip_raw(&d), d);
    }

    #[test]
    fn roundtrip_generic_encode_decode() {
        // Test the generic encode/decode path used by macro blocks
        #[derive(Clone, PartialEq, Debug)]
        struct Pair(u8, u8);

        impl crate::ToBytes for Pair {
            fn to_bytes(&self) -> Vec<u8> {
                vec![self.0, self.1]
            }
        }
        impl crate::FromBytes for Pair {
            fn from_bytes(bytes: &[u8]) -> (Self, usize) {
                (Pair(bytes[0], bytes[1]), 2)
            }
        }

        let data: Vec<Pair> = (0u8..64).map(|i| Pair(i, i.wrapping_mul(3))).collect();
        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            encode(&data, &mut w).unwrap();
            w.flush().unwrap();
        }
        let decoded: Vec<Pair> = decode(&mut BitStreamReader::new(Cursor::new(buf))).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn empty_returns_error() {
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        let result = encode_raw::<2, i16, _>(&[], &mut w);
        assert!(result.is_err());
    }
}
