//! Asymmetric Numeral Systems (rANS) entropy coder.
//!
//! # Stream layout
//!
//! ```text
//! [unique_count: u32]
//! for each symbol:
//!   [symbol bytes: N bytes via bit layer]
//!   [frequency: u32 via bit layer]
//! [align to byte]
//! [data_len: u32 via bit layer]
//! --- flush bit layer here ---
//! [state: u32 LE, raw]
//! [renorm words: u16 LE each, raw, in decode order]
//! ```
//!
//! The header goes through the bit layer (read/write) because it
//! may be preceded by other bit-layer writes. The ANS output (state + words)
//! is written raw after a flush(), allowing direct byte I/O on both sides.

use std::io::{Read, Write};

use num_traits::{ops::bytes::ToBytes, FromBytes, PrimInt};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    error::{Error, Result},
};

const BITS_PER_BYTE: usize = 8;
const MAX_ARRAY_SIZE: usize = 65536;
const I16_OFFSET: i32 = 32768;
const BYTE_SIZE_U8: usize = 1;
const BYTE_SIZE_I16: usize = 2;
const ANS_TABLE_SIZE: usize = 4096;
const RENORM_SHIFT_BITS: u32 = 16;
const RENORM_MASK: u32 = 0xFFFF;
const MIN_NORMALIZED_FREQ: u32 = 1;
// Estimated output bytes per input byte for capacity pre-sizing.
// Over-estimating is cheap; under-estimating causes realloc.
const BYTES_PER_INPUT_I16: usize = 2; // ~50 % compression typical
const BYTES_PER_INPUT_U8: usize = 1; // near 1:1 for u8 block data
const MIN_OUTPUT_WORDS: usize = 512;

// ─────────────────────────────────────────────────────────────────────────────
// Symbol index helpers — avoid repeated N == BYTE_SIZE_I16 branches
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn to_idx<const N: usize, T: PrimInt>(symbol: T) -> usize {
    if N == BYTE_SIZE_I16 {
        (symbol.to_i16().unwrap() as i32 + I16_OFFSET) as usize
    } else {
        symbol.to_u8().unwrap() as usize
    }
}

#[inline(always)]
fn from_idx<const N: usize, T: PrimInt>(idx: usize) -> T {
    if N == BYTE_SIZE_I16 {
        T::from(idx as i32 - I16_OFFSET).unwrap()
    } else {
        T::from(idx).unwrap()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frequency normalization — shared by encoder and decoder
// ─────────────────────────────────────────────────────────────────────────────

fn normalize_frequencies(freq_table: &[u32], array_size: usize) -> Vec<u32> {
    let total_freq: u32 = freq_table.iter().sum();
    let mut normalized = vec![0u32; array_size];
    let mut normalized_total = 0u32;

    for (idx, &freq) in freq_table.iter().enumerate() {
        if freq > 0 {
            let nf = ((freq as u64 * ANS_TABLE_SIZE as u64) / total_freq as u64) as u32;
            let nf = nf.max(MIN_NORMALIZED_FREQ);
            normalized[idx] = nf;
            normalized_total += nf;
        }
    }

    // Adjust so frequencies sum to exactly TABLE_SIZE
    if normalized_total != ANS_TABLE_SIZE as u32 {
        let diff = ANS_TABLE_SIZE as i64 - normalized_total as i64;
        let (max_idx, _) = normalized
            .iter()
            .enumerate()
            .max_by_key(|(_, &f)| f)
            .unwrap_or((0, &0));
        normalized[max_idx] =
            (normalized[max_idx] as i64 + diff).max(MIN_NORMALIZED_FREQ as i64) as u32;
    }

    normalized
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn encode<T>(data: &[T], stream: &mut BitStreamWriter<impl Write>) -> Result<()>
where
    T: crate::ToBytes,
{
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_bytes()).collect();
    encode_raw::<BYTE_SIZE_U8, u8, _>(&bytes, stream)
}

pub(crate) fn encode_raw<const N: usize, T, W>(
    data: &[T],
    stream: &mut BitStreamWriter<W>,
) -> Result<()>
where
    T: PrimInt + ToBytes<Bytes = [u8; N]> + Copy + Eq + Ord,
    W: Write,
{
    if data.is_empty() {
        return Err(Error::InvalidData);
    }

    let array_size = 1usize << (BITS_PER_BYTE * N);
    if array_size > MAX_ARRAY_SIZE {
        return Err(Error::UnsupportedAnsDataType(N * BITS_PER_BYTE));
    }

    // ── Build frequency table ─────────────────────────────────────────────────
    let mut freq_table = vec![0u32; array_size];
    for &sym in data {
        freq_table[to_idx::<N, T>(sym)] += 1;
    }

    // ── Build symbol map (idx → symbol, for header) ───────────────────────────
    let unique_count = freq_table.iter().filter(|&&f| f > 0).count();
    let mut symbol_map: Vec<(T, u32)> = Vec::with_capacity(unique_count);
    for (idx, &count) in freq_table.iter().enumerate() {
        if count > 0 {
            symbol_map.push((from_idx::<N, T>(idx), count));
        }
    }

    // ── Write header (through bit layer) ─────────────────────────────────────
    stream.write(unique_count as u32)?;
    for (sym, count) in &symbol_map {
        stream.write_bytes(&sym.to_be_bytes())?;
        stream.write(*count)?;
    }
    stream.align_to_byte()?;
    stream.write(data.len() as u32)?;

    if unique_count == 1 {
        stream.flush()?;
        return Ok(());
    }

    // ── Build encode table ────────────────────────────────────────────────────
    let normalized = normalize_frequencies(&freq_table, array_size);
    let mut encode_table = vec![(0u32, 0u32); array_size]; // (freq, cumulative)
    let mut cum = 0u32;
    for (idx, &nf) in normalized.iter().enumerate() {
        if nf > 0 {
            encode_table[idx] = (nf, cum);
            cum += nf;
        }
    }

    // ── rANS encode (reverse order so decode goes forward) ───────────────────
    let mut state = ANS_TABLE_SIZE as u32;
    let table_u32 = ANS_TABLE_SIZE as u32;
    let renorm_threshold = 1u32 << RENORM_SHIFT_BITS;

    let cap = (data.len()
        / if N == BYTE_SIZE_I16 {
            BYTES_PER_INPUT_I16
        } else {
            BYTES_PER_INPUT_U8
        })
    .max(MIN_OUTPUT_WORDS);
    let mut output_words: Vec<u16> = Vec::with_capacity(cap);

    for &sym in data.iter().rev() {
        let (freq, cum) = encode_table[to_idx::<N, T>(sym)];
        let threshold = freq * renorm_threshold;
        while state >= threshold {
            output_words.push((state & RENORM_MASK) as u16);
            state >>= RENORM_SHIFT_BITS;
        }
        state = (state / freq) * table_u32 + cum + (state % freq);
    }

    // ── Write ANS output as raw bytes ─────────────────────────────────────────
    // flush() drains the bit-layer accumulator (still holds data.len() bits)
    // before we bypass it with write_aligned_bytes.
    stream.flush()?;
    stream.write_aligned_bytes(&state.to_le_bytes())?;

    // Batch all words into one allocation → one write_all call
    let mut word_bytes: Vec<u8> = Vec::with_capacity(output_words.len() * 2);
    for &w in output_words.iter().rev() {
        word_bytes.extend_from_slice(&w.to_le_bytes());
    }
    stream.write_aligned_bytes(&word_bytes)?;

    Ok(())
}

pub(crate) fn decode<T, R>(stream: &mut BitStreamReader<R>) -> Result<Vec<T>>
where
    T: crate::FromBytes,
    R: Read,
{
    let raw: Vec<u8> = decode_raw(stream)?;
    let item_size = std::mem::size_of::<T>();
    let mut res = Vec::with_capacity(raw.len().checked_div(item_size).unwrap_or(0));
    let mut start = 0;
    while start < raw.len() {
        let (block, end) = T::from_bytes(&raw[start..]);
        debug_assert_ne!(end, 0, "from_bytes returned zero advance");
        if end == 0 {
            break;
        }
        res.push(block);
        start += end;
    }
    Ok(res)
}

pub(crate) fn decode_raw<const N: usize, T, R>(stream: &mut BitStreamReader<R>) -> Result<Vec<T>>
where
    T: PrimInt + FromBytes<Bytes = [u8; N]> + Copy + Eq + Ord,
    R: Read,
{
    let array_size = 1usize << (BITS_PER_BYTE * N);
    if array_size > MAX_ARRAY_SIZE {
        return Err(Error::UnsupportedAnsDataType(N * BITS_PER_BYTE));
    }

    // ── Read header (bit layer) ───────────────────────────────────────────────
    let num_symbols = stream
        .read::<u32>()?
        .ok_or(Error::FailedToDecode("num_symbols".to_owned()))? as usize;

    let mut freq_table = vec![0u32; array_size];
    let mut symbols: Vec<T> = Vec::with_capacity(num_symbols);

    for _ in 0..num_symbols {
        let bytes = stream.read_exact::<N>()?;
        let sym = T::from_be_bytes(&bytes);
        let freq = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("frequency".to_owned()))?;
        freq_table[to_idx::<N, T>(sym)] = freq;
        symbols.push(sym);
    }

    stream.align_to_byte()?;

    let len = stream
        .read::<u32>()?
        .ok_or(Error::FailedToDecode("len".to_owned()))? as usize;

    if num_symbols == 1 {
        return Ok(vec![symbols[0]; len]);
    }

    // ── Build decode table ────────────────────────────────────────────────────
    let normalized = normalize_frequencies(&freq_table, array_size);

    // decode_table[state % TABLE_SIZE] = (symbol_idx, freq, cumulative_start)
    // Stored as a flat array — TABLE_SIZE = 4096, fits on stack as a Vec.
    let mut decode_table: Vec<(usize, u32, u32)> = Vec::with_capacity(ANS_TABLE_SIZE);
    let mut cum = 0u32;
    for (idx, &nf) in normalized.iter().enumerate() {
        if nf > 0 {
            for _ in 0..nf {
                decode_table.push((idx, nf, cum));
            }
            cum += nf;
        }
    }
    // Pad to exactly TABLE_SIZE (rounding may leave a gap of 1-2 entries)
    if let Some(&first) = decode_table.first() {
        while decode_table.len() < ANS_TABLE_SIZE {
            decode_table.push(first);
        }
    }

    // ── Decode — raw byte I/O after align_to_byte ────────────────────────────
    stream.align_to_byte()?;

    let mut state = u32::from_le_bytes(
        stream
            .read_raw_bytes::<4>()
            .map_err(|e| Error::FailedToDecode(format!("state: {e}")))?,
    );

    let mut result: Vec<T> = Vec::with_capacity(len);

    for _ in 0..len {
        let table_idx = (state % ANS_TABLE_SIZE as u32) as usize;
        let (sym_idx, freq, start) = decode_table[table_idx];
        result.push(from_idx::<N, T>(sym_idx));
        state = freq * (state / ANS_TABLE_SIZE as u32) + (table_idx as u32 - start);

        while state < ANS_TABLE_SIZE as u32 {
            let word = u16::from_le_bytes(
                stream
                    .read_raw_bytes::<2>()
                    .map_err(|e| Error::FailedToDecode(format!("renorm: {e}")))?,
            );
            state = (state << RENORM_SHIFT_BITS) | word as u32;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{BitStreamReader, BitStreamWriter};

    fn roundtrip(data: &[i16]) -> Vec<i16> {
        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            encode_raw::<BYTE_SIZE_I16, i16, _>(data, &mut w).unwrap();
            w.flush().unwrap();
        }
        decode_raw::<BYTE_SIZE_I16, i16, _>(&mut BitStreamReader::new(Cursor::new(buf))).unwrap()
    }

    #[test]
    fn single_value() {
        let d = vec![42i16; 100];
        assert_eq!(roundtrip(&d), d);
    }

    #[test]
    fn two_values() {
        let d = vec![10i16, 20, 10, 20, 10, 20, 10, 20];
        assert_eq!(roundtrip(&d), d);
    }

    #[test]
    fn small_data() {
        let d = vec![1i16, 2, 3, 4, 5, 4, 3, 2, 1, 0];
        assert_eq!(roundtrip(&d), d);
    }

    #[test]
    fn dct_like() {
        let mut d = vec![0i16; 50];
        d.extend([1i16, -1, 2, -2, 1, -1, 0, 0, 0, 0]);
        assert_eq!(roundtrip(&d), d);
    }

    #[test]
    fn larger_data() {
        let d: Vec<i16> = (0..1000)
            .map(|i| match i % 100 {
                0..=64 => 0,
                65..=89 => ((i % 20) as i16) - 10,
                _ => ((i % 200) as i16) - 100,
            })
            .collect();
        assert_eq!(roundtrip(&d), d);
    }
}
