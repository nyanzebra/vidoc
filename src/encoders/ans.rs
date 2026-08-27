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
const U8_COMPRESSION_RATIO: usize = 8;
const I16_COMPRESSION_RATIO: usize = 4;
const MIN_OUTPUT_BUFFER_SIZE: usize = 1024;

pub(crate) fn encode<T>(data: &[T], stream: &mut BitStreamWriter<impl Write>) -> Result<()>
where
    T: crate::ToBytes,
{
    let data = data.iter().flat_map(|x| x.to_bytes()).collect::<Vec<u8>>();
    encode_raw::<BYTE_SIZE_U8, u8, _>(&data, stream)?;
    Ok(())
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

    let array_size = 1 << (BITS_PER_BYTE * N);
    if array_size > MAX_ARRAY_SIZE {
        return Err(Error::UnsupportedAnsDataType(N * BITS_PER_BYTE));
    }

    // ── Frequency table ──────────────────────────────────────────────────────
    let mut freq_table = vec![0u32; array_size];
    for &symbol in data {
        let idx = if N == BYTE_SIZE_I16 {
            (symbol.to_i16().unwrap() as i32 + I16_OFFSET) as usize
        } else {
            symbol.to_u8().unwrap() as usize
        };
        freq_table[idx] += 1;
    }

    let mut unique_count = 0usize;
    for &count in &freq_table {
        if count > 0 {
            unique_count += 1;
        }
    }

    let mut symbol_map = Vec::with_capacity(unique_count);
    for (idx, &count) in freq_table.iter().enumerate() {
        if count > 0 {
            let symbol = if N == BYTE_SIZE_I16 {
                T::from(idx as i32 - I16_OFFSET).unwrap()
            } else {
                T::from(idx).unwrap()
            };
            symbol_map.push((symbol, count));
        }
    }

    // ── Header (through the bit layer) ───────────────────────────────────────
    // write_val writes the full bit-width big-endian through the accumulator.
    // The decoder reads these with read_val/read_array through the same layer.
    stream.write_val(unique_count as u32)?;
    for (symbol, count) in &symbol_map {
        let bytes = symbol.to_be_bytes();
        stream.write_bytes(&bytes)?; // bit layer; matches read_array on decode
        stream.write_val(*count)?;
    }
    stream.align_to_byte()?;

    // data.len() goes through the bit layer into the accumulator.
    stream.write_val(data.len() as u32)?;

    if unique_count == 1 {
        // flush() drains the accumulator so the header is fully committed.
        stream.flush()?;
        return Ok(());
    }

    // ── Build encode table ───────────────────────────────────────────────────
    const TABLE_SIZE: usize = ANS_TABLE_SIZE;
    let total_freq: u32 = freq_table.iter().sum();

    let mut normalized_freqs = vec![0u32; array_size];
    let mut normalized_total = 0u32;
    for (idx, &freq) in freq_table.iter().enumerate() {
        if freq > 0 {
            let nf = ((freq as u64 * TABLE_SIZE as u64) / total_freq as u64) as u32;
            let nf = nf.max(MIN_NORMALIZED_FREQ);
            normalized_freqs[idx] = nf;
            normalized_total += nf;
        }
    }

    if normalized_total != TABLE_SIZE as u32 {
        let diff = TABLE_SIZE as i64 - normalized_total as i64;
        let (mut max_idx, mut max_freq) = (0, 0);
        for (idx, &freq) in normalized_freqs.iter().enumerate() {
            if freq > max_freq {
                max_freq = freq;
                max_idx = idx;
            }
        }
        normalized_freqs[max_idx] =
            (normalized_freqs[max_idx] as i64 + diff).max(MIN_NORMALIZED_FREQ as i64) as u32;
    }

    let mut encode_table = vec![(0u32, 0u32); array_size];
    let mut cumulative_freq = 0u32;
    for (idx, &freq) in normalized_freqs.iter().enumerate() {
        if freq > 0 {
            encode_table[idx] = (freq, cumulative_freq);
            cumulative_freq += freq;
        }
    }

    // ── ANS encode loop ──────────────────────────────────────────────────────
    let mut state = TABLE_SIZE as u32;
    let table_size_u32 = TABLE_SIZE as u32;
    let renorm_threshold = 1u32 << RENORM_SHIFT_BITS;

    let estimated = if N == BYTE_SIZE_I16 {
        data.len() / I16_COMPRESSION_RATIO
    } else {
        data.len() / U8_COMPRESSION_RATIO
    };
    let mut output_words = Vec::with_capacity(estimated.max(MIN_OUTPUT_BUFFER_SIZE));

    for &symbol in data.iter().rev() {
        let idx = if N == BYTE_SIZE_I16 {
            (symbol.to_i16().unwrap() as i32 + I16_OFFSET) as usize
        } else {
            symbol.to_u8().unwrap() as usize
        };
        let (freq, cum) = encode_table[idx];
        let threshold = freq * renorm_threshold;
        while state >= threshold {
            output_words.push((state & RENORM_MASK) as u16);
            state >>= RENORM_SHIFT_BITS;
        }
        let q = state / freq;
        let r = state % freq;
        state = q * table_size_u32 + cum + r;
    }

    // ── Write ANS output as raw bytes ────────────────────────────────────────
    // flush() drains the accumulator (which holds data.len()) to the underlying
    // writer before we switch to raw byte I/O via write_aligned_bytes.
    // write_aligned_bytes calls aligned_writer() which is only valid when the
    // accumulator is empty — flush() guarantees that.
    stream.flush()?;

    stream.write_aligned_bytes(&state.to_le_bytes())?;

    let mut byte_buf = Vec::with_capacity(output_words.len() * 2);
    for &word in output_words.iter().rev() {
        byte_buf.extend_from_slice(&word.to_le_bytes());
    }
    stream.write_aligned_bytes(&byte_buf)?;

    Ok(())
}

pub(crate) fn decode<T, R>(stream: &mut BitStreamReader<R>) -> Result<Vec<T>>
where
    T: crate::FromBytes,
    R: Read,
{
    let raw: Vec<u8> = decode_raw(stream)?;
    let mut start = 0;
    let item_size = std::mem::size_of::<T>();
    let mut res = Vec::with_capacity(raw.len().checked_div(item_size).unwrap_or(0));
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
    let array_size = 1 << (BITS_PER_BYTE * N);
    if array_size > MAX_ARRAY_SIZE {
        return Err(Error::UnsupportedAnsDataType(N * BITS_PER_BYTE));
    }

    // ── Header (bit layer) ───────────────────────────────────────────────────
    let num_symbols = stream
        .read_val::<u32>()?
        .ok_or(Error::FailedToDecode("num_symbols".to_owned()))? as usize;

    let mut freq_table = vec![0u32; array_size];
    let mut symbols = Vec::with_capacity(num_symbols);

    for _ in 0..num_symbols {
        let byte_array = stream.read_array::<N>()?; // bit layer; matches write_bytes
        let symbol = T::from_be_bytes(&byte_array);
        let freq = stream
            .read_val::<u32>()?
            .ok_or(Error::FailedToDecode("frequency".to_owned()))?;
        let idx = if N == BYTE_SIZE_I16 {
            (symbol.to_i16().unwrap() as i32 + I16_OFFSET) as usize
        } else {
            symbol.to_u8().unwrap() as usize
        };
        freq_table[idx] = freq;
        symbols.push(symbol);
    }

    stream.align_to_byte()?;

    let len = stream
        .read_val::<u32>()?
        .ok_or(Error::FailedToDecode("len".to_owned()))? as usize;

    let mut result = Vec::with_capacity(len);

    if num_symbols == 1 {
        result.resize(len, symbols[0]);
        return Ok(result);
    }

    // ── Build decode table ───────────────────────────────────────────────────
    const TABLE_SIZE: usize = ANS_TABLE_SIZE;
    let total_freq: u32 = freq_table.iter().sum();

    let mut normalized_freqs = vec![0u32; array_size];
    let mut normalized_total = 0u32;
    for (idx, &freq) in freq_table.iter().enumerate() {
        if freq > 0 {
            let nf = ((freq as u64 * TABLE_SIZE as u64) / total_freq as u64) as u32;
            let nf = nf.max(MIN_NORMALIZED_FREQ);
            normalized_freqs[idx] = nf;
            normalized_total += nf;
        }
    }

    if normalized_total != TABLE_SIZE as u32 {
        let diff = TABLE_SIZE as i64 - normalized_total as i64;
        let (mut max_idx, mut max_freq) = (0, 0);
        for (idx, &freq) in normalized_freqs.iter().enumerate() {
            if freq > max_freq {
                max_freq = freq;
                max_idx = idx;
            }
        }
        normalized_freqs[max_idx] =
            (normalized_freqs[max_idx] as i64 + diff).max(MIN_NORMALIZED_FREQ as i64) as u32;
    }

    let mut decode_table = Vec::with_capacity(TABLE_SIZE);
    let mut cumulative = 0u32;
    for (idx, &freq) in normalized_freqs.iter().enumerate().take(array_size) {
        if freq > 0 {
            for _ in 0..freq {
                decode_table.push((idx, freq, cumulative));
            }
            cumulative += freq;
        }
    }
    while decode_table.len() < TABLE_SIZE {
        if let Some(&entry) = decode_table.first() {
            decode_table.push(entry);
        } else {
            break;
        }
    }

    // ── ANS decode — raw byte I/O ────────────────────────────────────────────
    // align_to_byte() discards padding after data.len(), leaving the reader
    // on the byte boundary where the encoder wrote the raw ANS output.
    // read_raw_bytes uses aligned_reader() — safe only after align_to_byte().
    stream.align_to_byte()?;

    let mut state = u32::from_le_bytes(
        stream
            .read_raw_bytes::<4>()
            .map_err(|e| Error::FailedToDecode(format!("initial state: {e}")))?,
    );

    for _ in 0..len {
        let table_idx = (state % TABLE_SIZE as u32) as usize;
        if table_idx >= decode_table.len() {
            return Err(Error::FailedToDecode(format!(
                "table_idx {table_idx} >= {}",
                decode_table.len()
            )));
        }
        let (symbol_idx, freq, start) = decode_table[table_idx];
        let symbol = if N == BYTE_SIZE_I16 {
            T::from(symbol_idx as i32 - I16_OFFSET).unwrap()
        } else {
            T::from(symbol_idx).unwrap()
        };
        result.push(symbol);

        state = freq * (state / TABLE_SIZE as u32) + (table_idx as u32 - start);

        while state < TABLE_SIZE as u32 {
            let bits = u16::from_le_bytes(
                stream
                    .read_raw_bytes::<2>()
                    .map_err(|e| Error::FailedToDecode(format!("renorm bits: {e}")))?,
            );
            state = (state << RENORM_SHIFT_BITS) | bits as u32;
        }
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
            encode_raw::<BYTE_SIZE_I16, i16, _>(data, &mut w).unwrap();
            w.flush().unwrap();
        }
        let mut r = BitStreamReader::new(Cursor::new(buf));
        decode_raw::<BYTE_SIZE_I16, i16, _>(&mut r).unwrap()
    }

    #[test]
    fn test_ans_single_value() {
        let data = vec![42i16; 100];
        assert_eq!(roundtrip_raw(&data), data);
    }

    #[test]
    fn test_ans_two_values() {
        let data = vec![10i16, 20, 10, 20, 10, 20, 10, 20];
        assert_eq!(roundtrip_raw(&data), data);
    }

    #[test]
    fn test_ans_small_data() {
        let data = vec![1i16, 2, 3, 4, 5, 4, 3, 2, 1, 0];
        assert_eq!(roundtrip_raw(&data), data);
    }

    #[test]
    fn test_ans_dct_like() {
        let mut data = vec![0i16; 50];
        data.extend([1i16, -1, 2, -2, 1, -1, 0, 0, 0, 0]);
        assert_eq!(roundtrip_raw(&data), data);
    }

    #[test]
    fn test_ans_larger_data() {
        let data: Vec<i16> = (0..1000)
            .map(|i| match i % 100 {
                0..=64 => 0,
                65..=89 => ((i % 20) as i16) - 10,
                _ => ((i % 200) as i16) - 100,
            })
            .collect();
        let decoded = roundtrip_raw(&data);
        assert_eq!(data, decoded);
    }
}
