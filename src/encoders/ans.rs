use std::io::{Read, Write};

use num_traits::{ops::bytes::ToBytes, FromBytes, PrimInt};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    error::{Error, Result},
};

// ANS encoding/decoding constants
const BITS_PER_BYTE: usize = 8;
// Maximum supported array size (2^16 for i16)
const MAX_ARRAY_SIZE: usize = 65536;
// Offset to map i16 range to positive indices
const I16_OFFSET: i32 = 32768;
// N value for u8 (1 byte)
const BYTE_SIZE_U8: usize = 1;
// N value for i16 (2 bytes)
const BYTE_SIZE_I16: usize = 2;
// ANS state table size (2^12)
const ANS_TABLE_SIZE: usize = 4096;
// Number of bits to shift during renormalization
const RENORM_SHIFT_BITS: u32 = 16;
// Mask for extracting lower 16 bits
const RENORM_MASK: u32 = 0xFFFF;
// Minimum frequency after normalization
const MIN_NORMALIZED_FREQ: u32 = 1;
// Estimated output size ratio for u8
const U8_COMPRESSION_RATIO: usize = 8;
// Estimated output size ratio for i16
const I16_COMPRESSION_RATIO: usize = 4;
// Minimum buffer size for output
const MIN_OUTPUT_BUFFER_SIZE: usize = 1024;

pub(crate) fn encode<T>(data: &[T], stream: &mut BitStreamWriter<impl Write>) -> Result<()>
where
    T: crate::ToBytes,
{
    // .to_bytes() is a trivial copy — parallelizing it adds rayon overhead with no gain.
    // A plain iterator avoids the intermediate Vec allocation from par_iter.
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

    // Calculate array size: 2^(8*N) possible values
    // N=1 (u8): 256, N=2 (i16): 65536
    let array_size = 1 << (BITS_PER_BYTE * N);

    // Array-based approach for u8 and i16 types
    // For 16-bit depth, we use scaled quantization to ensure values fit in i16
    if array_size > MAX_ARRAY_SIZE {
        return Err(Error::UnsupportedAnsDataType(N * BITS_PER_BYTE));
    }

    // Build frequency table using array
    let mut freq_table = vec![0u32; array_size];

    for &symbol in data {
        // Convert symbol to array index (handle signed types by adding offset)
        let idx = if N == BYTE_SIZE_I16 {
            // i16: add 32768 to map -32768..32767 to 0..65535
            (symbol.to_i16().unwrap() as i32 + I16_OFFSET) as usize
        } else {
            // u8 or other small types: direct indexing
            symbol.to_u8().unwrap() as usize
        };
        freq_table[idx] += 1;
    }

    // Write frequency table
    let mut unique_count = 0usize;
    // Pre-count unique symbols to allocate vec once
    for &count in freq_table.iter() {
        if count > 0 {
            unique_count += 1;
        }
    }

    let mut symbol_map = Vec::with_capacity(unique_count);
    for (idx, &count) in freq_table.iter().enumerate() {
        if count > 0 {
            // Convert index back to symbol
            let symbol = if N == BYTE_SIZE_I16 {
                T::from(idx as i32 - I16_OFFSET).unwrap()
            } else {
                T::from(idx).unwrap()
            };
            symbol_map.push((symbol, count));
        }
    }

    stream.write(unique_count)?;
    for (symbol, count) in &symbol_map {
        let bytes = symbol.to_be_bytes();
        stream.write_slice(&bytes)?;
        stream.write(*count)?;
    }
    stream.align_to_byte()?;

    // Write data length
    stream.write(data.len())?;

    // Single-value optimization
    if unique_count == 1 {
        return Ok(());
    }

    // Build encoding table
    const TABLE_SIZE: usize = ANS_TABLE_SIZE;

    // Calculate total frequency
    let total_freq: u32 = freq_table.iter().sum();

    // Normalize frequencies to sum to TABLE_SIZE
    let mut normalized_freqs = vec![0u32; array_size];
    let mut normalized_total = 0u32;
    for (idx, &freq) in freq_table.iter().enumerate() {
        if freq > 0 {
            let normalized_freq = ((freq as u64 * TABLE_SIZE as u64) / total_freq as u64) as u32;
            let normalized_freq = normalized_freq.max(MIN_NORMALIZED_FREQ);
            normalized_freqs[idx] = normalized_freq;
            normalized_total += normalized_freq;
        }
    }

    // Adjust to make frequencies sum to exactly TABLE_SIZE
    if normalized_total != TABLE_SIZE as u32 {
        let diff = TABLE_SIZE as i64 - normalized_total as i64;
        // Find symbol with largest frequency to adjust
        let mut max_idx = 0;
        let mut max_freq = 0;
        for (idx, &freq) in normalized_freqs.iter().enumerate() {
            if freq > max_freq {
                max_freq = freq;
                max_idx = idx;
            }
        }
        let new_freq =
            (normalized_freqs[max_idx] as i64 + diff).max(MIN_NORMALIZED_FREQ as i64) as u32;
        normalized_freqs[max_idx] = new_freq;
    }

    // Build encode table with normalized frequencies
    let mut encode_table = vec![(0u32, 0u32); array_size];
    let mut cumulative_freq = 0u32;
    for (idx, &freq) in normalized_freqs.iter().enumerate() {
        if freq > 0 {
            encode_table[idx] = (freq, cumulative_freq);
            cumulative_freq += freq;
        }
    }

    // Initialize ANS state
    let mut state = TABLE_SIZE as u32;
    let table_size_u32 = TABLE_SIZE as u32;
    let renorm_threshold = 1u32 << RENORM_SHIFT_BITS;

    // Encode loop with array lookups
    // Pre-allocate with estimated capacity (roughly data.len() / 8 for u8, data.len() / 4 for i16)
    let estimated_output_size = if N == BYTE_SIZE_I16 {
        data.len() / I16_COMPRESSION_RATIO
    } else {
        data.len() / U8_COMPRESSION_RATIO
    };
    let mut output_words = Vec::with_capacity(estimated_output_size.max(MIN_OUTPUT_BUFFER_SIZE));

    for &symbol in data.iter().rev() {
        let idx = if N == BYTE_SIZE_I16 {
            (symbol.to_i16().unwrap() as i32 + I16_OFFSET) as usize
        } else {
            symbol.to_u8().unwrap() as usize
        };

        let (freq, cumulative_freq) = encode_table[idx];

        // Renormalize if needed (pre-calculate threshold to avoid repeated multiplication)
        let threshold = freq * renorm_threshold;
        while state >= threshold {
            output_words.push((state & RENORM_MASK) as u16);
            state >>= RENORM_SHIFT_BITS;
        }

        // Update state
        let quotient = state / freq;
        let remainder = state % freq;
        state = quotient * table_size_u32 + cumulative_freq + remainder;
    }

    // Write output
    stream.write(state)?;

    // Write output words in reverse
    for &word in output_words.iter().rev() {
        stream.write(word)?;
    }

    Ok(())
}

pub(crate) fn decode<T, R>(stream: &mut BitStreamReader<R>) -> Result<Vec<T>>
where
    T: crate::FromBytes,
    R: Read,
{
    // Decode directly into T without an intermediate Vec<u8> re-parse loop.
    let raw: Vec<u8> = decode_raw(stream)?;
    let mut start = 0;
    let item_size = std::mem::size_of::<T>();
    let mut res = Vec::with_capacity(if item_size > 0 { raw.len() / item_size } else { 0 });
    while start < raw.len() {
        let (block, end) = T::from_bytes(&raw[start..]);
        debug_assert_ne!(end, 0, "from_bytes returned zero advance — infinite loop risk");
        if end == 0 { break; }
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
    // Calculate array size: 2^(8*N) possible values
    let array_size = 1 << (BITS_PER_BYTE * N);

    // Array-based approach for u8 and i16 types only
    if array_size > MAX_ARRAY_SIZE {
        return Err(Error::UnsupportedAnsDataType(N * BITS_PER_BYTE));
    }

    // Read frequency table from stream
    let num_symbols = stream
        .read::<usize>()?
        .ok_or(Error::FailedToDecode("num_symbols".to_owned()))?;

    let mut freq_table = vec![0u32; array_size];
    let mut symbols = Vec::with_capacity(num_symbols);

    for _ in 0..num_symbols {
        let byte_array = stream.read_array::<N>()?;
        let symbol = T::from_be_bytes(&byte_array);

        let freq = stream
            .read::<u32>()?
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

    // Read number of symbols to decode
    let len = stream
        .read::<usize>()?
        .ok_or(Error::FailedToDecode("len".to_owned()))?;

    let mut result = Vec::with_capacity(len);

    // Check for single-value optimization
    if num_symbols == 1 {
        result.resize(len, symbols[0]);
        return Ok(result);
    }

    // Build ANS decoding table
    const TABLE_SIZE: usize = ANS_TABLE_SIZE;

    // Calculate total frequency
    let total_freq: u32 = freq_table.iter().sum();

    // Normalize frequencies
    let mut normalized_freqs = vec![0u32; array_size];
    let mut normalized_total = 0u32;
    for (idx, &freq) in freq_table.iter().enumerate() {
        if freq > 0 {
            let normalized_freq = ((freq as u64 * TABLE_SIZE as u64) / total_freq as u64) as u32;
            let normalized_freq = normalized_freq.max(MIN_NORMALIZED_FREQ);
            normalized_freqs[idx] = normalized_freq;
            normalized_total += normalized_freq;
        }
    }

    // Adjust to make frequencies sum to exactly TABLE_SIZE
    if normalized_total != TABLE_SIZE as u32 {
        let diff = TABLE_SIZE as i64 - normalized_total as i64;
        let mut max_idx = 0;
        let mut max_freq = 0;
        for (idx, &freq) in normalized_freqs.iter().enumerate() {
            if freq > max_freq {
                max_freq = freq;
                max_idx = idx;
            }
        }
        let new_freq =
            (normalized_freqs[max_idx] as i64 + diff).max(MIN_NORMALIZED_FREQ as i64) as u32;
        normalized_freqs[max_idx] = new_freq;
    }

    // Build decode table - maps table position to (symbol_idx, freq, cumulative_start)
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

    // Pad if necessary
    while decode_table.len() < TABLE_SIZE {
        if let Some(&(idx, freq, cum)) = decode_table.first() {
            decode_table.push((idx, freq, cum));
        } else {
            break;
        }
    }

    // Initialize ANS state
    let mut state = stream
        .read::<u32>()?
        .ok_or(Error::FailedToDecode("initial state".to_owned()))?;

    // Decode symbols
    for _ in 0..len {
        let table_idx = (state % TABLE_SIZE as u32) as usize;
        if table_idx >= decode_table.len() {
            return Err(Error::FailedToDecode(format!(
                "table_idx {} >= table size {}",
                table_idx,
                decode_table.len()
            )));
        }

        let (symbol_idx, freq, start) = decode_table[table_idx];

        // Convert index back to symbol
        let symbol = if N == BYTE_SIZE_I16 {
            T::from(symbol_idx as i32 - I16_OFFSET).unwrap()
        } else {
            T::from(symbol_idx).unwrap()
        };

        result.push(symbol);

        // Update state
        state = freq * (state / TABLE_SIZE as u32) + (table_idx as u32 - start);

        // Renormalize if needed
        while state < TABLE_SIZE as u32 {
            let bits = stream
                .read::<u16>()?
                .ok_or(Error::FailedToDecode("renorm bits".to_owned()))?;
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

    #[test]
    fn test_ans_single_value() {
        // Test with all same values (should use optimization)
        let data = vec![42i16; 100];

        let mut buffer = Vec::new();
        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode_raw::<BYTE_SIZE_I16, i16, _>(&data, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        println!("Single value encoded to {} bytes", buffer.len());

        let cursor = Cursor::new(buffer);
        let mut reader = BitStreamReader::new(cursor);
        let decoded = decode_raw::<BYTE_SIZE_I16, i16, _>(&mut reader).unwrap();

        assert_eq!(data, decoded, "Single value roundtrip failed");
    }

    #[test]
    fn test_ans_two_values() {
        // Test with two distinct values
        let data = vec![10i16, 20, 10, 20, 10, 20, 10, 20];

        let mut buffer = Vec::new();
        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode_raw::<BYTE_SIZE_I16, i16, _>(&data, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        println!("Two values encoded to {} bytes", buffer.len());

        let cursor = Cursor::new(buffer);
        let mut reader = BitStreamReader::new(cursor);
        let decoded = decode_raw::<BYTE_SIZE_I16, i16, _>(&mut reader).unwrap();

        assert_eq!(data, decoded, "Two value roundtrip failed");
    }

    #[test]
    fn test_ans_small_data() {
        // Test with small diverse data
        let data = vec![1i16, 2, 3, 4, 5, 4, 3, 2, 1, 0];

        let mut buffer = Vec::new();
        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode_raw::<BYTE_SIZE_I16, i16, _>(&data, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        println!("Small data encoded to {} bytes", buffer.len());

        let cursor = Cursor::new(buffer);
        let mut reader = BitStreamReader::new(cursor);
        let decoded = decode_raw::<BYTE_SIZE_I16, i16, _>(&mut reader).unwrap();

        assert_eq!(data, decoded, "Small data roundtrip failed");
    }

    #[test]
    fn test_ans_dct_like() {
        // Test with DCT-like distribution (lots of zeros)
        let mut data = vec![0i16; 50];
        data.extend(vec![1, -1, 2, -2, 1, -1, 0, 0, 0, 0]);

        let mut buffer = Vec::new();
        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode_raw::<BYTE_SIZE_I16, i16, _>(&data, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        println!(
            "DCT-like data encoded to {} bytes for {} values",
            buffer.len(),
            data.len()
        );

        let cursor = Cursor::new(buffer);
        let mut reader = BitStreamReader::new(cursor);
        let decoded = decode_raw::<BYTE_SIZE_I16, i16, _>(&mut reader).unwrap();

        assert_eq!(data, decoded, "DCT-like roundtrip failed");
    }

    #[test]
    fn test_ans_larger_data() {
        // Test with 1000 values
        let mut data = Vec::new();
        for i in 0..1000 {
            let value = match i % 100 {
                0..=64 => 0,
                65..=89 => ((i % 20) as i16) - 10,
                _ => ((i % 200) as i16) - 100,
            };
            data.push(value);
        }

        let mut buffer = Vec::new();
        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode_raw::<BYTE_SIZE_I16, i16, _>(&data, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        println!(
            "Large data encoded to {} bytes for {} values ({:.2} bytes/value)",
            buffer.len(),
            data.len(),
            buffer.len() as f64 / data.len() as f64
        );

        let cursor = Cursor::new(buffer);
        let mut reader = BitStreamReader::new(cursor);
        let decoded = decode_raw::<BYTE_SIZE_I16, i16, _>(&mut reader).unwrap();

        assert_eq!(data.len(), decoded.len(), "Length mismatch");
        assert_eq!(data, decoded, "Large data roundtrip failed");
    }
}
