// https://unix4lyfe.org/rice-coding/?ref=blog.tempus-ex.com
// TLDR:
// m = 2^k
// Let q = x / m (round fractions down)
// Write out q binary ones.
// Write out a binary zero.
// (Some people prefer to do it the other way - zeroes followed by a one)
// Write out the last k bits of x

use std::io::{Read, Write};

use crate::{BitStreamReader, BitStreamWriter, Result};

// Taken from https://github.com/tempus-ex/hello-video-codec/blob/main/src/codec.rs
pub(crate) fn encode<W>(k: u16, x: i16, stream: &mut BitStreamWriter<W>) -> Result<()>
where
    W: Write,
{
    let x = ((x as i32 >> 30) ^ (2 * x as i32)) as u32;
    let high_bits = x >> k;

    // Write `high_bits` zero bits followed by a terminating 1 bit.
    // write_val_bits(bits, value): bits first, value second.
    if high_bits > 0 {
        // We can't write more than 64 zeros at a time with a u64, so loop for
        // very large values (pathological; normal video never hits this).
        let mut remaining = high_bits;
        while remaining > 0 {
            let chunk = remaining.min(64);
            stream.write_val_bits(chunk, 0u64)?;
            remaining -= chunk;
        }
    }
    // Terminating 1 bit
    stream.write_bit(true)?;

    // Write the k low bits of x
    if k > 0 {
        stream.write_val_bits(k as u32, x & ((1 << k) - 1))?;
    }

    Ok(())
}

pub(crate) fn decode<R>(k: u16, stream: &mut BitStreamReader<R>) -> Result<i16>
where
    R: Read,
{
    // Count leading zero bits until a 1. count_leading_zeros() consumes
    // the terminating 1 bit automatically (it uses read_unary::<1>()).
    let high_bits = stream.count_leading_zeros()? as u32;

    // Read k low bits
    let low_bits: u32 = if k == 0 {
        0
    } else {
        stream.read_val_bits::<u32>(k as u32)?.unwrap_or(0)
    };

    let x = (high_bits << k) | low_bits;

    // Undo the zigzag encoding
    let result = (x as i32 >> 1) ^ ((x << 31) as i32 >> 31);
    Ok(result as i16)
}

// https://blog.tempus-ex.com/hello-video-codec/
pub(crate) fn k(a: u8, c: u8, b: u8, d: u8) -> u16 {
    let activity =
        (d as i16 - b as i16).abs() + (b as i16 - c as i16).abs() + (c as i16 - a as i16).abs();
    let mut k = 0;
    while 3 << k < activity {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{BitStreamReader, BitStreamWriter};

    // -------------------------------------------------------------------------
    // new_with_data is gone — BitStreamReader::new(Cursor::new(data)) replaces it.
    // The old new_with_data pre-loaded the bitvec from the stream. With bitstream-io
    // that's unnecessary: BitReader reads lazily on demand, so just wrap the data
    // in a Cursor and pass it to BitStreamReader::new.
    // -------------------------------------------------------------------------

    fn make_reader(data: Vec<u8>) -> BitStreamReader<Cursor<Vec<u8>>> {
        BitStreamReader::new(Cursor::new(data))
    }

    fn roundtrip(k: u16, value: i16) -> i16 {
        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            encode(k, value, &mut w).unwrap();
            w.flush().unwrap();
        }
        let mut r = make_reader(buf);
        decode(k, &mut r).unwrap()
    }

    #[test]
    fn test_bit_stream_basic() {
        let mut buf = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buf);

        // Write 3 zeros, 1 one, then 2 bits value=2 (binary 10)
        writer.write_val_bits(3u32, 0u8).expect("write zeros");
        writer.write_bit(true).expect("write one");
        writer.write_val_bits(2u32, 2u8).expect("write two bits");
        writer.flush().expect("flush");

        let mut reader = make_reader(buf);

        // Count zeros until the 1
        let zero_count = reader.count_leading_zeros().expect("count zeros");
        assert_eq!(zero_count, 3);

        // Read 2 bits
        let two_bits = reader.read_val_bits::<u8>(2).expect("read 2 bits");
        assert_eq!(two_bits, Some(2));
    }

    #[test]
    fn test_rice_specific_failure() {
        assert_eq!(roundtrip(0, 131), 131);
    }

    #[test]
    fn test_negative_value_debug() {
        assert_eq!(roundtrip(0, -128), -128);
    }

    #[test]
    fn test_rice_extreme_values() {
        let extreme = [-128i16, -127, -1, 0, 1, 127, -32768, 32767];
        for &value in &extreme {
            for k in 0u16..=5 {
                assert_eq!(roundtrip(k, value), value, "mismatch: k={k}, value={value}");
            }
        }
    }

    #[test]
    fn test_rice_sequential_values() {
        let test_values = [-128i16, -1, 0, 1, 127];
        let k = 0u16;

        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            for &v in &test_values {
                encode(k, v, &mut w).unwrap();
            }
            w.flush().unwrap();
        }

        let mut r = make_reader(buf);
        for &expected in &test_values {
            assert_eq!(decode(k, &mut r).unwrap(), expected);
        }
    }

    #[test]
    fn test_rice_encode_decode_consistency() {
        let test_values = [0i16, 1, -1, 2, -2, 3, -3, 100, -100, 255, -255];
        for k in 0u16..=5 {
            for &value in &test_values {
                assert_eq!(roundtrip(k, value), value, "k={k}, value={value}");
            }
        }
    }

    #[test]
    fn test_encode_decode_u8() {
        let input = [
            148i16, 131, 111, 147, 130, 110, 146, 129, 109, 149, 132, 112, 147, 130, 110, 149, 132,
            112, 144, 127, 107, 147, 130, 110, 148, 131, 111, 150, 133, 113, 151, 134, 114, 149,
            132, 112, 151, 134, 114, 149, 132, 112, 150, 133, 113, 152, 135, 115, 154, 139, 118,
            154, 139, 118, 153, 138, 117, 152, 137, 116, 150, 135, 114, 153, 138, 117, 151, 136,
            115, 145, 130, 109, 152, 137, 114, 158, 143, 120, 158, 143, 120, 151, 136, 113, 154,
            142, 120, 156, 144, 122, 157, 146, 126, 155, 147, 128, 154, 146, 133, 167,
        ];
        let mut buf = vec![];
        for k in 0..10 {
            for x in input {
                buf.clear();
                {
                    let mut dest = BitStreamWriter::new(&mut buf);
                    encode(k, x, &mut dest).unwrap();
                    dest.flush().unwrap();
                }
                let mut r = make_reader(buf.clone());
                assert_eq!(decode(k, &mut r).unwrap(), x);
            }
        }
    }

    #[test]
    fn test_rice_1024_values_image_simulation() {
        let pattern = [-2i16, -1, 0, 1, 2];
        let test_values: Vec<i16> = (0..1024).map(|i| pattern[i % pattern.len()]).collect();
        let k = 0u16;

        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            for &v in &test_values {
                encode(k, v, &mut w).unwrap();
            }
            w.flush().unwrap();
        }

        let mut r = make_reader(buf);
        for (i, &expected) in test_values.iter().enumerate() {
            let got = decode(k, &mut r).unwrap();
            assert_eq!(got, expected, "mismatch at index {i}");
        }
    }
}
