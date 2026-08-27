use std::io::{Read, Write};

use crate::{BitStreamReader, BitStreamWriter, Result};

pub(crate) fn encode<W>(k: u32, x: i32, stream: &mut BitStreamWriter<W>) -> Result<()>
where
    W: Write,
{
    let x = ((x >> 30) ^ (2 * x)) as u32;
    let high_bits = x >> k;

    // Write `high_bits` zero bits in chunks (write_val_bits max is the type width).
    let mut remaining = high_bits;
    while remaining > 0 {
        let chunk = remaining.min(32);
        stream.write_val_bits(chunk, 0u32)?;
        remaining -= chunk;
    }
    // Terminating 1 bit
    stream.write_bit(true)?;

    // Write k low bits
    if k > 0 {
        stream.write_val_bits(k, x & ((1 << k) - 1))?;
    }

    Ok(())
}

pub(crate) fn decode<R>(k: u32, stream: &mut BitStreamReader<R>) -> Result<i32>
where
    R: Read,
{
    // count_leading_zeros uses read_unary::<1>() — counts zeros, consumes the
    // terminating 1 bit automatically.
    let high_bits = stream.count_leading_zeros()? as u32;

    let low_bits: u32 = if k == 0 {
        0
    } else {
        stream.read_val_bits::<u32>(k)?.unwrap_or(0)
    };

    let x = (high_bits << k) | low_bits;
    let result = (x as i32 >> 1) ^ ((x << 31) as i32 >> 31);
    Ok(result)
}

pub(crate) fn k(a: u16, c: u16, b: u16, d: u16) -> u32 {
    let activity =
        (d as i32 - b as i32).abs() + (b as i32 - c as i32).abs() + (c as i32 - a as i32).abs();
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

    fn roundtrip(k: u32, value: i32) -> i32 {
        let mut buf = Vec::new();
        {
            let mut w = BitStreamWriter::new(&mut buf);
            encode(k, value, &mut w).unwrap();
            w.flush().unwrap();
        }
        let mut r = BitStreamReader::new(Cursor::new(buf));
        decode(k, &mut r).unwrap()
    }

    #[test]
    fn test_simple_encode_decode() {
        assert_eq!(roundtrip(0, 0), 0);
    }

    #[test]
    fn test_encode_decode_u8() {
        let input = [
            148i32, 131, 111, 147, 130, 110, 146, 129, 109, 149, 132, 112, 147, 130, 110, 149, 132,
            112, 144, 127, 107, 147, 130, 110, 148, 131, 111, 150, 133, 113, 151, 134, 114, 149,
            132, 112, 151, 134, 114, 149, 132, 112, 150, 133, 113, 152, 135, 115, 154, 139, 118,
            154, 139, 118, 153, 138, 117, 152, 137, 116, 150, 135, 114, 153, 138, 117, 151, 136,
            115, 145, 130, 109, 152, 137, 114, 158, 143, 120, 158, 143, 120, 151, 136, 113, 154,
            142, 120, 156, 144, 122, 157, 146, 126, 155, 147, 128, 154, 146, 133, 167,
        ];
        for k in 0..10 {
            for x in input {
                assert_eq!(roundtrip(k, x), x, "k={k}, x={x}");
            }
        }
    }
}
