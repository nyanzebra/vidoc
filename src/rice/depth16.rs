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
// It appears that using just unsigned as is for the rice coding is not great...
// it leads to a lot of wasted bits.
// This operation tries to make the lower bits more significant.
pub(crate) fn encode<W>(k: u32, x: i32, stream: &mut BitStreamWriter<W>) -> Result<()>
where
    W: Write,
{
    // EXACT COPY of reference implementation logic, adapted for i32 input
    let x = ((x >> 30) ^ (2 * x)) as u32;
    let high_bits = x >> k;

    // Write the unary encoding: high_bits zeros followed by a 1
    stream.write_zeros(high_bits as usize)?;
    stream.write_bits(1u128, 1)?;

    // Write low bits if any
    if k > 0 {
        stream.write_bits((x & ((1 << k) - 1)) as u128, k as usize)?;
    }

    Ok(())
}

pub(crate) fn decode<R>(k: u32, stream: &mut BitStreamReader<R>) -> Result<i32>
where
    R: Read,
{
    // Read unary: count leading zeros until we hit a 1
    // This exactly matches the reference implementation
    let mut high_bits = 0;
    while let Some(bit) = stream.read_bits(1)? {
        if bit == 0 {
            high_bits += 1;
        } else {
            break;
        }
    }

    // Read low bits exactly as reference - use regular read for k bits
    let low_bits = if k == 0 {
        0
    } else {
        stream.read_bits(k as usize)?.unwrap_or(0) as u32
    };

    let x = ((high_bits as u32) << k) | low_bits;

    // EXACT COPY of reference decode transform
    let result = (x as i32 >> 1) ^ ((x << 31) as i32 >> 31);
    Ok(result)
}

// https://blog.tempus-ex.com/hello-video-codec/
// We want to choose k values relatively well, as that will optimize are end compression
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
    use super::*;

    #[test]
    fn test_simple_encode_decode() {
        // Test a simple value first
        let value = 0i32;
        let k = 0u32;
        let mut buf = vec![];
        {
            let mut dest = BitStreamWriter::new(&mut buf);
            encode(k, value, &mut dest).unwrap();
            dest.flush().unwrap();
        }
        let mut bitstream = BitStreamReader::new(&*buf);
        let decoded = decode(k, &mut bitstream).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_encode_decode_u8() {
        let input = [
            148, 131, 111, 147, 130, 110, 146, 129, 109, 149, 132, 112, 147, 130, 110, 149, 132,
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
                let mut bitstream = BitStreamReader::new(&*buf);
                let decoded = decode(k, &mut bitstream).unwrap();
                assert_eq!(x, decoded);
            }
        }
    }
}
