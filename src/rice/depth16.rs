//! Rice/Golomb coding for signed 16-bit residuals.
//!
//! The bitstream format is:
//!
//!     zigzag(value)
//!         -> quotient = value >> k
//!         -> remainder = value & ((1 << k) - 1)
//!         -> quotient zero bits
//!         -> terminating one bit
//!         -> k remainder bits
//!
//! The unary portion is therefore zeroes followed by one.
//!
//! Signed values use zig-zag mapping so small positive and negative residuals
//! both receive short codes.

use std::io::{Read, Write};

use crate::{BitStreamReader, BitStreamWriter, Error, Result};

const MAX_K: u8 = 15;

/// Encode a signed i16 residual using Rice coding.
///
/// `k` must be in `0..=15`.
pub(crate) fn encode<W>(k: u32, x: i32, stream: &mut BitStreamWriter<W>) -> Result<()>
where
    W: Write,
{
    let k = validate_k(k)?;

    let x = i16::try_from(x).map_err(|_| Error::InvalidData)?;
    let mapped = zigzag_encode(x);

    let quotient = mapped >> k;
    let remainder = mapped & remainder_mask(k);

    // Unary quotient: q zeroes followed by one.
    stream.write_zeros(quotient as usize)?;
    stream.write_bits(1u128, 1)?;

    if k != 0 {
        stream.write_bits(remainder as u128, k as usize)?;
    }

    Ok(())
}

/// Decode one signed i16 residual.
///
/// Truncated entropy-coded data is always rejected.
pub(crate) fn decode<R>(k: u32, stream: &mut BitStreamReader<R>) -> Result<i32>
where
    R: Read,
{
    let k = validate_k(k)?;

    let quotient = read_unary_quotient(stream)?;

    let remainder = if k == 0 {
        0u32
    } else {
        stream
            .read_bits(k as usize)?
            .ok_or(Error::UnexpectedEndOfStream)? as u32
    };

    let mapped = (quotient << k) | remainder;

    // Zig-zag encoding of an i16 occupies exactly 16 bits.
    if mapped > u16::MAX as u32 {
        return Err(Error::InvalidData);
    }

    Ok(zigzag_decode(mapped as u16) as i32)
}

/// Choose a Rice parameter from four neighboring 16-bit samples.
///
/// The activity measure is deliberately computed in i32 so subtraction cannot
/// overflow at the edges of the u16 range.
pub(crate) fn k(a: u16, c: u16, b: u16, d: u16) -> u32 {
    let activity = abs_diff(a, b) as u32 + abs_diff(b, c) as u32 + abs_diff(c, d) as u32;

    choose_k(activity, MAX_K) as u32
}

#[inline]
fn validate_k(k: u32) -> Result<u8> {
    if k > MAX_K as u32 {
        return Err(Error::InvalidData);
    }

    Ok(k as u8)
}

#[inline]
fn remainder_mask(k: u8) -> u32 {
    if k == 0 {
        0
    } else {
        (1u32 << k) - 1
    }
}

#[inline]
fn zigzag_encode(value: i16) -> u16 {
    let value = value as i32;

    // Produces:
    //
    //      0 -> 0
    //     -1 -> 1
    //      1 -> 2
    //     -2 -> 3
    //      2 -> 4
    //
    // The operation is performed in i32 to avoid signed i16 overflow.
    ((value << 1) ^ (value >> 31)) as u16
}

#[inline]
fn zigzag_decode(value: u16) -> i16 {
    let value = value as i32;

    ((value >> 1) ^ -(value & 1)) as i16
}

#[inline]
fn read_unary_quotient<R>(stream: &mut BitStreamReader<R>) -> Result<u32>
where
    R: Read,
{
    let mut quotient = 0u32;

    loop {
        match stream.read_bit()? {
            Some(true) => return Ok(quotient),
            Some(false) => {
                quotient = quotient.checked_add(1).ok_or(Error::InvalidData)?;

                // An i16 zig-zag value is at most 65535. Consequently the
                // quotient can never exceed 65535.
                if quotient > u16::MAX as u32 {
                    return Err(Error::InvalidData);
                }
            }
            None => return Err(Error::UnexpectedEndOfStream),
        }
    }
}

#[inline]
fn abs_diff(a: u16, b: u16) -> u16 {
    a.abs_diff(b)
}

#[inline]
fn choose_k(activity: u32, max_k: u8) -> u8 {
    let mut k = 0u8;

    while k < max_k && (3u32 << k) < activity {
        k += 1;
    }

    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitStreamReader, BitStreamWriter};

    fn round_trip(value: i32, k: u32) -> Result<i32> {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode(k, value, &mut writer)?;
            writer.flush()?;
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());
        decode(k, &mut reader)
    }

    #[test]
    fn zigzag_mapping_is_correct() {
        let values = [
            (0i16, 0u16),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (2, 4),
            (-3, 5),
            (3, 6),
            (-32768, 65535),
            (32767, 65534),
        ];

        for (value, expected) in values {
            assert_eq!(zigzag_encode(value), expected);
            assert_eq!(zigzag_decode(expected), value);
        }
    }

    #[test]
    fn round_trip_all_i16_values_for_small_k() {
        // Testing every value against every possible k is useful but can
        // generate a lot of unary bits at k=0. Exercise the full domain with
        // representative k values and all values with the practically useful
        // range.
        for value in i16::MIN..=i16::MAX {
            for k in [0u32, 1, 2, 3, 4, 5, 7, 8, 10, 12, 15] {
                let decoded = round_trip(value as i32, k).unwrap();

                assert_eq!(
                    decoded, value as i32,
                    "round trip failed for value={} k={}",
                    value, k
                );
            }
        }
    }

    #[test]
    fn round_trip_common_residuals_for_all_k() {
        let values = [
            i32::MIN + 1,
            -32768,
            -32767,
            -16384,
            -8192,
            -4096,
            -1024,
            -256,
            -128,
            -64,
            -32,
            -16,
            -8,
            -4,
            -2,
            -1,
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            1024,
            4096,
            8192,
            16384,
            32766,
            32767,
        ];

        for &value in &values {
            for k in 0..=MAX_K as u32 {
                assert_eq!(
                    round_trip(value, k).unwrap(),
                    value,
                    "failed for value={} k={}",
                    value,
                    k
                );
            }
        }
    }

    #[test]
    fn rejects_out_of_range_input() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        assert!(encode(0, -32769, &mut writer).is_err());
        assert!(encode(0, 32768, &mut writer).is_err());
    }

    #[test]
    fn rejects_invalid_k() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        assert!(encode(16, 0, &mut writer).is_err());
        assert!(encode(u32::MAX, 0, &mut writer).is_err());
    }

    #[test]
    fn rejects_truncated_unary_code() {
        let data = [0u8];

        let mut reader = BitStreamReader::new(data.as_slice());

        assert!(matches!(
            decode(0, &mut reader),
            Err(Error::UnexpectedEndOfStream)
        ));
    }

    #[test]
    fn rejects_empty_stream() {
        let mut reader = BitStreamReader::new([].as_slice());

        assert!(matches!(
            decode(0, &mut reader),
            Err(Error::UnexpectedEndOfStream)
        ));
    }

    #[test]
    fn sequential_stream_round_trip() {
        let values = [
            -32768i32, -16384, -1024, -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32,
            64, 128, 1024, 16384, 32767,
        ];

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            for &value in &values {
                encode(7, value, &mut writer).unwrap();
            }

            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &values {
            let actual = decode(7, &mut reader).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn k_is_bounded() {
        for a in [0u16, 1, 127, 255, 1023, 32767, 65534, 65535] {
            for b in [0u16, 1, 127, 255, 1023, 32767, 65534, 65535] {
                for c in [0u16, 1, 127, 255, 1023, 32767, 65534, 65535] {
                    for d in [0u16, 1, 127, 255, 1023, 32767, 65534, 65535] {
                        assert!(k(a, c, b, d) <= MAX_K as u32);
                    }
                }
            }
        }
    }

    #[test]
    fn k_is_zero_for_constant_region() {
        assert_eq!(k(1000, 1000, 1000, 1000), 0);
        assert_eq!(k(0, 0, 0, 0), 0);
        assert_eq!(k(65535, 65535, 65535, 65535), 0);
    }

    #[test]
    fn high_activity_gets_higher_parameter() {
        let flat = k(1000, 1000, 1000, 1000);
        let active = k(0, 65535, 0, 65535);

        assert!(active >= flat);
    }

    #[test]
    fn signed_extremes_round_trip() {
        for &value in &[-32768i32, -32767, -1, 0, 1, 32766, 32767] {
            for k in 0..=15 {
                assert_eq!(round_trip(value, k).unwrap(), value);
            }
        }
    }
}
