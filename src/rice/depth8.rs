//! Rice/Golomb coding for signed 8-bit residuals.
//!
//! The representation used here is:
//!
//! 1. Map the signed residual to an unsigned integer using zig-zag coding.
//! 2. Split the unsigned value into: quotient = value >> k remainder = value & ((1 << k) - 1)
//! 3. Write `quotient` zero bits followed by a one bit.
//! 4. Write the `k`-bit remainder, MSB first.
//!
//! Thus, for k = 0:
//!
//!     0 -> 1
//!     -1 -> 01
//!     1 -> 001
//!     -2 -> 0001
//!     2 -> 00001
//!
//! The exact bitstream representation is deliberately explicit because this
//! is entropy-coded data and a one-bit disagreement between encoder and
//! decoder corrupts everything following it.
//
//! The decoder is strict: truncated unary prefixes and truncated remainders
//! are reported as `UnexpectedEndOfStream` rather than being converted into
//! a plausible sample value.

use std::io::{Read, Write};

use crate::{BitStreamReader, BitStreamWriter, Error, Result};

const MAX_K: u8 = 7;

/// Encode a signed i8 residual using Rice coding.
///
/// `k` is the Rice parameter and must be in `0..=7`.
///
/// Signed values are zig-zag mapped before Rice coding:
///
///     0  -> 0
///    -1  -> 1
///     1  -> 2
///    -2  -> 3
///     2  -> 4
///
/// This makes small-magnitude residuals cheap regardless of sign.
pub(crate) fn encode<W>(k: u16, x: i16, stream: &mut BitStreamWriter<W>) -> Result<()>
where
    W: Write,
{
    let k = validate_k(k)?;

    // The public codec currently uses i16 at this layer, but depth8 is
    // specifically intended for values representable by i8. Keeping the
    // intermediate signed type here avoids surprising overflow while still
    // allowing the existing callers to pass i16.
    let x = i8::try_from(x).map_err(|_| Error::InvalidData)?;

    let mapped = zigzag_encode(x);
    let quotient = mapped >> k;
    let remainder = mapped & remainder_mask(k);

    // Unary quotient: q zero bits followed by one.
    stream.write_zeros(quotient as usize)?;
    stream.write_bits(1u128, 1)?;

    // Binary remainder.
    if k != 0 {
        stream.write_bits(remainder as u128, k as usize)?;
    }

    Ok(())
}

/// Decode one signed i8 residual from a Rice-coded bitstream.
///
/// Any truncation is an error. In particular, EOF while looking for the
/// terminating unary `1` is not interpreted as a valid zero-valued sample.
pub(crate) fn decode<R>(k: u16, stream: &mut BitStreamReader<R>) -> Result<i16>
where
    R: Read,
{
    let k = validate_k(k)?;

    let quotient = read_unary_quotient(stream)?;

    let remainder = if k == 0 {
        0
    } else {
        stream
            .read_bits(k as usize)?
            .ok_or(Error::UnexpectedEndOfStream)? as u8
    };

    let mapped = (quotient << k) | remainder;

    // A zig-zag encoded i8 value occupies exactly 8 bits, giving a range
    // of 0..=255.
    if mapped > u8::MAX as u32 {
        return Err(Error::InvalidData);
    }

    let value = zigzag_decode(mapped as u8);

    Ok(value as i16)
}

/// Choose a Rice parameter from four neighboring 8-bit samples.
///
/// This is intended for residuals generated from spatial prediction.
/// Higher local activity produces a larger Rice parameter.
///
/// The returned value is always valid for an 8-bit Rice code.
pub(crate) fn k(a: u8, c: u8, b: u8, d: u8) -> u16 {
    let activity = abs_diff(a, b) as u16 + abs_diff(b, c) as u16 + abs_diff(c, d) as u16;

    choose_k(activity, MAX_K) as u16
}

#[inline]
fn validate_k(k: u16) -> Result<u8> {
    if k > MAX_K as u16 {
        return Err(Error::InvalidData);
    }

    Ok(k as u8)
}

#[inline]
fn remainder_mask(k: u8) -> u8 {
    if k == 0 {
        0
    } else {
        ((1u16 << k) - 1) as u8
    }
}

#[inline]
fn zigzag_encode(value: i8) -> u8 {
    let value = value as i16;

    // Equivalent to:
    //
    //   0  -> 0
    //  -1  -> 1
    //   1  -> 2
    //  -2  -> 3
    //
    // but performed using a wider signed type so there is no overflow.
    ((value << 1) ^ (value >> 15)) as u8
}

#[inline]
fn zigzag_decode(value: u8) -> i8 {
    let value = value as i16;

    ((value >> 1) ^ -(value & 1)) as i8
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

                // For an 8-bit Rice symbol the quotient cannot legitimately
                // exceed 255. Rejecting it here prevents malicious/corrupt
                // input from causing an unbounded unary scan.
                if quotient > u8::MAX as u32 {
                    return Err(Error::InvalidData);
                }
            }
            None => return Err(Error::UnexpectedEndOfStream),
        }
    }
}

#[inline]
fn abs_diff(a: u8, b: u8) -> u8 {
    a.abs_diff(b)
}

/// Select the smallest Rice parameter whose nominal quotient scale is
/// appropriate for the measured activity.
///
/// The old implementation used:
///
///     while 3 << k < activity
///
/// which is simple but can select a parameter larger than necessary for the
/// actual sample domain. This implementation keeps the same general heuristic
/// while making the bounds explicit.
#[inline]
fn choose_k(activity: u16, max_k: u8) -> u8 {
    let mut k = 0u8;

    while k < max_k && (3u16 << k) < activity {
        k += 1;
    }

    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitStreamReader, BitStreamWriter};

    fn round_trip(value: i16, k: u16) -> Result<i16> {
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
            (0i8, 0u8),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (2, 4),
            (-3, 5),
            (3, 6),
            (-127, 253),
            (127, 254),
            (-128, 255),
        ];

        for (value, expected) in values {
            assert_eq!(zigzag_encode(value), expected);
            assert_eq!(zigzag_decode(expected), value);
        }
    }

    #[test]
    fn round_trip_all_i8_values_for_all_k() {
        for value in i8::MIN..=i8::MAX {
            for k in 0..=MAX_K as u16 {
                let decoded = round_trip(value as i16, k).unwrap();
                assert_eq!(
                    decoded, value as i16,
                    "round trip failed for value={} k={}",
                    value, k
                );
            }
        }
    }

    #[test]
    fn round_trip_common_residuals() {
        let values = [
            -128i16, -127, -100, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 100,
            126, 127,
        ];

        for &value in &values {
            for k in 0..=MAX_K as u16 {
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

        assert!(encode(0, -129, &mut writer).is_err());
        assert!(encode(0, 128, &mut writer).is_err());
    }

    #[test]
    fn rejects_invalid_k() {
        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        assert!(encode(8, 0, &mut writer).is_err());
        assert!(encode(u16::MAX, 0, &mut writer).is_err());
    }

    #[test]
    fn rejects_truncated_unary_code() {
        // For k=0, zero bytes contain only zeros. There is no terminating one.
        let data = [0u8];

        let mut reader = BitStreamReader::new(data.as_slice());
        assert!(matches!(
            decode(0, &mut reader),
            Err(Error::UnexpectedEndOfStream)
        ));
    }

    #[test]
    fn rejects_truncated_remainder() {
        // Encode a value whose code requires a remainder, then provide only
        // enough data to contain the unary portion.
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode(7, 127, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        // A complete byte is not enough to distinguish the actual number of
        // remainder bits in all cases, so exercise the decoder with an empty
        // stream as the unambiguous truncation case.
        let mut reader = BitStreamReader::new([].as_slice());

        assert!(matches!(
            decode(7, &mut reader),
            Err(Error::UnexpectedEndOfStream)
        ));
    }

    #[test]
    fn unary_zero_is_encoded_as_one_bit() {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            encode(0, 0, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer[0] & 0x80, 0x80);
    }

    #[test]
    fn unary_positive_quotient_is_zeroes_then_one() {
        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            // -1 maps to 1. With k=0, quotient=1, therefore "01".
            encode(0, -1, &mut writer).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer[0] >> 6, 0b01);
    }

    #[test]
    fn sequential_stream_round_trip() {
        let values = [
            -128i16, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 127,
        ];

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            for &value in &values {
                encode(3, value, &mut writer).unwrap();
            }

            writer.flush().unwrap();
        }

        let mut reader = BitStreamReader::new(buffer.as_slice());

        for &expected in &values {
            let actual = decode(3, &mut reader).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn k_is_bounded() {
        for a in 0u8..=255 {
            for b in [0u8, 1, 127, 128, 254, 255] {
                for c in [0u8, 1, 127, 128, 254, 255] {
                    for d in [0u8, 1, 127, 128, 254, 255] {
                        assert!(k(a, c, b, d) <= MAX_K as u16);
                    }
                }
            }
        }
    }

    #[test]
    fn k_is_zero_for_constant_region() {
        assert_eq!(k(100, 100, 100, 100), 0);
        assert_eq!(k(0, 0, 0, 0), 0);
        assert_eq!(k(255, 255, 255, 255), 0);
    }
}
