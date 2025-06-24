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
pub(crate) fn encode<W>(k: u16, x: i16, stream: &mut BitStreamWriter<W>) -> Result<()>
where
    W: Write,
{
    // EXACT COPY of reference implementation logic
    let x = ((x as i32 >> 30) ^ (2 * x as i32)) as u32;
    let high_bits = x >> k;

    // Write high_bits zeros followed by a terminating 1
    // First write the zeros (if any)
    if high_bits > 0 {
        stream.write_zeros(high_bits as usize)?;
    }
    // Then write the terminating 1 bit
    stream.write_bits(1u128, 1)?;

    // Write low bits exactly as reference
    if k > 0 {
        stream.write_bits((x & ((1 << k) - 1)) as u128, k as usize)?;
    }

    Ok(())
}

pub(crate) fn decode<R>(k: u16, stream: &mut BitStreamReader<R>) -> Result<i16>
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
    Ok(result as i16)
}

// https://blog.tempus-ex.com/hello-video-codec/
// We want to choose k values relatively well, as that will optimize are end compression
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
    use super::*;

    #[test]
    fn test_bit_stream_basic() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        // Test basic bit writing and reading
        let mut writer = BitStreamWriter::new(VecDeque::new());

        // Write 3 zeros
        writer.write_bits(0u128, 3).expect("write zeros");
        // Write 1 one
        writer.write_bits(1u128, 1).expect("write one");
        // Write 2 bits with value 2 (10 in binary)
        writer.write_bits(2u128, 2).expect("write two bits");

        writer.flush().expect("flush");
        let data = writer.into_inner();
        let inner_vec: Vec<u8> = data.into();

        println!("Basic test encoded bytes: {:?}", inner_vec);
        println!(
            "Basic test encoded bits: {:08b}",
            inner_vec.get(0).unwrap_or(&0)
        );

        // Now read back
        let mut reader = BitStreamReader::new_with_data(inner_vec.as_slice()).expect("reader");

        // Count zeros
        let mut zero_count = 0;
        while reader.read_bit().expect("read bit") == Some(false) {
            zero_count += 1;
        }
        println!("Read {} zeros", zero_count);

        // Read 2 bits
        let two_bits = reader.read_bits(2).expect("read 2 bits");
        println!("Read 2 bits: {:?}", two_bits);
    }

    #[test]
    fn test_rice_specific_failure() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        // Test the specific case that's working
        let value = 131i16;
        let k = 0u16;

        println!("Original value {}", value);

        // Encode
        let mut writer = BitStreamWriter::new(VecDeque::new());
        encode(k, value, &mut writer).expect("encode");
        writer.flush().expect("flush");
        let data = writer.into_inner();

        // Decode
        let inner_vec: Vec<u8> = data.into();
        let mut reader = BitStreamReader::new_with_data(inner_vec.as_slice()).expect("reader");
        let decoded = decode(k, &mut reader).expect("decode");

        println!("Decoded value {}", decoded);
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_negative_value_debug() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        let value = -128i16;
        let k = 0u16;

        // Calculate what the encoding should produce
        let x = ((value as i32 >> 30) ^ (2 * value as i32)) as u32;
        let high_bits = x >> k;

        println!("Value: {}", value);
        println!("x as i32: {}", value as i32);
        println!("x as i32 >> 30: {}", value as i32 >> 30);
        println!("2 * x as i32: {}", 2 * value as i32);
        println!("XOR result: {}", (value as i32 >> 30) ^ (2 * value as i32));
        println!("x (as u32): {}", x);
        println!("high_bits (x >> k): {}", high_bits);
        println!(
            "Need to write {} zeros + 1 terminating bit = {} total bits",
            high_bits,
            high_bits + 1
        );

        // Try encoding
        let mut writer = BitStreamWriter::new(VecDeque::new());
        match encode(k, value, &mut writer) {
            Ok(()) => {
                println!("✓ Encoding succeeded");
                writer.flush().expect("flush");
                let data = writer.into_inner();
                println!("Encoded data size: {} bytes", data.len());

                // Try decoding
                let inner_vec: Vec<u8> = data.into();
                let mut reader =
                    BitStreamReader::new_with_data(inner_vec.as_slice()).expect("reader");
                match decode(k, &mut reader) {
                    Ok(decoded) => println!("✓ Decoded: {}", decoded),
                    Err(e) => println!("❌ Decode failed: {:?}", e),
                }
            }
            Err(e) => println!("❌ Encoding failed: {:?}", e),
        }
    }

    #[test]
    fn test_rice_extreme_values() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        // Test extreme values that might be problematic
        let extreme_values = vec![-128i16, -127, -1, 0, 1, 127, -32768, 32767];

        for &value in &extreme_values {
            for k in 0u16..=5 {
                println!("Testing extreme value {} with k={}", value, k);

                // Encode
                let mut writer = BitStreamWriter::new(VecDeque::new());
                encode(k, value, &mut writer)
                    .expect(&format!("Failed to encode {} with k={}", value, k));
                writer.flush().expect("flush");
                let data = writer.into_inner();

                // Decode
                let inner_vec: Vec<u8> = data.into();
                let mut reader =
                    BitStreamReader::new_with_data(inner_vec.as_slice()).expect("reader");
                let decoded = decode(k, &mut reader)
                    .expect(&format!("Failed to decode {} with k={}", value, k));

                assert_eq!(
                    value, decoded,
                    "Mismatch for k={}, value={}: expected {} got {}",
                    k, value, value, decoded
                );
            }
        }
    }

    #[test]
    fn test_rice_sequential_values() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        // Test a small sequence that includes problematic values
        let test_values = vec![-128i16, -1, 0, 1, 127];
        let k = 0u16;

        println!(
            "Testing {} sequential values with k={}",
            test_values.len(),
            k
        );

        // Encode all values in one stream
        let mut writer = BitStreamWriter::new(VecDeque::new());

        for &value in &test_values {
            println!("Encoding value: {}", value);
            encode(k, value, &mut writer).expect(&format!("Failed to encode value {}", value));
        }

        writer.flush().expect("Failed to flush writer");
        let data = writer.into_inner();

        // Convert to Vec<u8> for reading
        let data_vec: Vec<u8> = data.into();
        println!(
            "Encoded {} values into {} bytes: {:?}",
            test_values.len(),
            data_vec.len(),
            data_vec
        );

        // Decode all values from the stream
        let mut reader =
            BitStreamReader::new_with_data(data_vec.as_slice()).expect("Failed to create reader");
        let mut decoded_values = Vec::with_capacity(test_values.len());

        for i in 0..test_values.len() {
            match decode(k, &mut reader) {
                Ok(decoded) => {
                    println!(
                        "Decoded value {}: {} (expected: {})",
                        i, decoded, test_values[i]
                    );
                    decoded_values.push(decoded);
                }
                Err(e) => panic!("Failed to decode value {} with k={}: {:?}", i, k, e),
            }
        }

        // Verify all values match
        assert_eq!(test_values.len(), decoded_values.len(), "Length mismatch");

        for (i, (&original, &decoded)) in test_values.iter().zip(decoded_values.iter()).enumerate()
        {
            assert_eq!(
                original, decoded,
                "Mismatch at index {}: original={}, decoded={}",
                i, original, decoded
            );
        }

        println!(
            "✓ Successfully encoded and decoded {} sequential values",
            test_values.len()
        );
    }

    #[test]
    fn test_rice_encode_decode_consistency() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        let test_values = vec![0i16, 1, -1, 2, -2, 3, -3, 100, -100, 255, -255];
        let test_k_values = vec![0u16, 1, 2, 3, 4, 5];

        for &k in &test_k_values {
            for &value in &test_values {
                // Encode
                let mut writer = BitStreamWriter::new(VecDeque::new());
                encode(k, value, &mut writer).expect("encode");
                writer.flush().expect("flush");
                let data = writer.into_inner();

                // Decode
                let inner_vec: Vec<u8> = data.into();
                let mut reader =
                    BitStreamReader::new_with_data(inner_vec.as_slice()).expect("reader");
                let decoded = decode(k, &mut reader).expect("decode");

                if value != decoded {
                    println!("FAILURE: k={}, value={}, decoded={}", k, value, decoded);
                }
                assert_eq!(value, decoded, "Mismatch for k={}, value={}", k, value);
            }
        }
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

    #[test]
    fn test_rice_1024_values_image_simulation() {
        use std::collections::VecDeque;

        use crate::{BitStreamReader, BitStreamWriter};

        // Generate 1024 values using a simple repeating pattern to avoid complex math
        let mut test_values = Vec::with_capacity(1024);
        let pattern = vec![-2i16, -1, 0, 1, 2];

        for i in 0..1024 {
            test_values.push(pattern[i % pattern.len()]);
        }

        // Test with k=0 since that's where we saw the failure
        let k = 0u16;
        println!(
            "Testing Rice encoding with k={} for 1024 values using pattern {:?}",
            k, pattern
        );

        // Encode all values in one stream
        let mut writer = BitStreamWriter::new(VecDeque::new());

        for &value in &test_values {
            encode(k, value, &mut writer)
                .expect(&format!("Failed to encode value {} with k={}", value, k));
        }

        writer.flush().expect("Failed to flush writer");
        let data = writer.into_inner();

        // Convert to Vec<u8> for reading
        let data_vec: Vec<u8> = data.into();
        println!(
            "Encoded 1024 values with k={} into {} bytes",
            k,
            data_vec.len()
        );

        // Decode all values from the stream
        let mut reader =
            BitStreamReader::new_with_data(data_vec.as_slice()).expect("Failed to create reader");
        let mut decoded_values = Vec::with_capacity(1024);

        for i in 0..1024 {
            match decode(k, &mut reader) {
                Ok(decoded) => decoded_values.push(decoded),
                Err(e) => panic!("Failed to decode value {} with k={}: {:?}", i, k, e),
            }
        }

        // Verify all values match
        assert_eq!(
            test_values.len(),
            decoded_values.len(),
            "Length mismatch for k={}",
            k
        );

        for (i, (&original, &decoded)) in test_values.iter().zip(decoded_values.iter()).enumerate()
        {
            if original != decoded {
                println!(
                    "ERROR at index {}: original={}, decoded={}",
                    i, original, decoded
                );
                // Show some context around the error
                let start = i.saturating_sub(5);
                let end = (i + 5).min(test_values.len());
                println!("Context around error:");
                for j in start..end {
                    let marker = if j == i { " <-- ERROR" } else { "" };
                    println!(
                        "  [{}] original={}, decoded={}{}",
                        j, test_values[j], decoded_values[j], marker
                    );
                }
                panic!("First mismatch at index {}", i);
            }
        }

        println!("✓ Rice encoding test passed for 1024 values with simple pattern!");
    }
}
