// REF: https://www.sciencedirect.com/topics/engineering/quantization-table
// Separate quantization tables for images (JPEG-style) and video (H.264-style)
//
use std::{
    cmp::PartialOrd,
    fmt::Debug,
    io::{Read, Write},
    ops::{Div, Mul},
};

use num_traits::{Bounded, NumCast};

use super::Block;
use crate::{BitStreamReader, BitStreamWriter, Decodable, Encodable, Result};

#[rustfmt::skip]
const IMAGE_LUMINANCE_QUANTIZATION_I16: Block<i16> = Block([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]);

#[rustfmt::skip]
const IMAGE_CHROMINANCE_QUANTIZATION_I16: Block<i16> = Block([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]);

#[rustfmt::skip]
const VIDEO_LUMINANCE_QUANTIZATION_I16: Block<i16> = Block([
    6,  5,  4,  6,  10, 16, 20, 24,
    5,  5,  6,  8,  10, 22, 24, 22,
    6,  6,  7,  10, 16, 22, 28, 22,
    6,  7,  9,  12, 20, 34, 32, 24,
    7,  9,  15, 22, 26, 42, 40, 30,
    10, 14, 22, 26, 32, 40, 44, 36,
    20, 26, 30, 34, 40, 48, 48, 40,
    28, 36, 38, 38, 44, 40, 40, 38,
]);

#[rustfmt::skip]
const VIDEO_CHROMINANCE_QUANTIZATION_I16: Block<i16> = Block([
    6,  6,  8,  16, 32, 32, 32, 32,
    6,  7,  9,  22, 32, 32, 32, 32,
    8,  9,  18, 32, 32, 32, 32, 32,
    16, 22, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
]);

const REASONABLE_CLAMP_MIN_I16: i16 = i16::MIN;
const REASONABLE_CLAMP_MAX_I16: i16 = i16::MAX;

#[rustfmt::skip]
const IMAGE_LUMINANCE_QUANTIZATION_I32: Block<i32> = Block([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]);

#[rustfmt::skip]
const IMAGE_CHROMINANCE_QUANTIZATION_I32: Block<i32> = Block([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]);

#[rustfmt::skip]
const VIDEO_LUMINANCE_QUANTIZATION_I32: Block<i32> = Block([
    6,  5,  4,  6,  10, 16, 20, 24,
    5,  5,  6,  8,  10, 22, 24, 22,
    6,  6,  7,  10, 16, 22, 28, 22,
    6,  7,  9,  12, 20, 34, 32, 24,
    7,  9,  15, 22, 26, 42, 40, 30,
    10, 14, 22, 26, 32, 40, 44, 36,
    20, 26, 30, 34, 40, 48, 48, 40,
    28, 36, 38, 38, 44, 40, 40, 38,
]);

#[rustfmt::skip]
const VIDEO_CHROMINANCE_QUANTIZATION_I32: Block<i32> = Block([
    6,  6,  8,  16, 32, 32, 32, 32,
    6,  7,  9,  22, 32, 32, 32, 32,
    8,  9,  18, 32, 32, 32, 32, 32,
    16, 22, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
]);

const REASONABLE_CLAMP_MIN_I32: i32 = i32::MIN;
const REASONABLE_CLAMP_MAX_I32: i32 = i32::MAX;

#[derive(Copy, Clone, Debug)]
pub struct Quantizor<T>(Block<T>);

impl Quantizor<i16> {
    pub(crate) const fn image_luminance() -> Self {
        Self(IMAGE_LUMINANCE_QUANTIZATION_I16)
    }

    pub(crate) const fn image_chrominance() -> Self {
        Self(IMAGE_CHROMINANCE_QUANTIZATION_I16)
    }

    pub(crate) const fn video_luminance() -> Self {
        Self(VIDEO_LUMINANCE_QUANTIZATION_I16)
    }

    pub(crate) const fn video_chrominance() -> Self {
        Self(VIDEO_CHROMINANCE_QUANTIZATION_I16)
    }

    /// Quantize with clamping to i16 range to ensure values fit for array-based ANS encoding
    pub fn quantize(&self, block: Block<i16>) -> Block<i16> {
        (block / self.0).clamp(REASONABLE_CLAMP_MIN_I16, REASONABLE_CLAMP_MAX_I16)
    }

    pub(crate) fn dequantize(&self, block: Block<i16>) -> Block<i16> {
        block * self.0
    }
}

impl Quantizor<i32> {
    pub(crate) fn image_luminance() -> &'static Self {
        static Q: std::sync::OnceLock<Quantizor<i32>> = std::sync::OnceLock::new();
        Q.get_or_init(|| Self(IMAGE_LUMINANCE_QUANTIZATION_I32))
    }

    pub(crate) fn image_chrominance() -> &'static Self {
        static Q: std::sync::OnceLock<Quantizor<i32>> = std::sync::OnceLock::new();
        Q.get_or_init(|| Self(IMAGE_CHROMINANCE_QUANTIZATION_I32))
    }

    pub(crate) fn video_luminance() -> &'static Self {
        static Q: std::sync::OnceLock<Quantizor<i32>> = std::sync::OnceLock::new();
        Q.get_or_init(|| Self(VIDEO_LUMINANCE_QUANTIZATION_I32))
    }

    pub(crate) fn video_chrominance() -> &'static Self {
        static Q: std::sync::OnceLock<Quantizor<i32>> = std::sync::OnceLock::new();
        Q.get_or_init(|| Self(VIDEO_CHROMINANCE_QUANTIZATION_I32))
    }

    /// Quantize with clamping to i16 range to ensure values fit for array-based ANS encoding
    pub fn quantize(&self, block: Block<i32>) -> Block<i32> {
        (block / self.0).clamp(REASONABLE_CLAMP_MIN_I32, REASONABLE_CLAMP_MAX_I32)
    }

    pub(crate) fn dequantize(&self, block: Block<i32>) -> Block<i32> {
        block * self.0
    }
}

impl<const N: usize, T> Encodable for Quantizor<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.0.encode(stream)
    }
}

impl<const N: usize, T> Decodable for Quantizor<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self>
    where
        R: Read,
    {
        Ok(Self(Block::decode(stream)?))
    }
}
