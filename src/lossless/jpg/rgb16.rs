use std::{
    cmp::{max, min},
    io::{Read, Write},
};

use super::Jpg;
use crate::{
    color::Subsampling,
    dimensions::PixelDimensions,
    error::Error,
    image::{ImageRgb16, ImageRgba16},
    pixels::{Rgb16, Rgb16Ref, Rgba16, Rgba16Ref},
    rice::depth16::{decode, encode, k},
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Result,
};

struct SampleGroup {
    a: u16,
    b: u16,
    c: u16,
    d: u16,
}

impl Encodable for Jpg<'_, Rgb16Ref<'_>> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let dimensions = self.image.dimensions();
        let stride = self.image.line_stride();
        let pixels = self.image.pixels().as_slice();
        compress(dimensions, pixels, self.image.depth(), stride, stream)?;
        Ok(())
    }
}

impl Encodable for Jpg<'_, Rgba16Ref<'_>> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let dimensions = self.image.dimensions();
        let stride = self.image.line_stride();
        let pixels = self.image.pixels().as_slice();
        compress(dimensions, pixels, self.image.depth(), stride, stream)?;
        Ok(())
    }
}

fn compress<W>(
    dimensions: PixelDimensions,
    pixels: &[u16],
    depth: u8,
    stride: usize,
    stream: &mut BitStreamWriter<W>,
) -> Result<()>
where
    W: Write,
{
    dimensions.encode(stream)?;
    stream.write(depth)?;
    stream.write(stride as u32)?;

    let PixelDimensions { width, height } = dimensions;
    let mut sample_groups = vec![];
    for _ in 0..depth {
        sample_groups.push(SampleGroup {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
        });
    }

    let depth = depth as usize;
    for row in 0..height {
        for sample_group in sample_groups.iter_mut() {
            sample_group.a = 0;
            sample_group.c = 0;
        }

        for col in 0..(width * depth) {
            let sample_group = &mut sample_groups[col % depth];
            let x = pixels[(row * stride) + col];
            sample_group.d = if row > 0 && (col + depth) < (width * depth) {
                let prev_row = (row - 1) * stride;
                let next_col = col + depth;
                pixels[prev_row + next_col]
            } else {
                0
            };

            let prediction = sample_prediction(sample_group.a, sample_group.c, sample_group.b);
            let residual = x as i32 - prediction;

            encode(
                k(
                    sample_group.a,
                    sample_group.c,
                    sample_group.b,
                    sample_group.d,
                ),
                residual,
                stream,
            )?;

            sample_group.c = sample_group.b;
            sample_group.b = sample_group.d;
            sample_group.a = x;
        }
        for (i, sample_group) in sample_groups.iter_mut().enumerate() {
            sample_group.b = pixels[(row * stride) + i];
        }
    }

    stream.flush()
}

impl Decodable for Jpg<'_, Rgb16> {
    type Output = ImageRgb16;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let dimensions = PixelDimensions::decode(stream)?;
        let depth = stream
            .read()?
            .ok_or(Error::FailedToDecode("depth".to_owned()))?;
        let stride = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("stride".to_owned()))? as usize;
        let pixels = decompress(dimensions, depth, stride, stream)?;
        Ok(ImageRgb16::new(
            dimensions,
            Rgb16::new(pixels),
            Subsampling::Sample444,
        ))
    }
}

impl Decodable for Jpg<'_, Rgba16> {
    type Output = ImageRgba16;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let dimensions = PixelDimensions::decode(stream)?;
        let depth = stream
            .read()?
            .ok_or(Error::FailedToDecode("depth".to_owned()))?;
        let stride = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("stride".to_owned()))? as usize;
        let pixels = decompress(dimensions, depth, stride, stream)?;
        Ok(ImageRgba16::new(
            dimensions,
            Rgba16::new(pixels),
            Subsampling::Sample444,
        ))
    }
}

fn decompress<R>(
    dimensions: PixelDimensions,
    depth: u8,
    stride: usize,
    stream: &mut BitStreamReader<R>,
) -> Result<Vec<u16>>
where
    R: Read,
{
    let PixelDimensions { width, height } = dimensions;
    let mut sample_groups = vec![];
    for _ in 0..depth {
        sample_groups.push(SampleGroup {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
        });
    }

    let depth = depth as usize;

    // Validate input dimensions
    if depth == 0 {
        return Err(Error::InvalidDepth(0));
    }
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }
    if stride == 0 {
        return Err(Error::InvalidStride(stride));
    }

    let mut pixels = vec![0u16; height * stride];
    for row in 0..height {
        for sample_group in sample_groups.iter_mut() {
            sample_group.a = 0;
            sample_group.c = 0;
        }

        for col in 0..(width * depth) {
            let sample_group = &mut sample_groups[col % depth];
            sample_group.d = if row > 0 && (col + depth) < (width * depth) {
                let prev_row = (row - 1) * stride;
                let next_col = col + depth;
                pixels[prev_row + next_col]
            } else {
                0
            };

            let prediction = sample_prediction(sample_group.a, sample_group.c, sample_group.b);
            let residual = decode(
                k(
                    sample_group.a,
                    sample_group.c,
                    sample_group.b,
                    sample_group.d,
                ),
                stream,
            )?;
            let x = (prediction + residual) as u16;
            pixels[(row * stride) + (col)] = x;

            sample_group.c = sample_group.b;
            sample_group.b = sample_group.d;
            sample_group.a = x;
        }
        for (i, sample_group) in sample_groups.iter_mut().enumerate() {
            sample_group.b = pixels[row * stride + i];
        }
    }

    Ok(pixels)
}

pub(crate) fn sample_prediction(a: u16, c: u16, b: u16) -> i32 {
    if c >= max(a, b) {
        min(a, b) as i32
    } else if c <= min(a, b) {
        max(a, b) as i32
    } else {
        a as i32 + b as i32 - c as i32
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use image::GenericImageView as _;

    use super::*;
    use crate::color::Subsampling;

    #[test]
    fn encode_decode_jpeg_real_image() {
        // Test with a real image (may not compress well depending on content)
        let image = image::open("./test_imgs/input/rgb16/tos.tif").expect("open img");
        let dimensions = image.dimensions().into();

        match image.color() {
            image::ColorType::Rgb16 => {
                let image = image.as_rgb16().unwrap().clone();
                let size = image.len();
                let data = image.to_vec();
                let img = ImageRgb16::new(dimensions, Rgb16::new(data), Subsampling::Sample444);

                let codec = Jpg {
                    image: img.as_ref(),
                };
                let mut writer = BitStreamWriter::new(VecDeque::with_capacity(size));
                codec.encode(&mut writer).expect("compress");

                let inner = writer.into_inner();

                // Real images may not compress well, so just verify correctness
                let mut reader = BitStreamReader::new(inner);
                let decoded = Jpg::<'_, Rgb16>::decode(&mut reader).expect("decompress");

                assert_eq!(img.pixels().as_slice(), decoded.pixels().as_slice());
                assert_eq!(decoded.as_ref(), img.as_ref());
            }
            _ => panic!("unsupported color type"),
        }
    }
}
