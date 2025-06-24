use std::{
    cmp::{max, min},
    io::{Read, Write},
};

use super::Jpg;
use crate::{
    color::Subsampling,
    dimensions::PixelDimensions,
    error::Error,
    image::{ImageRgb8, ImageRgba8},
    pixels::{Rgb8, Rgb8Ref, Rgba8, Rgba8Ref},
    rice::depth8::{decode, encode, k},
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Result,
};

struct SampleGroup {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

impl Encodable for Jpg<'_, Rgb8Ref<'_>> {
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

impl Encodable for Jpg<'_, Rgba8Ref<'_>> {
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
    pixels: &[u8],
    depth: u8,
    stride: usize,
    stream: &mut BitStreamWriter<W>,
) -> Result<()>
where
    W: Write,
{
    dimensions.encode(stream)?;
    stream.write(depth)?;
    stream.write(stride)?;

    let depth = depth as usize;
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
            let residual = x as i16 - prediction;

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

impl Decodable for Jpg<'_, Rgb8> {
    type Output = ImageRgb8;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let dimensions = PixelDimensions::decode(stream)?;
        let depth = stream
            .read()?
            .ok_or(Error::FailedToDecode("depth".to_owned()))?;
        let stride = stream
            .read()?
            .ok_or(Error::FailedToDecode("stride".to_owned()))?;
        let pixels = decompress(dimensions, depth, stride, stream)?;
        Ok(ImageRgb8::new(
            dimensions,
            Rgb8::new(pixels),
            Subsampling::Sample444,
        ))
    }
}

impl Decodable for Jpg<'_, Rgba8> {
    type Output = ImageRgba8;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let dimensions = PixelDimensions::decode(stream)?;
        let depth = stream
            .read()?
            .ok_or(Error::FailedToDecode("depth".to_owned()))?;
        let stride = stream
            .read()?
            .ok_or(Error::FailedToDecode("stride".to_owned()))?;
        let pixels = decompress(dimensions, depth, stride, stream)?;
        Ok(ImageRgba8::new(
            dimensions,
            Rgba8::new(pixels),
            Subsampling::Sample444,
        ))
    }
}

fn decompress<R>(
    dimensions: PixelDimensions,
    depth: u8,
    stride: usize,
    stream: &mut BitStreamReader<R>,
) -> Result<Vec<u8>>
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

    let mut pixels = vec![0u8; height * stride];
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
            let x = (prediction as i32 + residual as i32).clamp(0, 255) as u8;
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

pub(crate) fn sample_prediction(a: u8, c: u8, b: u8) -> i16 {
    if c >= max(a, b) {
        min(a, b) as i16
    } else if c <= min(a, b) {
        max(a, b) as i16
    } else {
        a as i16 + b as i16 - c as i16
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use image::GenericImageView as _;

    use super::*;
    use crate::{color::Subsampling, Decodable, Encodable};

    #[test]
    #[ntest::timeout(300000)] // 5 minutes timeout
    fn encode_decode_jpeg_1234() {
        let image = image::open("./test_imgs/input/rgb8/hummingbird.jpg").expect("open img");
        let dimensions = image.dimensions().into();
        match image.color() {
            image::ColorType::Rgb8 => {
                let image = image.as_rgb8().unwrap().clone();

                let size = image.len();
                let data = image.to_vec();
                let img = ImageRgb8::new(dimensions, Rgb8::new(data), Subsampling::Sample444);

                let codec = Jpg {
                    image: img.as_ref(),
                };
                let mut writer = BitStreamWriter::new(VecDeque::with_capacity(size));
                codec.encode(&mut writer).expect("compress");

                let inner = writer.into_inner();

                assert!(
                    inner.len() < size,
                    "compressed size should be less than original size"
                );

                let inner_vec: Vec<u8> = inner.into();
                let mut reader =
                    BitStreamReader::new_with_data(inner_vec.as_slice()).expect("create reader");

                let decoded = Jpg::<'_, Rgb8>::decode(&mut reader).expect("decompress");

                let original = img.pixels().as_slice();
                let decoded_data = decoded.pixels().as_slice();

                // Debug information
                println!("Original size: {}", original.len());
                println!("Decoded size: {}", decoded_data.len());
                println!(
                    "Image dimensions: {}x{}",
                    dimensions.width, dimensions.height
                );
                println!("Image depth: {}", img.depth());
                println!("Line stride: {}", img.line_stride());

                if original.len() != decoded_data.len() {
                    panic!(
                        "Size mismatch: original {} vs decoded {}",
                        original.len(),
                        decoded_data.len()
                    );
                }

                let mut differences = 0;
                let mut first_diff_row = None;
                let mut first_diff_col = None;
                for (i, (orig, dec)) in original.iter().zip(decoded_data.iter()).enumerate() {
                    if orig != dec {
                        if differences < 10 {
                            let row = i / img.line_stride();
                            let col = i % img.line_stride();
                            let channel = col % img.depth() as usize;
                            println!(
                                "Difference at index {} (row {}, col {}, channel {}): original {} vs decoded {}",
                                i, row, col, channel, orig, dec
                            );

                            if first_diff_row.is_none() {
                                first_diff_row = Some(row);
                                first_diff_col = Some(col);
                            }
                        }
                        differences += 1;
                    }
                }

                if differences > 0 {
                    println!("Total differences: {}", differences);
                    if let (Some(row), Some(col)) = (first_diff_row, first_diff_col) {
                        println!("First difference at row {}, col {}", row, col);

                        // Show surrounding context
                        let start = row.saturating_sub(2);
                        let end = (row + 3).min(dimensions.height);
                        for r in start..end {
                            let line_start = r * img.line_stride();
                            let line_end = line_start + img.line_stride();
                            println!(
                                "Row {}: Original {:?}",
                                r,
                                &original[line_start..line_end.min(original.len())]
                            );
                            println!(
                                "Row {}: Decoded  {:?}",
                                r,
                                &decoded_data[line_start..line_end.min(decoded_data.len())]
                            );
                        }
                    }
                    panic!(
                        "Found {} differences between original and decoded data",
                        differences
                    );
                }

                assert_eq!(decoded.as_ref(), img.as_ref());
            }
            _ => panic!("unsupported color type"),
        }
    }
}
