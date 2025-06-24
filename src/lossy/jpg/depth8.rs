use std::{
    io::{Read, Write},
    mem::size_of,
};

use super::{decompress_from_stream, Jpg};
use crate::{
    image::Image,
    pixels::{Rgb8, Rgb8Ref, Rgba8, Rgba8Ref},
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Result,
};

const SIZE: usize = size_of::<i16>();

impl Encodable for Jpg<'_, Rgb8Ref<'_>> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let dimensions = self.image.dimensions();
        let ycbcr = self.image.pixels().to_ycbcr();

        self.compress_to_stream::<SIZE, i16, _>(dimensions, &ycbcr, stream)
    }
}

impl Encodable for Jpg<'_, Rgba8Ref<'_>> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let dimensions = self.image.dimensions();
        let ycbcr = self.image.pixels().to_ycbcr();

        self.compress_to_stream::<SIZE, i16, _>(dimensions, &ycbcr, stream)
    }
}

impl Decodable for Jpg<'_, Rgb8> {
    type Output = Image<Rgb8>;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let (dimensions, subsampling, pixels) = decompress_from_stream::<SIZE, i16, _, u8>(stream)?;

        Ok(Image::new(dimensions, Rgb8::new(pixels), subsampling))
    }
}

impl Decodable for Jpg<'_, Rgba8> {
    type Output = Image<Rgba8>;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let (dimensions, subsampling, pixels) = decompress_from_stream::<SIZE, i16, _, u8>(stream)?;

        Ok(Image::new(dimensions, Rgba8::new(pixels), subsampling))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, time::Instant};

    use image::GenericImageView as _;

    use super::*;
    use crate::{color::Subsampling, image::ImageRgb8, lossy::jpg::decompress_from_stream};

    #[test]
    #[ntest::timeout(300000)] // 5 minutes timeout - should be plenty now
    fn compress_then_decompress_simple() {
        // Create test output directory
        let output_dir =
            std::path::Path::new("./test_imgs/output").join("compress_then_decompress_simple_rgb8");
        std::fs::create_dir_all(&output_dir).expect("Failed to create test output directory");

        let image = image::open("./test_imgs/input/rgb8/hummingbird.jpg").expect("open img");
        let dimensions = image.dimensions().into();
        match image.color() {
            image::ColorType::Rgb8 => {
                for subsampling in [
                    Subsampling::Sample411,
                    Subsampling::Sample420,
                    Subsampling::Sample422,
                    Subsampling::Sample444,
                ] {
                    let original_image = image.as_rgb8().unwrap().clone();

                    println!("\nTesting subsampling mode: {subsampling:?}");

                    let size = original_image.len();
                    let data = original_image.to_vec();
                    println!("Original size: {} bytes", data.len());
                    let img = ImageRgb8::new(dimensions, Rgb8::new(data), Subsampling::Sample444);

                    let codec = Jpg {
                        subsampling,
                        image: img.as_ref(),
                    };
                    let mut writer = BitStreamWriter::new(VecDeque::with_capacity(size));
                    let now = Instant::now();
                    println!("Starting compression...");
                    codec.encode(&mut writer).expect("compress");
                    let elapsed = now.elapsed();
                    println!("Compression time: {elapsed:?}");

                    let inner = writer.into_inner();
                    println!("Compressed size: {} bytes", inner.len());

                    let mut reader = BitStreamReader::new(inner);

                    // Decompress directly to Vec<u8> without wrapping in Image struct
                    let now = Instant::now();
                    let (dimensions, subsampling, pixels) =
                        decompress_from_stream::<SIZE, i16, _, u8>(&mut reader)
                            .expect("decompress");
                    let elapsed = now.elapsed();
                    println!("Decompression time: {elapsed:?}");
                    println!("Decoded pixels length: {}", pixels.len());
                    println!(
                        "Expected length: {}",
                        dimensions.width * dimensions.height * 3
                    );

                    // Save directly from the decoded Vec<u8>
                    let output_image: image::RgbImage = image::ImageBuffer::from_raw(
                        dimensions.width as u32,
                        dimensions.height as u32,
                        pixels,
                    )
                    .expect("Failed to create output image");

                    let output_path = output_dir.join(format!("hummingbird_{subsampling:?}.jpg"));
                    output_image.save(&output_path).expect("save");
                    println!("Saved output to: {}", output_path.display());
                }
            }
            _ => panic!("unsupported color type"),
        }
    }
}
