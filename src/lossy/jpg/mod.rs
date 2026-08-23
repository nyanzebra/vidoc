use std::{
    cmp::PartialOrd,
    hash::Hash,
    io::{Read, Write},
    ops::{Div, Mul},
};

use num_traits::{Bounded, FromBytes, FromPrimitive, NumCast, PrimInt, ToBytes, ToPrimitive};
use rayon::{iter::ParallelIterator as _, slice::ParallelSlice as _};

use crate::{
    block::{quantization::Quantizor, Block},
    color::{Subsampling, Ycbcr},
    dimensions::PixelDimensions,
    encoders::ans,
    image::ImageRef,
    lossy::{reconstruct_pixels, subsample_into_block_ycbcr, SubSampleBlockGroup},
    BitStreamReader, BitStreamWriter, Decodable, Encodable as _, Result,
};

pub(crate) mod depth16;
pub(crate) mod depth8;

pub struct Jpg<'a, T> {
    image: ImageRef<'a, T>,
    subsampling: Subsampling,
}

impl<'a, T> Jpg<'a, T> {
    pub(crate) fn compress_to_stream<const N: usize, Q, W>(
        &self,
        dimensions: PixelDimensions,
        ycbcr: &[Ycbcr],
        stream: &mut BitStreamWriter<W>,
    ) -> Result<()>
    where
        W: Write,
        Q: Copy
            + Default
            + Div<Output = Q>
            + Mul<Output = Q>
            + Hash
            + NumCast
            + PrimInt
            + ToBytes<Bytes = [u8; N]>
            + PartialOrd
            + 'static,
    {
        dimensions.encode(stream)?;
        self.subsampling.encode(stream)?;

        let SubSampleBlockGroup {
            dimensions: _,
            subsampling: _,
            y,
            cb,
            cr,
        } = subsample_into_block_ycbcr(dimensions, ycbcr, self.subsampling);

        let lumi_quantizor = Quantizor::<Q>::image_luminance();
        let y_dct = y
            .iter()
            .flat_map(|y| {
                lumi_quantizor
                    .quantize(y.dct().clamp(i16::MIN as f64, i16::MAX as f64).convert_to())
                    .zigzag()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let chroma_quantizor = Quantizor::<Q>::image_chrominance();
        let cb_dct = cb
            .iter()
            .flat_map(|cb| {
                chroma_quantizor
                    .quantize(
                        cb.dct()
                            .clamp(i16::MIN as f64, i16::MAX as f64)
                            .convert_to(),
                    )
                    .zigzag()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let cr_dct = cr
            .iter()
            .flat_map(|cr| {
                chroma_quantizor
                    .quantize(
                        cr.dct()
                            .clamp(i16::MIN as f64, i16::MAX as f64)
                            .convert_to(),
                    )
                    .zigzag()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        ans::encode_raw(&y_dct, stream)?;
        ans::encode_raw(&cb_dct, stream)?;
        ans::encode_raw(&cr_dct, stream)?;

        stream.flush()
    }
}

pub(crate) fn decompress_from_stream<const N: usize, Q, R, T>(
    stream: &mut BitStreamReader<R>,
) -> Result<(PixelDimensions, Subsampling, Vec<T>)>
where
    Q: Copy
        + Default
        + Div<Output = Q>
        + Mul<Output = Q>
        + Hash
        + NumCast
        + PrimInt
        + ToBytes<Bytes = [u8; N]>
        + FromBytes<Bytes = [u8; N]>
        + PartialOrd
        + Send
        + Sync
        + 'static,
    R: Read,
    T: Bounded + FromPrimitive + ToPrimitive + Send + Sync,
{
    let dimensions = PixelDimensions::decode(stream)?;
    let subsampling = Subsampling::decode(stream)?;

    let lumi_quantizor = Quantizor::<Q>::image_luminance();
    let chroma_quantizor = Quantizor::<Q>::image_chrominance();

    let y = ans::decode_raw(stream)?
        .par_chunks_exact(Block::<Q>::size())
        .map(|chunk| {
            lumi_quantizor
                .dequantize(Block::<Q>::from(chunk).zagzig())
                .convert_to::<f64>()
                .idct()
        })
        .collect::<Vec<_>>();

    let cb = ans::decode_raw(stream)?
        .par_chunks_exact(Block::<Q>::size())
        .map(|chunk| {
            chroma_quantizor
                .dequantize(Block::<Q>::from(chunk).zagzig())
                .convert_to::<f64>()
                .idct()
        })
        .collect::<Vec<_>>();

    let cr = ans::decode_raw(stream)?
        .par_chunks_exact(Block::<Q>::size())
        .map(|chunk| {
            chroma_quantizor
                .dequantize(Block::<Q>::from(chunk).zagzig())
                .convert_to::<f64>()
                .idct()
        })
        .collect::<Vec<_>>();

    Ok((
        dimensions,
        subsampling,
        reconstruct_pixels(dimensions, &y, &cb, &cr, None, subsampling),
    ))
}
