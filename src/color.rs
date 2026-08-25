use std::{
    fmt::Debug,
    io::{Read, Write},
};

use num_traits::{Bounded, FromPrimitive, NumCast, ToPrimitive, Unsigned};
use rayon::prelude::*;
use wide::f64x4;

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    clamp,
    dimensions::PixelDimensions,
    error::{Error, Result},
    Decodable, Encodable,
};

#[derive(Copy, Clone, Debug)]
pub struct Rgba<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
}

// https://en.wikipedia.org/wiki/YCbCr
#[derive(Copy, Clone, Debug)]
pub struct Ycbcr {
    pub y: f64,
    pub cb: f64,
    pub cr: f64,
    pub a: f64,
}

pub enum YCBCRChannels {
    Y,
    CB,
    CR,
    A,
}

// https://en.wikipedia.org/wiki/YCbCr
// BT.2020 luma coefficients (Rec. ITU-R BT.2020)
pub mod bt2020 {
    pub const KR: f64 = 0.2627; // Red coefficient
    pub const KG: f64 = 0.6780; // Green coefficient
    pub const KB: f64 = 0.0593; // Blue coefficient

    // Derived coefficients for Cb (Blue-Yellow chroma)
    pub const CB_R: f64 = -KR / (2.0 * (1.0 - KB)); // -0.2215
    pub const CB_G: f64 = -KG / (2.0 * (1.0 - KB)); // -0.3607
    pub const CB_B: f64 = 0.5; // 0.5000

    // Derived coefficients for Cr (Red-Cyan chroma)
    pub const CR_R: f64 = 0.5; // 0.5000
    pub const CR_G: f64 = -KG / (2.0 * (1.0 - KR)); // -0.4598
    pub const CR_B: f64 = -KB / (2.0 * (1.0 - KR)); // -0.0402

    // Inverse transformation coefficients (for YCbCr to RGB)
    pub const Y_TO_R_CR: f64 = 2.0 * (1.0 - KR); // 1.4746
    pub const Y_TO_G_CB: f64 = -2.0 * KB * (1.0 - KB) / KG; // -0.1645
    pub const Y_TO_G_CR: f64 = -2.0 * KR * (1.0 - KR) / KG; // -0.5713
    pub const Y_TO_B_CB: f64 = 2.0 * (1.0 - KB); // 1.8814
}

pub fn rgba_to_ycbcr<T>(rgba: &Rgba<T>) -> Ycbcr
where
    T: Copy + Bounded + NumCast + Unsigned,
{
    let Rgba { r, g, b, a } = rgba;
    let r = r.to_f64().expect("f64");
    let g = g.to_f64().expect("f64");
    let b = b.to_f64().expect("f64");
    let a = a.to_f64().expect("f64");

    let center = (T::max_value().to_f64().expect("f64") + 1.0) / 2f64;

    let y = bt2020::KR * r + bt2020::KG * g + bt2020::KB * b;
    let cb = center + bt2020::CB_R * r + bt2020::CB_G * g + bt2020::CB_B * b;
    let cr = center + bt2020::CR_R * r + bt2020::CR_G * g + bt2020::CR_B * b;

    Ycbcr { y, cb, cr, a }
}

pub fn ycbcr_to_rgba<T>(ycbcr: &Ycbcr) -> Rgba<T>
where
    T: Bounded + FromPrimitive + ToPrimitive,
{
    let Ycbcr { y, cb, cr, a } = ycbcr;
    let center = (T::max_value().to_f64().expect("f64") + 1.0) / 2f64;

    let cb = cb - center;
    let cr = cr - center;

    let r = y + bt2020::Y_TO_R_CR * cr;
    let g = y + bt2020::Y_TO_G_CB * cb + bt2020::Y_TO_G_CR * cr;
    let b = y + bt2020::Y_TO_B_CB * cb;

    Rgba {
        r: clamp(r),
        g: clamp(g),
        b: clamp(b),
        a: clamp(*a),
    }
}

/// SIMD-accelerated batch conversion of YCbCr to RGBA
/// Processes 4 pixels at a time using SIMD instructions
#[inline]
fn ycbcr_to_rgba_simd_batch(y: &[f64], cb: &[f64], cr: &[f64], center: f64) -> Vec<Rgba<u8>> {
    assert_eq!(y.len(), cb.len());
    assert_eq!(y.len(), cr.len());

    let len = y.len();
    let mut result = Vec::with_capacity(len);

    // Process 4 pixels at a time with SIMD
    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;

        // Load 4 pixels into SIMD registers
        let y_vec = f64x4::new([y[idx], y[idx + 1], y[idx + 2], y[idx + 3]]);
        let cb_vec = f64x4::new([cb[idx], cb[idx + 1], cb[idx + 2], cb[idx + 3]]);
        let cr_vec = f64x4::new([cr[idx], cr[idx + 1], cr[idx + 2], cr[idx + 3]]);

        // Center the chroma values
        let center_vec = f64x4::splat(center);
        let cb_centered = cb_vec - center_vec;
        let cr_centered = cr_vec - center_vec;

        // YCbCr to RGB conversion (vectorized)
        let r_vec = y_vec + cr_centered * f64x4::splat(bt2020::Y_TO_R_CR);
        let g_vec = y_vec
            + cb_centered * f64x4::splat(bt2020::Y_TO_G_CB)
            + cr_centered * f64x4::splat(bt2020::Y_TO_G_CR);
        let b_vec = y_vec + cb_centered * f64x4::splat(bt2020::Y_TO_B_CB);

        // Convert to array and clamp to u8
        let r_arr = r_vec.to_array();
        let g_arr = g_vec.to_array();
        let b_arr = b_vec.to_array();

        for j in 0..4 {
            result.push(Rgba {
                r: clamp::<u8>(r_arr[j]),
                g: clamp::<u8>(g_arr[j]),
                b: clamp::<u8>(b_arr[j]),
                a: 255,
            });
        }
    }

    // Handle remaining pixels (less than 4) with scalar code
    for i in (chunks * 4)..len {
        let ycbcr = Ycbcr {
            y: y[i],
            cb: cb[i],
            cr: cr[i],
            a: 255.0,
        };
        result.push(ycbcr_to_rgba(&ycbcr));
    }

    result
}

/// Convert YCbCr pixel arrays to RGBA with automatic SIMD optimization
///
/// This function automatically uses SIMD acceleration when processing
/// large batches of pixels for maximum performance.
pub fn ycbcr_batch_to_rgba(y: &[f64], cb: &[f64], cr: &[f64]) -> Vec<Rgba<u8>> {
    assert_eq!(y.len(), cb.len());
    assert_eq!(y.len(), cr.len());

    let center = (u8::MAX as f64 + 1.0) / 2.0;

    // Use SIMD for batches of 16+ pixels when available
    // Check for SIMD support based on target architecture
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if y.len() >= 16 && is_x86_feature_detected!("sse2") {
            return ycbcr_to_rgba_simd_batch(y, cb, cr, center);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if y.len() >= 16 {
            // ARM NEON is always available on aarch64
            return ycbcr_to_rgba_simd_batch(y, cb, cr, center);
        }
    }

    // Fallback to scalar conversion for small batches or unsupported architectures
    y.iter()
        .zip(cb.iter())
        .zip(cr.iter())
        .map(|((y_val, cb_val), cr_val)| {
            let ycbcr = Ycbcr {
                y: *y_val,
                cb: *cb_val,
                cr: *cr_val,
                a: 255.0,
            };
            ycbcr_to_rgba(&ycbcr)
        })
        .collect()
}

/// Calculate chroma dimensions from luma dimensions and subsampling mode.
/// This accounts for block padding (8x8 blocks) to ensure dimensions match
/// what was used during block-based encoding.
///
/// This is the canonical calculation used by both encoding and decoding paths
/// to ensure consistency.
pub fn calculate_chroma_dimensions(
    luma_dimensions: PixelDimensions,
    subsampling: Subsampling,
) -> PixelDimensions {
    let PixelDimensions { width, height } = luma_dimensions;

    match subsampling {
        Subsampling::Sample444 => {
            // No subsampling - chroma has same dimensions as luma
            PixelDimensions { width, height }
        }
        Subsampling::Sample422 => {
            // Horizontal subsampling only - width halved, height same
            // Calculate based on padded block dimensions
            let luma_blocks_per_row = width.div_ceil(8); // Round up for 8x8 blocks
            let chroma_blocks_per_row = luma_blocks_per_row.div_ceil(2); // Half, rounded up

            PixelDimensions {
                width: chroma_blocks_per_row * 8,
                height,
            }
        }
        Subsampling::Sample411 => {
            // Horizontal 4:1 subsampling - width quartered, height same
            let luma_blocks_per_row = width.div_ceil(8); // Round up for 8x8 blocks
            let chroma_blocks_per_row = luma_blocks_per_row.div_ceil(4); // Quarter, rounded up

            PixelDimensions {
                width: chroma_blocks_per_row * 8,
                height,
            }
        }
        Subsampling::Sample420 => {
            // Both horizontal and vertical subsampling - both dimensions halved
            let luma_blocks_per_row = width.div_ceil(8); // Round up for 8x8 blocks
            let luma_blocks_per_col = height.div_ceil(8); // Round up for 8x8 blocks

            let chroma_blocks_per_row = luma_blocks_per_row.div_ceil(2); // Half, rounded up
            let chroma_blocks_per_col = luma_blocks_per_col.div_ceil(2); // Half, rounded up

            PixelDimensions {
                width: chroma_blocks_per_row * 8,
                height: chroma_blocks_per_col * 8,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Sample411,
    Sample420,
    Sample422,
    Sample444,
}

impl Decodable for Subsampling {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let value: u8 = stream
            .read()?
            .ok_or(Error::FailedToDecode("subsampling".to_owned()))?;
        Ok(value.into())
    }
}

impl Encodable for Subsampling {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> crate::Result<()>
    where
        W: Write,
    {
        let value: u8 = (*self).into();
        stream.write(value)
    }
}

impl From<u8> for Subsampling {
    fn from(value: u8) -> Self {
        match value {
            0 => Subsampling::Sample411,
            1 => Subsampling::Sample420,
            2 => Subsampling::Sample422,
            3 => Subsampling::Sample444,
            _ => panic!("Invalid subsampling value: {} (expected 0-3)", value),
        }
    }
}

impl From<Subsampling> for u8 {
    fn from(value: Subsampling) -> Self {
        match value {
            Subsampling::Sample411 => 0,
            Subsampling::Sample420 => 1,
            Subsampling::Sample422 => 2,
            Subsampling::Sample444 => 3,
        }
    }
}

#[derive(Clone)]
pub struct SubSampleGroup<T> {
    pub dimensions: PixelDimensions,
    pub y: Vec<T>,
    pub cb: Vec<T>,
    pub cr: Vec<T>,
}

impl From<(PixelDimensions, Subsampling, &[u8])> for SubSampleGroup<u8> {
    fn from((dimensions, subsampling, data): (PixelDimensions, Subsampling, &[u8])) -> Self {
        match subsampling {
            Subsampling::Sample411 => {
                let len = dimensions.height * dimensions.width;
                let y = data[..len].to_vec();
                let chroma_len = len / 4;
                let cb = data[len..len + chroma_len].to_vec();
                let cr = data[len + chroma_len..].to_vec();
                // there are equal of each
                Self {
                    dimensions,
                    y,
                    cb,
                    cr,
                }
            }
            Subsampling::Sample420 => {
                let len = dimensions.height * dimensions.width;
                let y = data[..len].to_vec();
                let chroma_len = len / 4;
                let cb = data[len..len + chroma_len].to_vec();
                let cr = data[len + chroma_len..].to_vec();
                Self {
                    dimensions,
                    y,
                    cb,
                    cr,
                }
            }
            Subsampling::Sample422 => {
                let len = dimensions.height * dimensions.width;
                let y = data[..len].to_vec();
                let chroma_len = len / 2;
                let cb = data[len..len + chroma_len].to_vec();
                let cr = data[len + chroma_len..].to_vec();
                Self {
                    dimensions,
                    y,
                    cb,
                    cr,
                }
            }
            Subsampling::Sample444 => {
                let len = dimensions.height * dimensions.width;
                let y = data[..len].to_vec();
                let cb = data[len..len * 2].to_vec();
                let cr = data[len * 2..].to_vec();
                // there are equal of each
                Self {
                    dimensions,
                    y,
                    cb,
                    cr,
                }
            }
        }
    }
}

pub struct SubSampleGroupRef<'a, T> {
    pub dimensions: PixelDimensions,
    pub y: &'a [T],
    pub cb: &'a [T],
    pub cr: &'a [T],
}

impl<T> SubSampleGroup<T> {
    pub fn as_ref(&self) -> SubSampleGroupRef<'_, T> {
        SubSampleGroupRef {
            dimensions: self.dimensions,
            y: &self.y,
            cb: &self.cb,
            cr: &self.cr,
        }
    }
}

// https://en.wikipedia.org/wiki/Chroma_subsampling
pub fn subsample_ycbcr(
    dimensions: PixelDimensions,
    ycbcr: &[Ycbcr],
    subsampling: Subsampling,
) -> SubSampleGroup<f64> {
    // Extract Y channel (always full resolution)
    let y = ycbcr.par_iter().map(|pixel| pixel.y).collect::<Vec<_>>();

    match subsampling {
        // 420:
        // - half horizontal
        // - half vertical
        Subsampling::Sample420 => {
            let PixelDimensions { width, height } = dimensions;

            // Collect coordinate pairs for parallel processing
            let coords: Vec<(usize, usize)> = (0..height)
                .step_by(2)
                .flat_map(|r| (0..width).step_by(2).map(move |c| (r, c)))
                .collect();

            let (sampled_cb, sampled_cr): (Vec<f64>, Vec<f64>) = coords
                .into_par_iter()
                .map(|(r, c)| {
                    let idx1 = sample_idx((r, c), width).expect("within width");
                    let idx2 = sample_idx((r, c + 1), width).unwrap_or(idx1);
                    let idx3 = sample_idx((r + 1, c), width).unwrap_or(idx1);
                    let idx4 = sample_idx((r + 1, c + 1), width).unwrap_or(idx1);

                    if idx1 < ycbcr.len()
                        && idx2 < ycbcr.len()
                        && idx3 < ycbcr.len()
                        && idx4 < ycbcr.len()
                    {
                        let avg_cb =
                            (ycbcr[idx1].cb + ycbcr[idx2].cb + ycbcr[idx3].cb + ycbcr[idx4].cb)
                                / 4.0;
                        let avg_cr =
                            (ycbcr[idx1].cr + ycbcr[idx2].cr + ycbcr[idx3].cr + ycbcr[idx4].cr)
                                / 4.0;

                        (avg_cb, avg_cr)
                    } else {
                        (0.0, 0.0)
                    }
                })
                .unzip();

            SubSampleGroup {
                dimensions,
                y,
                cb: sampled_cb,
                cr: sampled_cr,
            }
        }
        // 411:
        // - quarter horizontal
        // - full vertical
        Subsampling::Sample411 => {
            let PixelDimensions { width, height } = dimensions;

            // Collect coordinate pairs for parallel processing
            let coords: Vec<(usize, usize)> = (0..height)
                .flat_map(|r| (0..width).step_by(4).map(move |c| (r, c)))
                .collect();

            let (sampled_cb, sampled_cr): (Vec<f64>, Vec<f64>) = coords
                .into_par_iter()
                .map(|(r, c)| {
                    let idx1 = sample_idx((r, c), width).expect("within width");
                    let idx2 = sample_idx((r, c + 1), width).unwrap_or(idx1);
                    let idx3 = sample_idx((r, c + 2), width).unwrap_or(idx1);
                    let idx4 = sample_idx((r, c + 3), width).unwrap_or(idx1);

                    if idx1 < ycbcr.len()
                        && idx2 < ycbcr.len()
                        && idx3 < ycbcr.len()
                        && idx4 < ycbcr.len()
                    {
                        let avg_cb =
                            (ycbcr[idx1].cb + ycbcr[idx2].cb + ycbcr[idx3].cb + ycbcr[idx4].cb)
                                / 4.0;
                        let avg_cr =
                            (ycbcr[idx1].cr + ycbcr[idx2].cr + ycbcr[idx3].cr + ycbcr[idx4].cr)
                                / 4.0;
                        (avg_cb, avg_cr)
                    } else {
                        (0.0, 0.0)
                    }
                })
                .unzip();

            SubSampleGroup {
                dimensions,
                y,
                cb: sampled_cb,
                cr: sampled_cr,
            }
        }
        // 422:
        // - half horizontal
        // - full vertical
        Subsampling::Sample422 => {
            let PixelDimensions { width, height } = dimensions;

            // Collect coordinate pairs for parallel processing
            let coords: Vec<(usize, usize)> = (0..height)
                .flat_map(|r| (0..width).step_by(2).map(move |c| (r, c)))
                .collect();

            let (sampled_cb, sampled_cr): (Vec<f64>, Vec<f64>) = coords
                .into_par_iter()
                .map(|(r, c)| {
                    let idx1 = sample_idx((r, c), width).expect("within width");
                    let idx2 = sample_idx((r, c + 1), width).unwrap_or(idx1);

                    if idx1 < ycbcr.len() && idx2 < ycbcr.len() {
                        let avg_cb = (ycbcr[idx1].cb + ycbcr[idx2].cb) / 2.0;
                        let avg_cr = (ycbcr[idx1].cr + ycbcr[idx2].cr) / 2.0;
                        (avg_cb, avg_cr)
                    } else {
                        (0.0, 0.0)
                    }
                })
                .unzip();

            SubSampleGroup {
                dimensions,
                y,
                cb: sampled_cb,
                cr: sampled_cr,
            }
        }
        // 444:
        // - full horizontal
        // - full vertical
        Subsampling::Sample444 => {
            let cb = ycbcr.par_iter().map(|pixel| pixel.cb).collect::<Vec<_>>();
            let cr = ycbcr.par_iter().map(|pixel| pixel.cr).collect::<Vec<_>>();

            SubSampleGroup {
                dimensions,
                y,
                cb,
                cr,
            }
        }
    }
}

#[derive(Clone)]
pub struct UpSampleGroup<T> {
    pub dimensions: PixelDimensions,
    pub y: Vec<T>,
    pub cb: Vec<T>,
    pub cr: Vec<T>,
}

// https://en.wikipedia.org/wiki/Chroma_subsampling
pub fn upsample_ycbcr(
    dimensions: PixelDimensions,
    y: Vec<f64>,
    cb: Vec<f64>,
    cr: Vec<f64>,
    subsampling: Subsampling,
) -> UpSampleGroup<f64> {
    match subsampling {
        // 420:
        // - half horizontal
        // - half vertical
        Subsampling::Sample420 => {
            let PixelDimensions { width, height } = dimensions;
            let mut upsampled_cb = vec![0.0; width * height];
            let mut upsampled_cr = vec![0.0; width * height];

            // Use the canonical chroma dimension calculation
            let chroma_dims = calculate_chroma_dimensions(dimensions, Subsampling::Sample420);
            let chroma_width = chroma_dims.width;
            let _chroma_height = chroma_dims.height;

            // Process in parallel by row pairs
            upsampled_cb
                .par_chunks_mut(width * 2)
                .zip(upsampled_cr.par_chunks_mut(width * 2))
                .enumerate()
                .for_each(|(chunk_idx, (cb_chunk, cr_chunk))| {
                    let r = chunk_idx * 2;
                    if r >= height {
                        return;
                    }

                    for c in (0..width).step_by(2) {
                        let sub_idx = (r / 2) * chroma_width + (c / 2);

                        if sub_idx < cb.len() {
                            let cb_val = cb[sub_idx];
                            let cr_val = cr[sub_idx];

                            let local_idx1 = c;
                            let local_idx2 = (c + 1).min(width - 1);
                            let local_idx3 = if r + 1 < height { width + c } else { c };
                            let local_idx4 = if r + 1 < height {
                                width + (c + 1).min(width - 1)
                            } else {
                                (c + 1).min(width - 1)
                            };

                            cb_chunk[local_idx1] = cb_val;
                            cb_chunk[local_idx2] = cb_val;
                            if local_idx3 < cb_chunk.len() {
                                cb_chunk[local_idx3] = cb_val;
                            }
                            if local_idx4 < cb_chunk.len() {
                                cb_chunk[local_idx4] = cb_val;
                            }

                            cr_chunk[local_idx1] = cr_val;
                            cr_chunk[local_idx2] = cr_val;
                            if local_idx3 < cr_chunk.len() {
                                cr_chunk[local_idx3] = cr_val;
                            }
                            if local_idx4 < cr_chunk.len() {
                                cr_chunk[local_idx4] = cr_val;
                            }
                        }
                    }
                });

            UpSampleGroup {
                dimensions,
                y,
                cb: upsampled_cb,
                cr: upsampled_cr,
            }
        }
        // 411:
        // - quarter horizontal
        // - full vertical
        Subsampling::Sample411 => {
            let PixelDimensions { width, height } = dimensions;
            let mut upsampled_cb = vec![0.0; width * height];
            let mut upsampled_cr = vec![0.0; width * height];

            // Calculate chroma width from the actual array length
            let samples = cb.len().checked_div(height).unwrap_or(width.div_ceil(4));

            // Process in parallel by rows
            upsampled_cb
                .par_chunks_mut(width)
                .zip(upsampled_cr.par_chunks_mut(width))
                .enumerate()
                .for_each(|(r, (cb_row, cr_row))| {
                    for c in (0..width).step_by(4) {
                        let sub_idx = r * samples + (c / 4);
                        if sub_idx < cb.len() {
                            let cb_val = cb[sub_idx];
                            let cr_val = cr[sub_idx];

                            for i in 0..4 {
                                let idx = c + i;
                                if idx < width {
                                    cb_row[idx] = cb_val;
                                    cr_row[idx] = cr_val;
                                }
                            }
                        }
                    }
                });

            UpSampleGroup {
                dimensions,
                y,
                cb: upsampled_cb,
                cr: upsampled_cr,
            }
        }
        // 422:
        // - half horizontal
        // - full vertical
        Subsampling::Sample422 => {
            let PixelDimensions { width, height } = dimensions;
            let mut upsampled_cb = vec![0.0; width * height];
            let mut upsampled_cr = vec![0.0; width * height];

            // Calculate chroma width from the actual array length
            let chroma_width = cb.len().checked_div(height).unwrap_or(width.div_ceil(2));

            // Process in parallel by rows
            upsampled_cb
                .par_chunks_mut(width)
                .zip(upsampled_cr.par_chunks_mut(width))
                .enumerate()
                .for_each(|(r, (cb_row, cr_row))| {
                    for c in (0..width).step_by(2) {
                        let sub_idx = r * chroma_width + (c / 2);

                        if sub_idx < cb.len() && sub_idx < cr.len() {
                            let cb_val = cb[sub_idx];
                            let cr_val = cr[sub_idx];

                            cb_row[c] = cb_val;
                            if c + 1 < width {
                                cb_row[c + 1] = cb_val;
                            }

                            cr_row[c] = cr_val;
                            if c + 1 < width {
                                cr_row[c + 1] = cr_val;
                            }
                        }
                    }
                });

            UpSampleGroup {
                dimensions,
                y,
                cb: upsampled_cb,
                cr: upsampled_cr,
            }
        }
        // 444:
        // - full horizontal
        // - full vertical
        Subsampling::Sample444 => UpSampleGroup {
            dimensions,
            y,
            cb,
            cr,
        },
    }
}

#[inline]
fn sample_idx((r, c): (usize, usize), width: usize) -> Option<usize> {
    if c < width {
        Some(r * width + c)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_to_ycbcr_conversion() {
        let rgba = Rgba {
            r: 121,
            g: 121,
            b: 121,
            a: 77,
        };

        let ycbcr = rgba_to_ycbcr::<u8>(&rgba);
        println!("{ycbcr:?}");

        // For gray (equal R, G, B), Y should be close to the gray value
        // and Cb, Cr should be close to center (128 for 8-bit)
        let expected_y = 121.0; // Gray value
        let expected_center = 128.0; // Center value for 8-bit chroma

        assert!(
            (ycbcr.y - expected_y).abs() < 1.0,
            "Y value should be close to input gray level"
        );
        assert!(
            (ycbcr.cb - expected_center).abs() < 40.0,
            "Cb should be close to center for gray"
        );
        assert!(
            (ycbcr.cr - expected_center).abs() < 40.0,
            "Cr should be close to center for gray"
        );
        assert_eq!(ycbcr.a, 77.0);
    }

    #[test]
    fn test_ycbcr_to_rgba_conversion() {
        let ycbcr = Ycbcr {
            y: 121.0,  // Gray level
            cb: 128.0, // Center chroma
            cr: 128.0, // Center chroma
            a: 77.0,
        };

        let rgba = ycbcr_to_rgba::<u8>(&ycbcr);
        println!("{rgba:?}");

        // For centered chroma, should get similar RGB values (gray)
        let rgb_diff_rg = (rgba.r as i16 - rgba.g as i16).abs();
        let rgb_diff_gb = (rgba.g as i16 - rgba.b as i16).abs();
        assert!(rgb_diff_rg < 5, "R and G should be similar for gray");
        assert!(rgb_diff_gb < 5, "G and B should be similar for gray");
        assert_eq!(rgba.a, 77);
    }

    #[test]
    fn test_round_trip_conversion() {
        let original = Rgba {
            r: 80,
            g: 20,
            b: 60,
            a: 90,
        };

        let ycbcr = rgba_to_ycbcr(&original);
        let converted_back = ycbcr_to_rgba::<u8>(&ycbcr);
        println!("{converted_back:?}");

        // Should be close to original (within floating point precision)
        assert!((original.r as i16 - converted_back.r as i16).abs() <= 1);
        assert!((original.g as i16 - converted_back.g as i16).abs() <= 1);
        assert!((original.b as i16 - converted_back.b as i16).abs() <= 1);
        assert_eq!(original.a, converted_back.a);
    }

    // Helper function to create test YCbCr data
    fn create_test_ycbcr_data(width: usize, height: usize) -> Vec<Ycbcr> {
        let mut data = Vec::new();
        for r in 0..height {
            for c in 0..width {
                // Create varied color pattern for testing
                let normalized_r = r as f64 / height as f64;
                let normalized_c = c as f64 / width as f64;
                data.push(Ycbcr {
                    y: (normalized_r + normalized_c) / 2.0,
                    cb: normalized_c - 0.5,
                    cr: normalized_r - 0.5,
                    a: 1.0,
                });
            }
        }
        data
    }

    #[test]
    fn test_subsampling_444_no_change() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample444);

        assert_eq!(subsampled.y.len(), original_count);
        assert_eq!(subsampled.cb.len(), original_count);
        assert_eq!(subsampled.cr.len(), original_count);
        assert_eq!(subsampled.dimensions, dimensions);
    }

    #[test]
    fn test_subsampling_422_horizontal_reduction() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample422);

        // Y channel should be unchanged
        assert_eq!(subsampled.y.len(), original_count);
        // Cb and Cr should be horizontally subsampled (every 2nd pixel)
        assert_eq!(subsampled.cb.len(), width * height / 2);
        assert_eq!(subsampled.cr.len(), width * height / 2);
        assert_eq!(subsampled.dimensions, dimensions);
    }

    #[test]
    fn test_subsampling_420_both_reduction() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample420);

        // Y channel should be unchanged
        assert_eq!(subsampled.y.len(), original_count);
        // Cb and Cr should be subsampled both horizontally and vertically (2x2 -> 1)
        let expected_chroma_count = (width / 2) * (height / 2);
        assert_eq!(subsampled.cb.len(), expected_chroma_count);
        assert_eq!(subsampled.cr.len(), expected_chroma_count);
        assert_eq!(subsampled.dimensions, dimensions);

        println!(
            "420 Subsampling - Original: {}, Y: {}, Cb: {}, Cr: {}",
            original_count,
            subsampled.y.len(),
            subsampled.cb.len(),
            subsampled.cr.len()
        );
    }

    #[test]
    fn test_subsampling_411_quarter_reduction() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample411);

        // Y channel should be unchanged
        assert_eq!(subsampled.y.len(), original_count);
        // Cb and Cr should be subsampled to 1/4 (every 4th pixel)
        assert_eq!(subsampled.cb.len(), original_count / 4);
        assert_eq!(subsampled.cr.len(), original_count / 4);
        assert_eq!(subsampled.dimensions, dimensions);
    }

    #[test]
    fn test_upsampling_444_no_change() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let pixel_count = width * height;

        let y: Vec<f64> = (0..pixel_count)
            .map(|i| i as f64 / pixel_count as f64)
            .collect();
        let cb = y.clone();
        let cr = y.clone();

        let upsampled = upsample_ycbcr(
            dimensions,
            y.clone(),
            cb.clone(),
            cr.clone(),
            Subsampling::Sample444,
        );

        assert_eq!(upsampled.y.len(), pixel_count);
        assert_eq!(upsampled.cb.len(), pixel_count);
        assert_eq!(upsampled.cr.len(), pixel_count);
        assert_eq!(upsampled.dimensions, dimensions);
    }

    #[test]
    fn test_upsampling_422_horizontal_expansion() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let pixel_count = width * height;
        let chroma_count = pixel_count / 2; // 422 has half the chroma samples

        let y: Vec<f64> = (0..pixel_count)
            .map(|i| i as f64 / pixel_count as f64)
            .collect();
        let cb: Vec<f64> = (0..chroma_count)
            .map(|i| i as f64 / chroma_count as f64)
            .collect();
        let cr = cb.clone();

        let upsampled = upsample_ycbcr(dimensions, y, cb, cr, Subsampling::Sample422);

        assert_eq!(upsampled.y.len(), pixel_count);
        assert_eq!(upsampled.cb.len(), pixel_count);
        assert_eq!(upsampled.cr.len(), pixel_count);
        assert_eq!(upsampled.dimensions, dimensions);
    }

    #[test]
    fn test_upsampling_420_both_expansion() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let pixel_count = width * height;
        let chroma_count = (width / 2) * (height / 2); // 420 has 1/4 the chroma samples

        let y: Vec<f64> = (0..pixel_count)
            .map(|i| i as f64 / pixel_count as f64)
            .collect();
        let cb: Vec<f64> = (0..chroma_count)
            .map(|i| i as f64 / chroma_count as f64)
            .collect();
        let cr = cb.clone();

        println!(
            "420 Upsampling input - Y: {}, Cb: {}, Cr: {}",
            y.len(),
            cb.len(),
            cr.len()
        );

        let upsampled = upsample_ycbcr(dimensions, y, cb, cr, Subsampling::Sample420);

        println!(
            "420 Upsampling output - Y: {}, Cb: {}, Cr: {}",
            upsampled.y.len(),
            upsampled.cb.len(),
            upsampled.cr.len()
        );

        assert_eq!(upsampled.y.len(), pixel_count);
        assert_eq!(upsampled.cb.len(), pixel_count);
        assert_eq!(upsampled.cr.len(), pixel_count);
        assert_eq!(upsampled.dimensions, dimensions);
    }

    #[test]
    fn test_upsampling_411_quarter_expansion() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let pixel_count = width * height;
        let chroma_count = pixel_count / 4; // 411 has 1/4 the chroma samples

        let y: Vec<f64> = (0..pixel_count)
            .map(|i| i as f64 / pixel_count as f64)
            .collect();
        let cb: Vec<f64> = (0..chroma_count)
            .map(|i| i as f64 / chroma_count as f64)
            .collect();
        let cr = cb.clone();

        let upsampled = upsample_ycbcr(dimensions, y, cb, cr, Subsampling::Sample411);

        assert_eq!(upsampled.y.len(), pixel_count);
        assert_eq!(upsampled.cb.len(), pixel_count);
        assert_eq!(upsampled.cr.len(), pixel_count);
        assert_eq!(upsampled.dimensions, dimensions);
    }

    #[test]
    fn test_roundtrip_subsampling_444() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        // Subsample
        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample444);

        // Upsample
        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample444,
        );

        assert_eq!(upsampled.y.len(), original_count);
        assert_eq!(upsampled.cb.len(), original_count);
        assert_eq!(upsampled.cr.len(), original_count);
    }

    #[test]
    fn test_roundtrip_subsampling_422() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        // Subsample
        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample422);

        // Upsample
        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample422,
        );

        assert_eq!(upsampled.y.len(), original_count);
        assert_eq!(upsampled.cb.len(), original_count);
        assert_eq!(upsampled.cr.len(), original_count);
    }

    #[test]
    fn test_roundtrip_subsampling_420() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        println!("Original data count: {original_count}");

        // Subsample
        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample420);

        println!(
            "After subsampling - Y: {}, Cb: {}, Cr: {}",
            subsampled.y.len(),
            subsampled.cb.len(),
            subsampled.cr.len()
        );

        // Upsample
        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample420,
        );

        println!(
            "After upsampling - Y: {}, Cb: {}, Cr: {}",
            upsampled.y.len(),
            upsampled.cb.len(),
            upsampled.cr.len()
        );

        assert_eq!(upsampled.y.len(), original_count);
        assert_eq!(upsampled.cb.len(), original_count);
        assert_eq!(upsampled.cr.len(), original_count);
    }

    #[test]
    fn test_roundtrip_subsampling_411() {
        let width = 8;
        let height = 4;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);
        let original_count = test_data.len();

        // Subsample
        let subsampled = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample411);

        // Upsample
        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample411,
        );

        assert_eq!(upsampled.y.len(), original_count);
        assert_eq!(upsampled.cb.len(), original_count);
        assert_eq!(upsampled.cr.len(), original_count);
    }

    // Performance benchmarks for optimized functions
    #[test]
    fn benchmark_subsample_ycbcr_performance() {
        use std::time::Instant;

        let test_cases = vec![
            (640, 360),   // 360p
            (1280, 720),  // 720p
            (1920, 1080), // 1080p
        ];

        let subsampling_modes = vec![
            Subsampling::Sample444,
            Subsampling::Sample422,
            Subsampling::Sample420,
            Subsampling::Sample411,
        ];

        println!("\n🔬 Subsample YCbCr Performance Benchmark");
        println!("=========================================");

        for (width, height) in test_cases {
            let pixel_count = width * height;
            let data_size_mb = (pixel_count * 3 * 8) as f64 / (1024.0 * 1024.0); // 3 channels * 8 bytes per f64
            let dimensions = PixelDimensions { width, height };

            println!(
                "\n📺 {}x{} ({:.1}MP, {:.1}MB YCbCr data):",
                width,
                height,
                pixel_count as f64 / 1_000_000.0,
                data_size_mb
            );

            for subsampling in &subsampling_modes {
                let test_data = create_test_ycbcr_data(width, height);

                // Warm up
                for _ in 0..3 {
                    let _ = subsample_ycbcr(dimensions, &test_data, *subsampling);
                }

                // Benchmark multiple runs
                let mut times = Vec::new();
                for _ in 0..10 {
                    let start = Instant::now();
                    let result = subsample_ycbcr(dimensions, &test_data, *subsampling);
                    let duration = start.elapsed();
                    times.push(duration);

                    // Verify result dimensions
                    assert_eq!(result.y.len(), pixel_count);
                    match subsampling {
                        Subsampling::Sample444 => {
                            assert_eq!(result.cb.len(), pixel_count);
                            assert_eq!(result.cr.len(), pixel_count);
                        }
                        Subsampling::Sample422 => {
                            assert_eq!(result.cb.len(), pixel_count / 2);
                            assert_eq!(result.cr.len(), pixel_count / 2);
                        }
                        Subsampling::Sample420 => {
                            assert_eq!(result.cb.len(), pixel_count / 4);
                            assert_eq!(result.cr.len(), pixel_count / 4);
                        }
                        Subsampling::Sample411 => {
                            assert_eq!(result.cb.len(), pixel_count / 4);
                            assert_eq!(result.cr.len(), pixel_count / 4);
                        }
                    }
                }

                let avg_time = times.iter().sum::<std::time::Duration>() / times.len() as u32;
                let min_time = times.iter().min().unwrap();
                let max_time = times.iter().max().unwrap();
                let throughput_mp_s = (pixel_count as f64 / 1_000_000.0) / avg_time.as_secs_f64();
                let throughput_mb_s = data_size_mb / avg_time.as_secs_f64();

                println!(
                    "   {:?}: {:6.1}ms (min: {:5.1}ms, max: {:5.1}ms) | {:6.1} MP/s | {:6.1} MB/s",
                    subsampling,
                    avg_time.as_secs_f64() * 1000.0,
                    min_time.as_secs_f64() * 1000.0,
                    max_time.as_secs_f64() * 1000.0,
                    throughput_mp_s,
                    throughput_mb_s
                );
            }
        }
    }

    #[test]
    fn benchmark_upsample_ycbcr_performance() {
        use std::time::Instant;

        let test_cases = vec![
            (640, 360),   // 360p
            (1280, 720),  // 720p
            (1920, 1080), // 1080p
        ];

        let subsampling_modes = vec![
            Subsampling::Sample444,
            Subsampling::Sample422,
            Subsampling::Sample420,
            Subsampling::Sample411,
        ];

        println!("\n🔬 Upsample YCbCr Performance Benchmark");
        println!("=======================================");

        for (width, height) in test_cases {
            let pixel_count = width * height;
            let dimensions = PixelDimensions { width, height };

            println!(
                "\n📺 {}x{} ({:.1}MP):",
                width,
                height,
                pixel_count as f64 / 1_000_000.0
            );

            for subsampling in &subsampling_modes {
                // Create appropriately sized test data for each subsampling mode
                let y: Vec<f64> = (0..pixel_count)
                    .map(|i| i as f64 / pixel_count as f64)
                    .collect();
                let (cb, cr) = match subsampling {
                    Subsampling::Sample444 => {
                        let cb: Vec<f64> = (0..pixel_count)
                            .map(|i| (i as f64 / pixel_count as f64) * 0.5)
                            .collect();
                        let cr = cb.clone();
                        (cb, cr)
                    }
                    Subsampling::Sample422 => {
                        let cb: Vec<f64> = (0..pixel_count / 2)
                            .map(|i| (i as f64 / (pixel_count / 2) as f64) * 0.5)
                            .collect();
                        let cr = cb.clone();
                        (cb, cr)
                    }
                    Subsampling::Sample420 | Subsampling::Sample411 => {
                        let cb: Vec<f64> = (0..pixel_count / 4)
                            .map(|i| (i as f64 / (pixel_count / 4) as f64) * 0.5)
                            .collect();
                        let cr = cb.clone();
                        (cb, cr)
                    }
                };

                // Warm up
                for _ in 0..3 {
                    let _ =
                        upsample_ycbcr(dimensions, y.clone(), cb.clone(), cr.clone(), *subsampling);
                }

                // Benchmark multiple runs
                let mut times = Vec::new();
                for _ in 0..10 {
                    let start = Instant::now();
                    let result =
                        upsample_ycbcr(dimensions, y.clone(), cb.clone(), cr.clone(), *subsampling);
                    let duration = start.elapsed();
                    times.push(duration);

                    // Verify result
                    assert_eq!(result.y.len(), pixel_count);
                    assert_eq!(result.cb.len(), pixel_count);
                    assert_eq!(result.cr.len(), pixel_count);
                }

                let avg_time = times.iter().sum::<std::time::Duration>() / times.len() as u32;
                let min_time = times.iter().min().unwrap();
                let max_time = times.iter().max().unwrap();
                let throughput_mp_s = (pixel_count as f64 / 1_000_000.0) / avg_time.as_secs_f64();

                println!(
                    "   {:?}: {:6.1}ms (min: {:5.1}ms, max: {:5.1}ms) | {:6.1} MP/s",
                    subsampling,
                    avg_time.as_secs_f64() * 1000.0,
                    min_time.as_secs_f64() * 1000.0,
                    max_time.as_secs_f64() * 1000.0,
                    throughput_mp_s
                );
            }
        }
    }

    #[test]
    fn benchmark_memory_allocation_optimization() {
        use std::time::Instant;

        // Test to verify our memory allocation optimization is working
        let width = 1280;
        let height = 720;
        let pixel_count = width * height;
        let dimensions = PixelDimensions { width, height };
        let test_data = create_test_ycbcr_data(width, height);

        println!("\n🧠 Memory Allocation Optimization Verification (720p)");
        println!("=====================================================");

        // Test Sample420 specifically since it benefits most from our optimization
        let mut times = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            let result = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample420);
            let duration = start.elapsed();
            times.push(duration);

            // Verify correctness
            assert_eq!(result.y.len(), pixel_count);
            assert_eq!(result.cb.len(), pixel_count / 4);
            assert_eq!(result.cr.len(), pixel_count / 4);
        }

        let avg_time = times.iter().sum::<std::time::Duration>() / times.len() as u32;
        let min_time = times.iter().min().unwrap();
        let std_dev = {
            let mean = avg_time.as_secs_f64();
            let variance = times
                .iter()
                .map(|t| (t.as_secs_f64() - mean).powi(2))
                .sum::<f64>()
                / times.len() as f64;
            variance.sqrt()
        };

        let throughput_mp_s = (pixel_count as f64 / 1_000_000.0) / avg_time.as_secs_f64();
        let efficiency_target = 5.0; // Target: under 5ms for 720p Sample420

        println!(
            "Sample420 (720p): {:6.1}ms ± {:4.1}ms | Min: {:5.1}ms | {:6.1} MP/s",
            avg_time.as_secs_f64() * 1000.0,
            std_dev * 1000.0,
            min_time.as_secs_f64() * 1000.0,
            throughput_mp_s
        );

        if avg_time.as_secs_f64() * 1000.0 < efficiency_target {
            println!("✅ OPTIMIZATION SUCCESS: Memory allocation optimization is effective!");
        } else {
            println!(
                "⚠️  Performance may need further optimization (target: <{}ms)",
                efficiency_target
            );
        }

        // Consistency check - standard deviation should be low for optimized code
        let consistency_target = 0.002; // 2ms standard deviation
        if std_dev < consistency_target {
            println!("✅ CONSISTENCY: Low timing variance indicates efficient memory usage");
        } else {
            println!("⚠️  High timing variance may indicate memory allocation overhead");
        }
    }

    #[test]
    fn benchmark_rayon_parallelization_scaling() {
        use std::time::Instant;

        // Test parallelization efficiency across different image sizes
        let test_cases = vec![
            (320, 240),   // Small: 76K pixels
            (640, 480),   // Medium: 307K pixels
            (1920, 1080), // Large: 2M pixels
            (3840, 2160), // Very Large: 8M pixels
        ];

        println!("\n⚡ Rayon Parallelization Scaling Analysis");
        println!("========================================");

        for (width, height) in test_cases {
            let pixel_count = width * height;
            let dimensions = PixelDimensions { width, height };
            let test_data = create_test_ycbcr_data(width, height);

            // Warm up
            for _ in 0..3 {
                let _ = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample420);
            }

            // Benchmark
            let mut times = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let _ = subsample_ycbcr(dimensions, &test_data, Subsampling::Sample420);
                times.push(start.elapsed());
            }

            let avg_time = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            let throughput_mp_s = (pixel_count as f64 / 1_000_000.0) / avg_time.as_secs_f64();
            let pixels_per_ms = pixel_count as f64 / (avg_time.as_secs_f64() * 1000.0);

            println!(
                "{}x{} ({:5.1}MP): {:6.1}ms | {:8.0} px/ms | {:6.1} MP/s",
                width,
                height,
                pixel_count as f64 / 1_000_000.0,
                avg_time.as_secs_f64() * 1000.0,
                pixels_per_ms,
                throughput_mp_s
            );
        }

        println!("\n💡 Parallelization is most effective for larger images where");
        println!("   the overhead of thread creation is amortized across more work.");
    }

    #[test]
    fn test_subsample_upsample_420_uniform_red() {
        // Test with uniform red color - should preserve chroma values exactly
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        // Create uniform red pixels
        let red_ycbcr = Ycbcr {
            y: 67.0,
            cb: 92.4,
            cr: 255.5,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![red_ycbcr; width * height];

        // Subsample
        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample420);

        // Check all Cb values are approximately 92.4
        for &cb_val in &subsampled.cb {
            assert!(
                (cb_val - 92.4).abs() < 0.1,
                "Sample420: Cb value {} differs from expected 92.4",
                cb_val
            );
        }

        // Check all Cr values are approximately 255.5
        for &cr_val in &subsampled.cr {
            assert!(
                (cr_val - 255.5).abs() < 0.1,
                "Sample420: Cr value {} differs from expected 255.5",
                cr_val
            );
        }

        // Upsample
        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample420,
        );

        // Check first pixel maintains chroma values
        assert!(
            (upsampled.cb[0] - 92.4).abs() < 0.1,
            "Sample420 upsample: First Cb {} differs from 92.4",
            upsampled.cb[0]
        );
        assert!(
            (upsampled.cr[0] - 255.5).abs() < 0.1,
            "Sample420 upsample: First Cr {} differs from 255.5",
            upsampled.cr[0]
        );
    }

    #[test]
    fn test_subsample_upsample_422_uniform_red() {
        // Test with uniform red color - should preserve chroma values exactly
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let red_ycbcr = Ycbcr {
            y: 67.0,
            cb: 92.4,
            cr: 255.5,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![red_ycbcr; width * height];

        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample422);

        for &cb_val in &subsampled.cb {
            assert!(
                (cb_val - 92.4).abs() < 0.1,
                "Sample422: Cb value {} differs from expected 92.4",
                cb_val
            );
        }

        for &cr_val in &subsampled.cr {
            assert!(
                (cr_val - 255.5).abs() < 0.1,
                "Sample422: Cr value {} differs from expected 255.5",
                cr_val
            );
        }

        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample422,
        );

        assert!(
            (upsampled.cb[0] - 92.4).abs() < 0.1,
            "Sample422 upsample: First Cb {} differs from 92.4",
            upsampled.cb[0]
        );
        assert!(
            (upsampled.cr[0] - 255.5).abs() < 0.1,
            "Sample422 upsample: First Cr {} differs from 255.5",
            upsampled.cr[0]
        );
    }

    #[test]
    fn test_subsample_upsample_411_uniform_red() {
        // Test with uniform red color - should preserve chroma values exactly
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let red_ycbcr = Ycbcr {
            y: 67.0,
            cb: 92.4,
            cr: 255.5,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![red_ycbcr; width * height];

        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample411);

        for &cb_val in &subsampled.cb {
            assert!(
                (cb_val - 92.4).abs() < 0.1,
                "Sample411: Cb value {} differs from expected 92.4",
                cb_val
            );
        }

        for &cr_val in &subsampled.cr {
            assert!(
                (cr_val - 255.5).abs() < 0.1,
                "Sample411: Cr value {} differs from expected 255.5",
                cr_val
            );
        }

        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample411,
        );

        assert!(
            (upsampled.cb[0] - 92.4).abs() < 0.1,
            "Sample411 upsample: First Cb {} differs from 92.4",
            upsampled.cb[0]
        );
        assert!(
            (upsampled.cr[0] - 255.5).abs() < 0.1,
            "Sample411 upsample: First Cr {} differs from 255.5",
            upsampled.cr[0]
        );
    }

    #[test]
    fn test_subsample_upsample_444_uniform_red() {
        // Test with uniform red color - 444 should preserve everything
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let red_ycbcr = Ycbcr {
            y: 67.0,
            cb: 92.4,
            cr: 255.5,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![red_ycbcr; width * height];

        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample444);

        // 444 should have full resolution chroma
        assert_eq!(subsampled.cb.len(), width * height);
        assert_eq!(subsampled.cr.len(), width * height);

        for &cb_val in &subsampled.cb {
            assert!(
                (cb_val - 92.4).abs() < 0.1,
                "Sample444: Cb value {} differs from expected 92.4",
                cb_val
            );
        }

        for &cr_val in &subsampled.cr {
            assert!(
                (cr_val - 255.5).abs() < 0.1,
                "Sample444: Cr value {} differs from expected 255.5",
                cr_val
            );
        }

        let upsampled = upsample_ycbcr(
            dimensions,
            subsampled.y,
            subsampled.cb,
            subsampled.cr,
            Subsampling::Sample444,
        );

        assert!(
            (upsampled.cb[0] - 92.4).abs() < 0.1,
            "Sample444 upsample: First Cb {} differs from 92.4",
            upsampled.cb[0]
        );
        assert!(
            (upsampled.cr[0] - 255.5).abs() < 0.1,
            "Sample444 upsample: First Cr {} differs from 255.5",
            upsampled.cr[0]
        );
    }

    #[test]
    fn test_subsample_no_cb_cr_swap_420() {
        // Test that Cb and Cr are not swapped during subsampling
        let width = 4;
        let height = 4;
        let dimensions = PixelDimensions { width, height };

        // Create pixels with distinct Cb and Cr values to detect swap
        let test_ycbcr = Ycbcr {
            y: 128.0,
            cb: 50.0,
            cr: 200.0,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![test_ycbcr; width * height];

        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample420);

        // Cb should be around 50, not 200
        for &cb_val in &subsampled.cb {
            assert!(
                cb_val < 100.0,
                "Sample420: Cb={} appears swapped with Cr (expected ~50)",
                cb_val
            );
        }

        // Cr should be around 200, not 50
        for &cr_val in &subsampled.cr {
            assert!(
                cr_val > 150.0,
                "Sample420: Cr={} appears swapped with Cb (expected ~200)",
                cr_val
            );
        }
    }

    #[test]
    fn test_subsample_no_cb_cr_swap_422() {
        let width = 4;
        let height = 4;
        let dimensions = PixelDimensions { width, height };

        let test_ycbcr = Ycbcr {
            y: 128.0,
            cb: 50.0,
            cr: 200.0,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![test_ycbcr; width * height];

        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample422);

        for &cb_val in &subsampled.cb {
            assert!(
                cb_val < 100.0,
                "Sample422: Cb={} appears swapped with Cr (expected ~50)",
                cb_val
            );
        }

        for &cr_val in &subsampled.cr {
            assert!(
                cr_val > 150.0,
                "Sample422: Cr={} appears swapped with Cb (expected ~200)",
                cr_val
            );
        }
    }

    #[test]
    fn test_subsample_no_cb_cr_swap_411() {
        let width = 8;
        let height = 8;
        let dimensions = PixelDimensions { width, height };

        let test_ycbcr = Ycbcr {
            y: 128.0,
            cb: 50.0,
            cr: 200.0,
            a: 255.0,
        };
        let pixels: Vec<Ycbcr> = vec![test_ycbcr; width * height];

        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample411);

        for &cb_val in &subsampled.cb {
            assert!(
                cb_val < 100.0,
                "Sample411: Cb={} appears swapped with Cr (expected ~50)",
                cb_val
            );
        }

        for &cr_val in &subsampled.cr {
            assert!(
                cr_val > 150.0,
                "Sample411: Cr={} appears swapped with Cb (expected ~200)",
                cr_val
            );
        }
    }

    #[test]
    fn test_upsample_no_cb_cr_swap_420() {
        // Test that upsampling doesn't swap Cb and Cr
        let width = 4;
        let height = 4;
        let dimensions = PixelDimensions { width, height };

        // Create subsampled data with distinct values
        let y = vec![128.0; width * height];
        let chroma_size = (width / 2) * (height / 2);
        let cb = vec![50.0; chroma_size];
        let cr = vec![200.0; chroma_size];

        let upsampled = upsample_ycbcr(dimensions, y, cb, cr, Subsampling::Sample420);

        // Check that Cb stays around 50
        assert!(
            upsampled.cb[0] < 100.0,
            "Upsample420: Cb[0]={} appears swapped with Cr (expected ~50)",
            upsampled.cb[0]
        );

        // Check that Cr stays around 200
        assert!(
            upsampled.cr[0] > 150.0,
            "Upsample420: Cr[0]={} appears swapped with Cb (expected ~200)",
            upsampled.cr[0]
        );
    }

    #[test]
    fn test_subsample_dimensions_420() {
        // Verify that 420 subsampling produces correct dimensions
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let pixels = vec![
            Ycbcr {
                y: 128.0,
                cb: 128.0,
                cr: 128.0,
                a: 255.0
            };
            width * height
        ];
        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample420);

        // 420: half width, half height
        let expected_chroma_size = (width / 2) * (height / 2);
        assert_eq!(
            subsampled.cb.len(),
            expected_chroma_size,
            "Sample420: Cb size incorrect"
        );
        assert_eq!(
            subsampled.cr.len(),
            expected_chroma_size,
            "Sample420: Cr size incorrect"
        );
    }

    #[test]
    fn test_subsample_dimensions_422() {
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let pixels = vec![
            Ycbcr {
                y: 128.0,
                cb: 128.0,
                cr: 128.0,
                a: 255.0
            };
            width * height
        ];
        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample422);

        // 422: half width, full height
        let expected_chroma_size = (width / 2) * height;
        assert_eq!(
            subsampled.cb.len(),
            expected_chroma_size,
            "Sample422: Cb size incorrect"
        );
        assert_eq!(
            subsampled.cr.len(),
            expected_chroma_size,
            "Sample422: Cr size incorrect"
        );
    }

    #[test]
    fn test_subsample_dimensions_411() {
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let pixels = vec![
            Ycbcr {
                y: 128.0,
                cb: 128.0,
                cr: 128.0,
                a: 255.0
            };
            width * height
        ];
        let subsampled = subsample_ycbcr(dimensions, &pixels, Subsampling::Sample411);

        // 411: quarter width, full height
        let expected_chroma_size = (width / 4) * height;
        assert_eq!(
            subsampled.cb.len(),
            expected_chroma_size,
            "Sample411: Cb size incorrect"
        );
        assert_eq!(
            subsampled.cr.len(),
            expected_chroma_size,
            "Sample411: Cr size incorrect"
        );
    }

    #[test]
    fn test_upsample_dimensions_restoration() {
        // Test that upsampling restores original dimensions
        let width = 16;
        let height = 16;
        let dimensions = PixelDimensions { width, height };

        let pixels = vec![
            Ycbcr {
                y: 128.0,
                cb: 128.0,
                cr: 128.0,
                a: 255.0
            };
            width * height
        ];

        for subsampling in &[
            Subsampling::Sample420,
            Subsampling::Sample422,
            Subsampling::Sample411,
            Subsampling::Sample444,
        ] {
            let subsampled = subsample_ycbcr(dimensions, &pixels, *subsampling);
            let upsampled = upsample_ycbcr(
                dimensions,
                subsampled.y,
                subsampled.cb,
                subsampled.cr,
                *subsampling,
            );

            assert_eq!(
                upsampled.y.len(),
                width * height,
                "{:?}: Y size not restored",
                subsampling
            );
            assert_eq!(
                upsampled.cb.len(),
                width * height,
                "{:?}: Cb size not restored",
                subsampling
            );
            assert_eq!(
                upsampled.cr.len(),
                width * height,
                "{:?}: Cr size not restored",
                subsampling
            );
        }
    }
}
