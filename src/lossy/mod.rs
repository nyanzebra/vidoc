use std::{ops::AddAssign, sync::Arc};

use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use rayon::{
    iter::{
        IndexedParallelIterator as _, IntoParallelIterator as _, IntoParallelRefIterator as _,
        ParallelIterator as _,
    },
    slice::{ParallelSlice as _, ParallelSliceMut as _},
};

use crate::{
    block::Block,
    clamp,
    color::{
        calculate_chroma_dimensions, subsample_ycbcr, upsample_ycbcr, Rgba, SubSampleGroup,
        Subsampling, UpSampleGroup, Ycbcr,
    },
    dimensions::{BlockDimensions, PixelDimensions},
};

pub mod frame;
pub mod jpg;

#[derive(Clone)]
pub struct SubSampleBlockGroup<T>(Arc<SubSampleBlockGroupInner<T>>);

impl<T> SubSampleBlockGroup<T> {
    pub fn new(
        dimensions: BlockDimensions,
        subsampling: Subsampling,
        y: Vec<Block<T>>,
        cb: Vec<Block<T>>,
        cr: Vec<Block<T>>,
    ) -> Self {
        Self(Arc::new(SubSampleBlockGroupInner {
            dimensions,
            subsampling,
            y,
            cb,
            cr,
        }))
    }
}

impl<T> SubSampleBlockGroup<T> {
    pub fn dimensions(&self) -> BlockDimensions {
        self.0.dimensions
    }

    pub fn subsampling(&self) -> Subsampling {
        self.0.subsampling
    }

    pub fn y(&self) -> &[Block<T>] {
        &self.0.y
    }

    pub fn cb(&self) -> &[Block<T>] {
        &self.0.cb
    }

    pub fn cr(&self) -> &[Block<T>] {
        &self.0.cr
    }
}

impl<T> SubSampleBlockGroup<T> {
    pub fn as_ref(&self) -> SubSampleBlockGroupRef<'_, T> {
        SubSampleBlockGroupRef {
            dimensions: self.0.dimensions,
            subsampling: self.0.subsampling,
            y: &self.0.y,
            cb: &self.0.cb,
            cr: &self.0.cr,
        }
    }
}

struct SubSampleBlockGroupInner<T> {
    dimensions: BlockDimensions,
    subsampling: Subsampling,
    y: Vec<Block<T>>,
    cb: Vec<Block<T>>,
    cr: Vec<Block<T>>,
}

impl From<SubSampleBlockGroupInner<f32>> for SubSampleBlockGroupInner<i16> {
    fn from(value: SubSampleBlockGroupInner<f32>) -> Self {
        SubSampleBlockGroupInner {
            dimensions: value.dimensions,
            subsampling: value.subsampling,
            y: value.y.into_par_iter().map(|block| block.into()).collect(),
            cb: value.cb.into_par_iter().map(|block| block.into()).collect(),
            cr: value.cr.into_par_iter().map(|block| block.into()).collect(),
        }
    }
}

impl From<SubSampleBlockGroupInner<f32>> for SubSampleBlockGroupInner<i32> {
    fn from(value: SubSampleBlockGroupInner<f32>) -> Self {
        SubSampleBlockGroupInner {
            dimensions: value.dimensions,
            subsampling: value.subsampling,
            y: value.y.into_par_iter().map(|block| block.into()).collect(),
            cb: value.cb.into_par_iter().map(|block| block.into()).collect(),
            cr: value.cr.into_par_iter().map(|block| block.into()).collect(),
        }
    }
}

impl From<SubSampleBlockGroupInner<i16>> for SubSampleBlockGroupInner<f32> {
    fn from(value: SubSampleBlockGroupInner<i16>) -> Self {
        SubSampleBlockGroupInner {
            dimensions: value.dimensions,
            subsampling: value.subsampling,
            y: value.y.into_par_iter().map(|block| block.into()).collect(),
            cb: value.cb.into_par_iter().map(|block| block.into()).collect(),
            cr: value.cr.into_par_iter().map(|block| block.into()).collect(),
        }
    }
}

impl From<SubSampleBlockGroupInner<i32>> for SubSampleBlockGroupInner<f32> {
    fn from(value: SubSampleBlockGroupInner<i32>) -> Self {
        SubSampleBlockGroupInner {
            dimensions: value.dimensions,
            subsampling: value.subsampling,
            y: value.y.into_par_iter().map(|block| block.into()).collect(),
            cb: value.cb.into_par_iter().map(|block| block.into()).collect(),
            cr: value.cr.into_par_iter().map(|block| block.into()).collect(),
        }
    }
}

impl<T> SubSampleBlockGroup<T>
where
    T: Signed + Default + AddAssign + Copy + ToPrimitive,
{
    /// Calculates the sum of absolute difference on the y-channel.
    pub(crate) fn sum_of_abs_difference(&self, other: SubSampleBlockGroup<T>) -> i64 {
        self.as_ref().sum_of_abs_difference(other.as_ref())
    }
}

impl From<SubSampleBlockGroup<f32>> for SubSampleBlockGroup<i16> {
    fn from(value: SubSampleBlockGroup<f32>) -> Self {
        SubSampleBlockGroup(Arc::new(SubSampleBlockGroupInner {
            dimensions: value.0.dimensions,
            subsampling: value.0.subsampling,
            y: value
                .y()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cb: value
                .cb()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cr: value
                .cr()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
        }))
    }
}

impl From<SubSampleBlockGroup<f32>> for SubSampleBlockGroup<i32> {
    fn from(value: SubSampleBlockGroup<f32>) -> Self {
        SubSampleBlockGroup(Arc::new(SubSampleBlockGroupInner {
            dimensions: value.0.dimensions,
            subsampling: value.0.subsampling,
            y: value
                .y()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cb: value
                .cb()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cr: value
                .cr()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
        }))
    }
}

impl From<SubSampleBlockGroup<i16>> for SubSampleBlockGroup<f32> {
    fn from(value: SubSampleBlockGroup<i16>) -> Self {
        SubSampleBlockGroup(Arc::new(SubSampleBlockGroupInner {
            dimensions: value.0.dimensions,
            subsampling: value.0.subsampling,
            y: value
                .y()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cb: value
                .cb()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cr: value
                .cr()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
        }))
    }
}

impl From<SubSampleBlockGroup<i32>> for SubSampleBlockGroup<f32> {
    fn from(value: SubSampleBlockGroup<i32>) -> Self {
        SubSampleBlockGroup(Arc::new(SubSampleBlockGroupInner {
            dimensions: value.0.dimensions,
            subsampling: value.0.subsampling,
            y: value
                .y()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cb: value
                .cb()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
            cr: value
                .cr()
                .into_par_iter()
                .map(|block| (*block).into())
                .collect(),
        }))
    }
}

pub struct SubSampleBlockGroupRef<'a, T> {
    pub dimensions: BlockDimensions,
    pub subsampling: Subsampling,
    pub y: &'a [Block<T>],
    pub cb: &'a [Block<T>],
    pub cr: &'a [Block<T>],
}

impl<T> SubSampleBlockGroupRef<'_, T>
where
    T: Signed + Default + AddAssign + Copy + ToPrimitive,
{
    /// Calculates the sum of absolute difference on the y-channel.
    pub(crate) fn sum_of_abs_difference(&self, other: SubSampleBlockGroupRef<'_, T>) -> i64 {
        let mut sad = 0;

        for y in 0..self.y.len() {
            sad += self.y[y]
                .sum_of_abs_difference(&other.y[y])
                .to_i64()
                .unwrap_or(0);
        }

        sad
    }
}

// https://en.wikipedia.org/wiki/Chroma_subsampling
pub fn subsample_into_block_ycbcr(
    dimensions: PixelDimensions,
    ycbcr: &[Ycbcr],
    subsampling: Subsampling,
) -> SubSampleBlockGroup<f32> {
    let SubSampleGroup {
        dimensions,
        y,
        cb,
        cr,
    } = subsample_ycbcr(dimensions, ycbcr, subsampling);

    // Calculate the actual pixel dimensions for chroma channels after subsampling
    let PixelDimensions {
        width: chroma_width,
        height: chroma_height,
    } = dimensions.subsample(subsampling);

    let PixelDimensions { width, height } = dimensions;

    // Create Y blocks using original dimensions (parallel)
    let y_positions: Vec<(usize, usize)> = (0..height)
        .step_by(Block::<f32>::rows())
        .flat_map(|r| {
            (0..width)
                .step_by(Block::<f32>::cols())
                .map(move |c| (r, c))
        })
        .collect();

    let y_blocks: Vec<Block<f32>> = y_positions
        .par_iter()
        .map(|&(r, c)| build_block(&y, r, c, width))
        .collect();

    // Create Cb blocks using chroma pixel dimensions (parallel)
    let cb_positions: Vec<(usize, usize)> = (0..chroma_height)
        .step_by(Block::<f32>::rows())
        .flat_map(|r| {
            (0..chroma_width)
                .step_by(Block::<f32>::cols())
                .map(move |c| (r, c))
        })
        .collect();

    let cb_blocks: Vec<Block<f32>> = cb_positions
        .par_iter()
        .map(|&(r, c)| build_block(&cb, r, c, chroma_width))
        .collect();

    // Create Cr blocks using chroma pixel dimensions (parallel)
    let cr_positions: Vec<(usize, usize)> = (0..chroma_height)
        .step_by(Block::<f32>::rows())
        .flat_map(|r| {
            (0..chroma_width)
                .step_by(Block::<f32>::cols())
                .map(move |c| (r, c))
        })
        .collect();

    let cr_blocks: Vec<Block<f32>> = cr_positions
        .par_iter()
        .map(|&(r, c)| build_block(&cr, r, c, chroma_width))
        .collect();

    SubSampleBlockGroup(Arc::new(SubSampleBlockGroupInner {
        dimensions: dimensions.into(),
        subsampling,
        y: y_blocks,
        cb: cb_blocks,
        cr: cr_blocks,
    }))
}

#[inline]
pub(crate) fn build_block<T>(pixels: &[T], x: usize, y: usize, width: usize) -> Block<T>
where
    T: Copy + Default,
{
    let x_start = x;
    let x_end = x_start + Block::<T>::rows();
    let y_start = y;
    let y_end = y_start + Block::<T>::cols();

    let mut block = Block::<T>::default();
    for x in x_start..x_end {
        for y in y_start..y_end {
            let r = x - x_start;
            let c = y - y_start;
            let pixel_index = x * width + y;
            if pixel_index < pixels.len() {
                let idx = r * 8 + c;
                block[idx] = pixels[pixel_index];
            }
        }
    }

    block
}

#[inline]
pub fn reconstruct_pixels<T>(
    dimensions: PixelDimensions,
    y_blocks: &[Block<f32>],
    cb_blocks: &[Block<f32>],
    cr_blocks: &[Block<f32>],
    _a_blocks: Option<&[Block<f32>]>,
    subsampling: Subsampling,
) -> Vec<T>
where
    T: Bounded + FromPrimitive + ToPrimitive + Send + Sync,
{
    let PixelDimensions { width, height } = dimensions;
    let mut ys = vec![0.0; height * width];
    let mut _alphas = vec![0.0; height * width];

    // Use the canonical chroma dimension calculation from color module
    // This ensures consistency between encoding and decoding paths
    let chroma_dims = calculate_chroma_dimensions(dimensions, subsampling);
    let chroma_width = chroma_dims.width;
    let chroma_height = chroma_dims.height;

    // Initialize chroma arrays with actual dimensions needed for blocks
    let mut cbs = vec![0.0; chroma_height * chroma_width];
    let mut crs = vec![0.0; chroma_height * chroma_width];

    // Reconstruct Y blocks in parallel
    {
        let blocks_per_row = width.div_ceil(Block::<T>::cols());
        ys.par_chunks_mut(width * Block::<T>::rows())
            .enumerate()
            .for_each(|(block_row, row_chunk)| {
                let r = block_row * Block::<T>::rows();
                for block_col in 0..blocks_per_row {
                    let block_idx = block_row * blocks_per_row + block_col;
                    if block_idx < y_blocks.len() {
                        let c = block_col * Block::<T>::cols();
                        let y_block = &y_blocks[block_idx];
                        break_block(row_chunk, y_block, r, c, width);
                    }
                }
            });
    }

    // Reconstruct Cb blocks in parallel
    {
        let blocks_per_row = chroma_width.div_ceil(Block::<T>::cols());
        cbs.par_chunks_mut(chroma_width * Block::<T>::rows())
            .enumerate()
            .for_each(|(block_row, row_chunk)| {
                let r = block_row * Block::<T>::rows();
                for block_col in 0..blocks_per_row {
                    let block_idx = block_row * blocks_per_row + block_col;
                    if block_idx < cb_blocks.len() {
                        let c = block_col * Block::<T>::cols();
                        let cb_block = &cb_blocks[block_idx];

                        break_block(row_chunk, cb_block, r, c, chroma_width);
                    }
                }
            });
    }

    // Reconstruct Cr blocks in parallel
    {
        let blocks_per_row = chroma_width.div_ceil(Block::<T>::cols());
        crs.par_chunks_mut(chroma_width * Block::<T>::rows())
            .enumerate()
            .for_each(|(block_row, row_chunk)| {
                let r = block_row * Block::<T>::rows();
                for block_col in 0..blocks_per_row {
                    let block_idx = block_row * blocks_per_row + block_col;
                    if block_idx < cr_blocks.len() {
                        let c = block_col * Block::<T>::cols();
                        let cr_block = &cr_blocks[block_idx];

                        break_block(row_chunk, cr_block, r, c, chroma_width);
                    }
                }
            });
    }

    let UpSampleGroup {
        dimensions: _upsampled_dims,
        y,
        cb,
        cr,
    } = upsample_ycbcr(dimensions, ys, cbs, crs, subsampling);

    // SIMD-accelerated color conversion from YCbCr to RGB
    // Uses chunked processing with SIMD when available for better performance
    use crate::color::ycbcr_batch_to_rgba;

    // Process in parallel chunks for better cache locality and SIMD utilization
    let chunk_size = 1024; // Process 1K pixels at a time

    y.par_chunks(chunk_size)
        .zip(cb.par_chunks(chunk_size))
        .zip(cr.par_chunks(chunk_size))
        .flat_map(|((y_chunk, cb_chunk), cr_chunk)| {
            // Use SIMD batch conversion for each chunk
            let rgba_pixels = ycbcr_batch_to_rgba(y_chunk, cb_chunk, cr_chunk);

            // Convert to flat RGB array with correct type
            rgba_pixels
                .into_iter()
                .flat_map(|Rgba { r, g, b, a: _ }| {
                    [
                        T::from_u8(r).unwrap_or_else(|| T::min_value()),
                        T::from_u8(g).unwrap_or_else(|| T::min_value()),
                        T::from_u8(b).unwrap_or_else(|| T::min_value()),
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

#[inline]
fn break_block<T>(
    chunk: &mut [T],
    block: &Block<f32>,
    block_row: usize,
    block_col: usize,
    full_width: usize,
) where
    T: Bounded + FromPrimitive + ToPrimitive + Send + Sync,
{
    let chunk_start_row = (block_row / Block::<f32>::rows()) * Block::<f32>::rows();

    for block_r in 0..Block::<f32>::rows() {
        for block_c in 0..Block::<f32>::cols() {
            let pixel_row = block_row + block_r;
            let pixel_col = block_col + block_c;

            // Calculate position within the chunk
            let local_row = pixel_row - chunk_start_row;
            let chunk_idx = local_row * full_width + pixel_col;

            if chunk_idx < chunk.len() {
                let block_idx = block_r * 8 + block_c;
                chunk[chunk_idx] = clamp(block[block_idx]);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{
        block::Block,
        color::Subsampling,
        dimensions::BlockDimensions,
        lossy::{SubSampleBlockGroup, SubSampleBlockGroupRef},
    };

    #[derive(Debug, Clone)]
    pub(crate) struct TestSubSampleBlockGroup<T> {
        pub dimensions: BlockDimensions,
        pub subsampling: Subsampling,
        pub y: Vec<Block<T>>,
        pub cb: Vec<Block<T>>,
        pub cr: Vec<Block<T>>,
    }

    impl<T> TestSubSampleBlockGroup<T>
    where
        T: Copy + Default,
    {
        pub(crate) fn test_frame(width: usize, height: usize, fill: T) -> Self {
            let block_dims = BlockDimensions { width, height };

            let num_blocks = width * height;
            let mut blocks = Vec::with_capacity(num_blocks);

            for _ in 0..num_blocks {
                let mut block = Block::<T>::default();
                for r in 0..8 {
                    for c in 0..8 {
                        block.set(r, c, fill);
                    }
                }
                blocks.push(block);
            }

            // For 4:2:0, chroma is half resolution in both dimensions
            let chroma_blocks = (width / 2) * (height / 2);
            let mut cb_blocks = Vec::with_capacity(chroma_blocks);
            let mut cr_blocks = Vec::with_capacity(chroma_blocks);

            for _ in 0..chroma_blocks {
                cb_blocks.push(Block::<T>::default());
                cr_blocks.push(Block::<T>::default());
            }

            Self {
                dimensions: block_dims,
                subsampling: Subsampling::Sample420,
                y: blocks,
                cb: cb_blocks,
                cr: cr_blocks,
            }
        }
    }

    impl<T> TestSubSampleBlockGroup<T> {
        pub(crate) fn as_ref(&self) -> SubSampleBlockGroupRef<'_, T> {
            SubSampleBlockGroupRef {
                dimensions: self.dimensions,
                subsampling: self.subsampling,
                y: &self.y,
                cb: &self.cb,
                cr: &self.cr,
            }
        }
    }

    impl<T> From<TestSubSampleBlockGroup<T>> for SubSampleBlockGroup<T> {
        fn from(value: TestSubSampleBlockGroup<T>) -> Self {
            SubSampleBlockGroup::new(
                value.dimensions,
                value.subsampling,
                value.y,
                value.cb,
                value.cr,
            )
        }
    }
}
