use std::{
    cmp::{max, PartialOrd},
    ops::{AddAssign, DivAssign, MulAssign},
    sync::OnceLock,
};

use crate::{
    block::{quantization::Quantizor, Block, Blocks, REASONABLE_SUM_OF_ABS_DIFF_I16},
    color::Subsampling,
    dimensions::BlockDimensions,
    lossy::SubSampleBlockGroupRef,
    point::Point,
};

pub mod bframe;
pub mod gop;
pub mod iframe;

pub mod r#macro;
use num_traits::{FromPrimitive, NumCast, Signed, ToPrimitive};
use r#macro::{BlockLocation, IMacroBlock, Prediction, Residuals};

mod motion_vector;
pub(crate) use motion_vector::MotionVector;

pub mod pframe;

type PredictedBlocks = (Vec<Block<i16>>, Vec<Block<i16>>, Vec<Block<i16>>);

pub const FRAME_CODE_I: u8 = 0b01;
pub const FRAME_CODE_P: u8 = 0b10;
pub const FRAME_CODE_B: u8 = 0b11;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    B,
    I,
    P,
}

impl From<u8> for Kind {
    fn from(value: u8) -> Self {
        match value {
            FRAME_CODE_I => Kind::I,
            FRAME_CODE_P => Kind::P,
            FRAME_CODE_B => Kind::B,
            _ => panic!("invalid frame code: '{value}'"),
        }
    }
}

impl From<Kind> for u8 {
    fn from(value: Kind) -> Self {
        match value {
            Kind::B => FRAME_CODE_B,
            Kind::I => FRAME_CODE_I,
            Kind::P => FRAME_CODE_P,
        }
    }
}

/// Block of 4
/// ```ignore
/// _ x
/// x x
/// ```
const ONE_OUT: [(usize, usize); 3] = [(0, 1), (1, 1), (1, 0)];
/// Block of 9
/// ```ignore
/// _ _ x
/// _ _ x
/// x x x
/// ```
const TWO_OUT: [(usize, usize); 5] = [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)];
/// Block of 16
/// ```ignore
/// _ _ _ x
/// _ _ _ x
/// _ _ _ x
/// x x x x
/// ```
const THREE_OUT: [(usize, usize); 7] = [(0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0)];

pub(crate) fn build_macro_blocks<T>(
    blocks: &[Block<T>],
    dimensions: BlockDimensions,
) -> Vec<IMacroBlock<T>>
where
    T: Signed
        + Default
        + AddAssign
        + DivAssign
        + MulAssign
        + Copy
        + NumCast
        + FromPrimitive
        + PartialOrd
        + ToPrimitive
        + 'static,
{
    if blocks.is_empty() {
        return vec![];
    }

    let BlockDimensions { width, height } = dimensions;
    let mut used_table = vec![vec![false; width]; height];

    let mut res = vec![];

    for r in 0..height {
        for c in 0..width {
            if used_table[r][c] {
                continue;
            }

            let idx = r * width + c;
            if idx >= blocks.len() {
                continue;
            }
            let block = blocks[idx];
            let mut current_macro: IMacroBlock<T> = IMacroBlock {
                location: BlockLocation {
                    start: Point { row: r, col: c },
                    end: Point { row: r, col: c },
                },
                blocks: vec![block],
            };

            let new_end = expand_macro_block(
                blocks,
                dimensions,
                current_macro.location,
                &mut used_table,
                &mut current_macro.blocks,
                &ONE_OUT,
            );

            if current_macro.location.end != new_end {
                assert_eq!(current_macro.blocks.len(), ONE_OUT.len() + 1);
                current_macro.location.end = new_end;

                let new_end = expand_macro_block(
                    blocks,
                    dimensions,
                    current_macro.location,
                    &mut used_table,
                    &mut current_macro.blocks,
                    &TWO_OUT,
                );

                if current_macro.location.end != new_end {
                    current_macro.location.end = new_end;
                    assert_eq!(
                        current_macro.blocks.len(),
                        ONE_OUT.len() + TWO_OUT.len() + 1
                    );

                    let new_end = expand_macro_block(
                        blocks,
                        dimensions,
                        current_macro.location,
                        &mut used_table,
                        &mut current_macro.blocks,
                        &THREE_OUT,
                    );

                    if current_macro.location.end != new_end {
                        current_macro.location.end = new_end;
                        assert_eq!(
                            current_macro.blocks.len(),
                            ONE_OUT.len() + TWO_OUT.len() + THREE_OUT.len() + 1
                        );
                    }
                }
            }

            // Mark the starting block as used
            used_table[r][c] = true;
            res.push(current_macro);
        }
    }

    res
}

fn expand_macro_block<T>(
    blocks: &[Block<T>],
    dimensions: BlockDimensions,
    location: BlockLocation,
    used_table: &mut [Vec<bool>],
    macro_blocks: &mut Vec<Block<T>>,
    expansion: &[(usize, usize)],
) -> Point
where
    T: Signed + Default + AddAssign + Copy + FromPrimitive + PartialOrd,
{
    let BlockDimensions { width, height } = dimensions;
    let mut end = location.end;
    let mut others = vec![];
    for (r1, c1) in expansion {
        let r = location.start.row + r1;
        let c = location.start.col + c1;
        if r >= height || c >= width {
            break;
        }

        end.row = max(end.row, r);
        end.col = max(end.col, c);

        let idx = r * width + c;
        if idx >= blocks.len() {
            break;
        }
        let other = blocks[idx];
        let soad = macro_blocks[0].sum_of_abs_difference(&other);
        if soad <= T::from_i16(REASONABLE_SUM_OF_ABS_DIFF_I16).expect("i16->T") {
            others.push((other, (r, c)));
        }
    }

    if others.len() == expansion.len() {
        for (other, (r, c)) in others {
            macro_blocks.push(other);
            used_table[r][c] = true;
        }
    } else {
        end = location.end;
    }

    end
}

fn reconstruct_blocks_from_macroblock<T>(
    mb: &IMacroBlock<T>,
    blocks_array: &mut [Block<f64>],
    grid_width: usize,
) where
    T: Copy + ToPrimitive + 'static,
{
    if mb.blocks.is_empty() {
        return;
    }

    // First block is always at the start position
    let start_idx = mb.location.start.row * grid_width + mb.location.start.col;
    if start_idx < blocks_array.len() {
        blocks_array[start_idx] = mb.blocks[0].convert_to();
    }

    // Reconstruct additional blocks following expansion patterns
    let mut block_idx = 1;

    // Check if we used ONE_OUT expansion (4 blocks total)
    if mb.blocks.len() > ONE_OUT.len() {
        for (r_offset, c_offset) in ONE_OUT.iter() {
            if block_idx < mb.blocks.len() {
                let r = mb.location.start.row + r_offset;
                let c = mb.location.start.col + c_offset;
                let idx = r * grid_width + c;
                if idx < blocks_array.len() {
                    blocks_array[idx] = mb.blocks[block_idx].convert_to();
                }
                block_idx += 1;
            }
        }
    }

    // Check if we used TWO_OUT expansion (9 blocks total)
    if mb.blocks.len() > ONE_OUT.len() + TWO_OUT.len() {
        for (r_offset, c_offset) in TWO_OUT.iter() {
            if block_idx < mb.blocks.len() {
                let r = mb.location.start.row + r_offset;
                let c = mb.location.start.col + c_offset;
                let idx = r * grid_width + c;
                if idx < blocks_array.len() {
                    blocks_array[idx] = mb.blocks[block_idx].convert_to();
                }
                block_idx += 1;
            }
        }
    }

    // Check if we used THREE_OUT expansion (16 blocks total)
    if mb.blocks.len() > ONE_OUT.len() + TWO_OUT.len() + THREE_OUT.len() {
        for (r_offset, c_offset) in THREE_OUT.iter() {
            if block_idx < mb.blocks.len() {
                let r = mb.location.start.row + r_offset;
                let c = mb.location.start.col + c_offset;
                let idx = r * grid_width + c;
                if idx < blocks_array.len() {
                    blocks_array[idx] = mb.blocks[block_idx].convert_to();
                }
                block_idx += 1;
            }
        }
    }
}

/// Build predicted blocks for all channels (Y, Cb, Cr) based on prediction mode.
///
/// This is a common helper used by both P-frames and B-frames.
/// - P-frames use Prediction::Backward with backward_ref as the previous frame
/// - B-frames use Prediction::Forward, Prediction::Backward, or Prediction::Both
///
/// Returns (predicted_y, predicted_cb, predicted_cr) vectors of blocks.
pub(crate) fn build_predicted_blocks(
    location: &BlockLocation,
    prediction: &Prediction,
    dimensions: &BlockDimensions,
    subsampling: Subsampling,
    forward_ref: Option<&SubSampleBlockGroupRef<i16>>,
    backward_ref: &SubSampleBlockGroupRef<i16>,
) -> PredictedBlocks {
    let BlockLocation { start, end } = location;
    let luma_count = (end.row - start.row + 1) * (end.col - start.col + 1);
    let mut predicted_y = Vec::with_capacity(luma_count);

    // Extract Y channel slices for easier access
    let forward_y = forward_ref.map(|f| f.y);
    let backward_y = backward_ref.y;

    // Build predicted Y blocks using the helper function
    for r in start.row..=end.row {
        for c in start.col..=end.col {
            let predicted_block =
                predicted_block((r, c), *prediction, *dimensions, backward_y, forward_y);
            predicted_y.push(predicted_block);
        }
    }

    // Build predicted chroma blocks
    let chroma_dims = dimensions.subsample(subsampling);

    // Pre-size: chroma blocks are at most (luma_blocks / chroma_ratio)
    let luma_blocks = (end.row - start.row + 1) * (end.col - start.col + 1);
    let mut predicted_cb = Vec::with_capacity(luma_blocks / 4 + 1);
    let mut predicted_cr = Vec::with_capacity(luma_blocks / 4 + 1);

    // Use a small stack-allocated bitset instead of HashSet — avoids heap allocation.
    // Macroblocks span at most a few rows/cols so a fixed 32x32 bool array is safe.
    let chroma_start_r = start.row / 2;
    let chroma_start_c = start.col / 2;
    // Stack-allocate a small processed table (max macroblock spans a handful of chroma blocks)
    let mut processed_chroma_blocks = [[false; 32]; 32];

    for r in start.row..=end.row {
        for c in start.col..=end.col {
            let chroma_r = match subsampling {
                Subsampling::Sample420 | Subsampling::Sample411 => r / 2,
                _ => r,
            };
            let chroma_c = match subsampling {
                Subsampling::Sample420 | Subsampling::Sample422 => c / 2,
                Subsampling::Sample411 => c / 4,
                _ => c,
            };

            let local_r = chroma_r.saturating_sub(chroma_start_r).min(31);
            let local_c = chroma_c.saturating_sub(chroma_start_c).min(31);
            if processed_chroma_blocks[local_r][local_c] {
                continue;
            }
            processed_chroma_blocks[local_r][local_c] = true;

            // Use the helper to predict chroma blocks
            let pred_cb = predicted_chroma_block(
                (chroma_r, chroma_c),
                *prediction,
                chroma_dims,
                subsampling,
                backward_ref.cb,
                forward_ref.map(|f| f.cb),
            );

            let pred_cr = predicted_chroma_block(
                (chroma_r, chroma_c),
                *prediction,
                chroma_dims,
                subsampling,
                backward_ref.cr,
                forward_ref.map(|f| f.cr),
            );

            predicted_cb.push(pred_cb);
            predicted_cr.push(pred_cr);
        }
    }

    (predicted_y, predicted_cb, predicted_cr)
}

/// Calculate residuals for all channels (Y, Cb, Cr) given current and predicted blocks.
///
/// This is a common helper used by both P-frames and B-frames.
/// Returns Residuals struct containing y, cb, and cr residuals after DCT and quantization.
pub(crate) fn calculate_residuals_for_macroblock(
    location: &BlockLocation,
    current: &SubSampleBlockGroupRef<i16>,
    predicted_y: &[Block<i16>],
    predicted_cb: &[Block<i16>],
    predicted_cr: &[Block<i16>],
) -> Residuals<i16> {
    let BlockLocation { start, end } = location;
    // These are the same tables every call — construct once via OnceLock.
    // Note: f64 Quantizor can't use the static_* helpers (those are i16/i32 only)
    // so we use a local OnceLock here.
    static Q_Y: OnceLock<Quantizor<f64>> = OnceLock::new();
    static Q_C: OnceLock<Quantizor<f64>> = OnceLock::new();
    let quantizor_y = Q_Y.get_or_init(Quantizor::<f64>::video_luminance);
    let quantizor_chroma = Q_C.get_or_init(Quantizor::<f64>::video_chrominance);

    let luma_count = (end.row - start.row + 1) * (end.col - start.col + 1);
    let mut y_residuals = Vec::with_capacity(luma_count);

    // Calculate luma residuals — single pass, pre-sized
    let mut pred_idx = 0;
    for r in start.row..=end.row {
        for c in start.col..=end.col {
            let idx = r * current.dimensions.width + c;

            let residual_spatial = if idx < current.y.len() && pred_idx < predicted_y.len() {
                // Use Block subtraction: current - predicted
                (current.y[idx] - predicted_y[pred_idx]).convert_to()
            } else {
                Block::<f64>::default()
            };

            // Apply DCT and quantization
            let residual_dct = residual_spatial.dct();
            let residual_quantized = quantizor_y.quantize(residual_dct);
            let residual_i16: Block<i16> = residual_quantized.convert_to();
            y_residuals.push(residual_i16);
            pred_idx += 1;
        }
    }

    // Calculate chroma residuals
    let chroma_dims = current.dimensions.subsample(current.subsampling);
    let chroma_width = chroma_dims.width;
    let chroma_height = chroma_dims.height;

    let mut cb_residuals = Vec::new();
    let mut cr_residuals = Vec::new();

    // Track which chroma blocks we've already processed to avoid duplicates
    // Use a 2D bool array that maps to the chroma space
    let mut processed_chroma_blocks = vec![vec![false; chroma_width]; chroma_height];
    let mut pred_chroma_idx = 0;

    for r in start.row..=end.row {
        for c in start.col..=end.col {
            // Map luma position to chroma position
            let chroma_r = match current.subsampling {
                Subsampling::Sample420 | Subsampling::Sample411 => r / 2,
                _ => r,
            };
            let chroma_c = match current.subsampling {
                Subsampling::Sample420 | Subsampling::Sample422 => c / 2,
                Subsampling::Sample411 => c / 4,
                _ => c,
            };

            // Skip if we've already processed this chroma block
            if chroma_r >= chroma_height || chroma_c >= chroma_width {
                continue;
            }
            if processed_chroma_blocks[chroma_r][chroma_c] {
                continue;
            }
            processed_chroma_blocks[chroma_r][chroma_c] = true;

            let chroma_idx = chroma_r * chroma_width + chroma_c;

            // Calculate Cb residual using Block subtraction
            let cb_residual_spatial =
                if chroma_idx < current.cb.len() && pred_chroma_idx < predicted_cb.len() {
                    (current.cb[chroma_idx] - predicted_cb[pred_chroma_idx]).convert_to()
                } else {
                    Block::<f64>::default()
                };
            let cb_dct = cb_residual_spatial.dct();
            let cb_quantized = quantizor_chroma.quantize(cb_dct);
            cb_residuals.push(cb_quantized.convert_to());

            // Calculate Cr residual using Block subtraction
            let cr_residual_spatial =
                if chroma_idx < current.cr.len() && pred_chroma_idx < predicted_cr.len() {
                    (current.cr[chroma_idx] - predicted_cr[pred_chroma_idx]).convert_to()
                } else {
                    Block::<f64>::default()
                };
            let cr_dct = cr_residual_spatial.dct();
            let cr_quantized = quantizor_chroma.quantize(cr_dct);
            cr_residuals.push(cr_quantized.convert_to());

            pred_chroma_idx += 1;
        }
    }

    Residuals {
        y: Blocks::new(y_residuals),
        cb: Blocks::new(cb_residuals),
        cr: Blocks::new(cr_residuals),
    }
}

pub(crate) fn try_compress_motion_vectors(
    dimensions: &BlockDimensions,
    mvs: &[Vec<(Prediction, i16)>],
    (row, col): (usize, usize),
    pattern: &[(usize, usize)],
) -> BlockLocation {
    let current_mv = mvs[row][col].0;
    let mut max_row = row;
    let mut max_col = col;

    for &(dr, dc) in pattern {
        let target_row = row + dr;
        let target_col = col + dc;

        // Check bounds against both dimensions AND actual mvs vector size
        if target_row < dimensions.height
            && target_col < dimensions.width
            && target_row < mvs.len()
            && target_col < mvs[target_row].len()
        {
            if mvs[target_row][target_col].0 != current_mv {
                return BlockLocation {
                    start: Point { row, col },
                    end: Point { row, col },
                };
            }
            max_row = max_row.max(target_row);
            max_col = max_col.max(target_col);
        }
    }

    BlockLocation {
        start: Point { row, col },
        end: Point {
            row: max_row,
            col: max_col,
        },
    }
}

pub(crate) fn reassemble_frame<MB: r#macro::AssemblableMacroBlock>(
    forward_ref: Option<&SubSampleBlockGroupRef<'_, i16>>,
    backward_ref: &SubSampleBlockGroupRef<'_, i16>,
    macro_blocks: &[MB],
) -> crate::Result<crate::lossy::SubSampleBlockGroup<i16>> {
    use crate::lossy::SubSampleBlockGroup;

    // Use backward ref as base if no forward ref
    let base_ref = forward_ref.unwrap_or(backward_ref);

    let SubSampleBlockGroupRef {
        dimensions,
        subsampling,
        y: base_y,
        cb: base_cb,
        cr: base_cr,
    } = base_ref;

    let backward_y = &backward_ref.y;

    let BlockDimensions { width, height: _ } = *dimensions;

    // Calculate chroma dimensions based on subsampling
    let chroma_dims = dimensions.subsample(*subsampling);
    let chroma_width = chroma_dims.width;

    // Use forward ref if available, otherwise fall back to backward ref for Y channel
    let forward_y = forward_ref.map(|f| f.y).unwrap_or(backward_y);

    // Start with a copy of the base reference frame
    let mut y_blocks: Vec<Block<i16>> = base_y.to_vec();
    let mut cb_blocks: Vec<Block<i16>> = base_cb.to_vec();
    let mut cr_blocks: Vec<Block<i16>> = base_cr.to_vec();

    // Create quantizors for dequantizing residuals — cached via OnceLock
    static Q_Y_R: OnceLock<Quantizor<f64>> = OnceLock::new();
    static Q_C_R: OnceLock<Quantizor<f64>> = OnceLock::new();
    let quantizor = Q_Y_R.get_or_init(Quantizor::<f64>::video_luminance);
    let chroma_quantizor = Q_C_R.get_or_init(Quantizor::<f64>::video_chrominance);

    for mb in macro_blocks.iter() {
        let BlockLocation { start, end } = mb.location();
        let prediction = mb.prediction();
        let residuals = mb.residuals();
        let mut y_residual_idx = 0;

        for row in start.row..=end.row {
            for col in start.col..=end.col {
                let idx = row * width + col;
                // Apply Y residual
                if y_residual_idx < residuals.y.len() && idx < y_blocks.len() {
                    let residual_quantized = &residuals.y[y_residual_idx];

                    y_blocks[idx] = (predicted_block(
                        (row, col),
                        prediction,
                        *dimensions,
                        backward_y,
                        Some(forward_y),
                    )
                    .convert_to()
                        + quantizor.dequantize(residual_quantized.convert_to()).idct())
                    .clamp(i16::MIN as f64, i16::MAX as f64)
                    .convert_to();
                }
                y_residual_idx += 1;
            }
        }
    }

    for mb in macro_blocks.iter() {
        let location = mb.location();
        let prediction = mb.prediction();
        let residuals = mb.residuals();

        // Map luma block range to chroma block range
        let chroma_location = location.map_to_chroma(*subsampling);

        let mut chroma_residual_idx = 0;

        for row in chroma_location.start.row..=chroma_location.end.row {
            for col in chroma_location.start.col..=chroma_location.end.col {
                let idx = row * chroma_width + col;

                if idx >= cb_blocks.len() {
                    continue;
                }

                // Use helper to get motion-compensated chroma predictions
                let mut predicted_cb = predicted_chroma_block(
                    (row, col),
                    prediction,
                    chroma_dims,
                    *subsampling,
                    backward_ref.cb,
                    forward_ref.map(|f| f.cb),
                );

                let mut predicted_cr = predicted_chroma_block(
                    (row, col),
                    prediction,
                    chroma_dims,
                    *subsampling,
                    backward_ref.cr,
                    forward_ref.map(|f| f.cr),
                );

                // Apply chroma residuals if available
                if chroma_residual_idx < residuals.cb.len()
                    && chroma_residual_idx < residuals.cr.len()
                {
                    let cb_residual_quantized = &residuals.cb[chroma_residual_idx];
                    let cr_residual_quantized = &residuals.cr[chroma_residual_idx];

                    predicted_cb = (predicted_cb.convert_to()
                        + chroma_quantizor
                            .dequantize(cb_residual_quantized.convert_to())
                            .idct())
                    .clamp(i16::MIN as f64, i16::MAX as f64)
                    .convert_to();
                    predicted_cr = (predicted_cr.convert_to()
                        + chroma_quantizor
                            .dequantize(cr_residual_quantized.convert_to())
                            .idct())
                    .clamp(i16::MIN as f64, i16::MAX as f64)
                    .convert_to();

                    chroma_residual_idx += 1;
                }

                // Update chroma blocks
                cb_blocks[idx] = predicted_cb;
                cr_blocks[idx] = predicted_cr;
            }
        }
    }

    Ok(SubSampleBlockGroup::new(
        *dimensions,
        *subsampling,
        y_blocks,
        cb_blocks,
        cr_blocks,
    ))
}

pub(crate) fn compressed_motion_vectors(
    mvs: &[Vec<(Prediction, i16)>],
    dimensions: &BlockDimensions,
) -> Vec<((Prediction, i16), BlockLocation)> {
    let mut mv_locations = vec![];
    let mut used_table = UsedTable::new(mvs.len(), mvs[0].len());

    for row in 0..mvs.len() {
        for col in 0..mvs[0].len() {
            if used_table.is_used(row, col) {
                continue;
            }

            let mut location = BlockLocation {
                start: Point { row, col },
                end: Point { row, col },
            };

            let new_location = try_compress_motion_vectors(dimensions, mvs, (row, col), &ONE_OUT);
            if location == new_location {
                mv_locations.push((mvs[row][col], location));
                // Mark the single block as used
                used_table.mark_used(row, col);
                continue;
            }

            location = new_location;

            let new_location = try_compress_motion_vectors(dimensions, mvs, (row, col), &TWO_OUT);
            if location == new_location {
                mv_locations.push((mvs[row][col], location));
                used_table.mark_area_used(
                    location.start.row,
                    location.start.col,
                    location.end.row,
                    location.end.col,
                );

                continue;
            }

            let new_location = try_compress_motion_vectors(dimensions, mvs, (row, col), &THREE_OUT);
            if location == new_location {
                mv_locations.push((mvs[row][col], location));
                // Mark all blocks in the location as used
                used_table.mark_area_used(
                    location.start.row,
                    location.start.col,
                    location.end.row,
                    location.end.col,
                );
                continue;
            }

            mv_locations.push((mvs[row][col], location));
            used_table.mark_area_used(
                location.start.row,
                location.start.col,
                location.end.row,
                location.end.col,
            );
        }
    }

    mv_locations
}

pub(crate) struct UsedTable {
    table: Vec<u8>,
    cols: usize,
}

impl UsedTable {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Self {
            table: vec![0; rows * cols],
            cols,
        }
    }

    pub(crate) fn is_used(&self, row: usize, col: usize) -> bool {
        self.table[row * self.cols + col] != 0
    }

    pub(crate) fn mark_used(&mut self, row: usize, col: usize) {
        self.table[row * self.cols + col] = 1;
    }

    pub(crate) fn mark_area_used(
        &mut self,
        start_row: usize,
        start_col: usize,
        end_row: usize,
        end_col: usize,
    ) {
        for r in start_row..=end_row {
            for c in start_col..=end_col {
                self.table[r * self.cols + c] = 1;
            }
        }
    }
}

fn predicted_chroma_block(
    (chroma_row, chroma_col): (usize, usize),
    prediction: Prediction,
    chroma_dimensions: BlockDimensions,
    subsampling: Subsampling,
    backward_channel: &[Block<i16>],
    forward_channel: Option<&[Block<i16>]>,
) -> Block<i16> {
    // Scale the prediction's motion vectors for chroma
    let scaled_prediction = match prediction {
        Prediction::Forward(mv) => Prediction::Forward(mv.scale_for_chroma(subsampling)),
        Prediction::Backward(mv) => Prediction::Backward(mv.scale_for_chroma(subsampling)),
        Prediction::Both { forward, backward } => Prediction::Both {
            forward: forward.scale_for_chroma(subsampling),
            backward: backward.scale_for_chroma(subsampling),
        },
    };

    // Use the base predicted_block helper for this chroma channel
    predicted_block(
        (chroma_row, chroma_col),
        scaled_prediction,
        chroma_dimensions,
        backward_channel,
        forward_channel,
    )
}

fn predicted_block(
    (row, col): (usize, usize),
    prediction: Prediction,
    dimensions: BlockDimensions,
    backward: &[Block<i16>],
    forward: Option<&[Block<i16>]>,
) -> Block<i16> {
    let BlockDimensions { width, height } = dimensions;
    let idx = row * width + col;

    match prediction {
        Prediction::Forward(mv) => {
            let forward = forward.expect("Forward prediction requires forward reference frame");
            let src_r = row as isize + mv.y;
            let src_c = col as isize + mv.x;

            if src_r >= 0 && src_r < height as isize && src_c >= 0 && src_c < width as isize {
                let idx = (src_r as usize) * width + (src_c as usize);
                forward.get(idx).copied().unwrap_or_default()
            } else {
                // Fallback to current position if motion vector goes out of bounds
                forward.get(idx).copied().unwrap_or_default()
            }
        }
        Prediction::Backward(mv) => {
            let src_r = row as isize + mv.y;
            let src_c = col as isize + mv.x;

            if src_r >= 0 && src_r < height as isize && src_c >= 0 && src_c < width as isize {
                let idx = (src_r as usize) * width + (src_c as usize);
                backward.get(idx).copied().unwrap_or_default()
            } else {
                // Fallback to current position if motion vector goes out of bounds
                backward.get(idx).copied().unwrap_or_default()
            }
        }
        Prediction::Both {
            forward: forward_mv,
            backward: backward_mv,
        } => {
            let forward =
                forward.expect("Bidirectional prediction requires forward reference frame");

            let forward_r = row as isize + forward_mv.y;
            let forward_c = col as isize + forward_mv.x;
            let backward_r = row as isize + backward_mv.y;
            let backward_c = col as isize + backward_mv.x;

            let forward_in_bounds = forward_r >= 0
                && forward_r < height as isize
                && forward_c >= 0
                && forward_c < width as isize;
            let backward_in_bounds = backward_r >= 0
                && backward_r < height as isize
                && backward_c >= 0
                && backward_c < width as isize;

            match (forward_in_bounds, backward_in_bounds) {
                (false, false) => forward.get(idx).copied().unwrap_or_default(),
                (false, true) => {
                    let backward_idx = (backward_r as usize) * width + (backward_c as usize);
                    backward.get(backward_idx).copied().unwrap_or_default()
                }
                (true, false) => {
                    let forward_idx = (forward_r as usize) * width + (forward_c as usize);
                    forward.get(forward_idx).copied().unwrap_or_default()
                }
                (true, true) => {
                    let forward_idx = (forward_r as usize) * width + (forward_c as usize);
                    let backward_idx = (backward_r as usize) * width + (backward_c as usize);

                    // Average the two predictions
                    let fwd_block = forward[forward_idx];
                    let bwd_block = backward[backward_idx];

                    // Average element-wise
                    let mut result = Block::default();
                    for i in 0..Block::<i16>::size() {
                        let avg = (fwd_block[i] as i32 + bwd_block[i] as i32) / 2;
                        result[i] = avg as i16;
                    }
                    result
                }
            }
        }
    }
}
