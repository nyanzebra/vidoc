use super::{
    MotionVector, HALF_PIXEL_OFFSETS, LARGE_DIAMOND, SMALL_DIAMOND, SUBPIXEL_SAD_THRESHOLD,
};
use crate::{block::Block, dimensions::BlockDimensions, point::Point};

/// Large Diamond Search Pattern for i16 blocks with sub-pixel refinement
pub(crate) fn ldsp_blocks(
    current: &Block<i16>,
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
) -> (MotionVector, i16) {
    // Stage 1: Integer-pixel search with large diamond
    let mut best = MotionVector { x: 0, y: 0 };
    let mut best_score = sum_of_abs_diff_block(
        current,
        reference,
        dimensions,
        point,
        MotionVector { x: 0, y: 0 },
        i16::MAX,
    );

    for (dx, dy) in LARGE_DIAMOND {
        let x = point.col as isize + dx;
        let y = point.row as isize + dy;
        if x < 0 || y < 0 || x >= dimensions.width as isize || y >= dimensions.height as isize {
            continue;
        }
        let score = sum_of_abs_diff_block(
            current,
            reference,
            dimensions,
            point,
            MotionVector { x: dx, y: dy },
            best_score,
        );
        if score < best_score {
            best = MotionVector { x: dx, y: dy };
            best_score = score;
        }
    }

    // Stage 2: Refine with small diamond
    let (best, best_score) = sdsp_blocks(
        current,
        reference,
        dimensions,
        point,
        Some(best),
        Some(best_score),
    );

    // Stage 3: Exhaustive refinement at integer level
    let (best_integer, best_score) =
        exhaustive_refine(current, reference, dimensions, point, best, best_score);

    // Stage 4: Half-pixel refinement using 6-tap interpolation
    const ENABLE_SUBPIXEL_ME: bool = false;

    if ENABLE_SUBPIXEL_ME {
        half_pixel_refine(
            current,
            reference,
            dimensions,
            point,
            best_integer,
            best_score,
        )
    } else {
        (best_integer, best_score)
    }
}

/// Small Diamond Search Pattern for i16 blocks
pub(crate) fn sdsp_blocks(
    current: &Block<i16>,
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
    best_mv: Option<MotionVector>,
    best_score: Option<i16>,
) -> (MotionVector, i16) {
    let mut best = best_mv.unwrap_or_default();
    let mut best_score = best_score.unwrap_or_else(|| {
        sum_of_abs_diff_block(
            current,
            reference,
            dimensions,
            point,
            MotionVector { x: 0, y: 0 },
            i16::MAX,
        )
    });

    for (dx, dy) in SMALL_DIAMOND {
        let x = point.col as isize + dx;
        let y = point.row as isize + dy;
        if x < 0 || y < 0 || x >= dimensions.width as isize || y >= dimensions.height as isize {
            continue;
        }
        let score = sum_of_abs_diff_block(
            current,
            reference,
            dimensions,
            point,
            MotionVector { x: dx, y: dy },
            best_score,
        );
        if score < best_score {
            best = MotionVector { x: dx, y: dy };
            best_score = score;
        }
    }

    (best, best_score)
}

/// Exhaustive refinement in ±1 block around the best position
fn exhaustive_refine(
    current: &Block<i16>,
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
    mut best: MotionVector,
    mut best_score: i16,
) -> (MotionVector, i16) {
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }

            let test_mv = MotionVector {
                x: best.x + dx,
                y: best.y + dy,
            };

            let test_x = point.col as isize + test_mv.x;
            let test_y = point.row as isize + test_mv.y;

            if test_x < 0
                || test_y < 0
                || test_x >= dimensions.width as isize
                || test_y >= dimensions.height as isize
            {
                continue;
            }

            let score =
                sum_of_abs_diff_block(current, reference, dimensions, point, test_mv, best_score);
            if score < best_score {
                best = test_mv;
                best_score = score;
            }
        }
    }

    (best, best_score)
}

/// Half-pixel refinement around the best integer position
fn half_pixel_refine(
    current: &Block<i16>,
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
    best_integer: MotionVector,
    best_score: i16,
) -> (MotionVector, i16) {
    // Early termination: if integer match is already very good, skip sub-pixel
    if best_score < SUBPIXEL_SAD_THRESHOLD {
        return (best_integer, best_score);
    }

    let mut best = best_integer;
    let mut best_score_mut = best_score;

    // Test 8 half-pixel positions around the integer position
    for (dx_half, dy_half) in HALF_PIXEL_OFFSETS {
        let test_mv = MotionVector {
            x: best_integer.x + dx_half,
            y: best_integer.y + dy_half,
        };

        let integer_x = test_mv.integer_x();
        let integer_y = test_mv.integer_y();
        let target_col = point.col as isize + integer_x;
        let target_row = point.row as isize + integer_y;

        if target_col < 0
            || target_row < 0
            || target_col >= dimensions.width as isize
            || target_row >= dimensions.height as isize
        {
            continue;
        }

        let score = sum_of_abs_diff_subpixel(current, reference, dimensions, point, test_mv);

        if score < best_score_mut {
            best = test_mv;
            best_score_mut = score;
        }
    }

    (best, best_score_mut)
}

/// Sum of Absolute Differences for integer block positions.
pub(crate) fn sum_of_abs_diff_block(
    current: &Block<i16>,
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
    mv: MotionVector,
    threshold: i16,
) -> i16 {
    let (row, col) = (point.row as isize + mv.y, point.col as isize + mv.x);
    if row >= dimensions.height as isize || row < 0 || col >= dimensions.width as isize || col < 0 {
        return i16::MAX;
    }

    let idx = row * dimensions.width as isize + col;
    if idx as usize >= reference.len() {
        return i16::MAX;
    }

    current.sum_of_abs_difference_early_exit(&reference[idx as usize], threshold)
}

/// Sum of Absolute Differences for sub-pixel positions (uses interpolation)
fn sum_of_abs_diff_subpixel(
    current: &Block<i16>,
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
    mv: MotionVector,
) -> i16 {
    let integer_x = mv.integer_x();
    let integer_y = mv.integer_y();
    let half_x = mv.has_half_pixel_x();
    let half_y = mv.has_half_pixel_y();

    let target_col = point.col as isize + integer_x;
    let target_row = point.row as isize + integer_y;

    if target_col < 0
        || target_row < 0
        || target_col >= dimensions.width as isize
        || target_row >= dimensions.height as isize
    {
        return i16::MAX;
    }

    if half_x || half_y {
        let base_pixel_x = (point.col as isize + integer_x) * 8;
        let base_pixel_y = (point.row as isize + integer_y) * 8;
        let max_pixel_width = (dimensions.width * 8) as isize;
        let max_pixel_height = (dimensions.height * 8) as isize;

        if half_x && (base_pixel_x - 2 < 0 || base_pixel_x + 10 >= max_pixel_width) {
            return i16::MAX;
        }
        if half_y && (base_pixel_y - 2 < 0 || base_pixel_y + 10 >= max_pixel_height) {
            return i16::MAX;
        }
    }

    if !half_x && !half_y {
        return sum_of_abs_diff_block(
            current,
            reference,
            dimensions,
            point,
            MotionVector {
                x: integer_x,
                y: integer_y,
            },
            i16::MAX,
        );
    }

    let interpolated = interpolate_half_pixel_block(
        reference, dimensions, point, half_x, half_y, integer_x, integer_y,
    );

    current.sum_of_abs_difference(&interpolated)
}

/// Get a pixel value from a block array with bounds checking.
#[inline(always)]
fn get_pixel_from_blocks(
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    pixel_x: isize,
    pixel_y: isize,
) -> i16 {
    if pixel_x < 0 || pixel_y < 0 {
        return 0;
    }

    let px = pixel_x as usize;
    let py = pixel_y as usize;

    // Use bit shifts instead of division (3x faster)
    let block_x = px >> 3; // pixel_x / 8
    let block_y = py >> 3; // pixel_y / 8
    let in_block_x = px & 7; // pixel_x % 8
    let in_block_y = py & 7; // pixel_y % 8

    if block_x >= dimensions.width || block_y >= dimensions.height {
        return 0;
    }

    let block_idx = block_y * dimensions.width + block_x;
    if block_idx >= reference.len() {
        return 0;
    }

    reference[block_idx].get(in_block_y, in_block_x)
}

/// 6-tap interpolation filter (H.264 standard)
/// Filter: (1, -5, 20, 20, -5, 1) / 32
fn apply_6tap_filter(samples: [i16; 6]) -> i16 {
    let [s0, s1, s2, s3, s4, s5] = samples.map(|s| s as i32);
    let result = (s0 - 5 * s1 + 20 * s2 + 20 * s3 - 5 * s4 + s5 + 16) >> 5;
    result.clamp(-128, 127) as i16
}

/// Interpolate a block at half-pixel position using 6-tap filter
fn interpolate_half_pixel_block(
    reference: &[Block<i16>],
    dimensions: &BlockDimensions,
    point: Point,
    mv_half_x: bool,
    mv_half_y: bool,
    integer_x: isize,
    integer_y: isize,
) -> Block<i16> {
    let mut result = Block::<i16>::default();
    let base_pixel_x = (point.col as isize + integer_x) * 8;
    let base_pixel_y = (point.row as isize + integer_y) * 8;

    for r in 0..8 {
        for c in 0..8 {
            let pixel_x = base_pixel_x + c as isize;
            let pixel_y = base_pixel_y + r as isize;

            let value = if !mv_half_x && !mv_half_y {
                // Integer position - direct copy
                get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y)
            } else if mv_half_x && !mv_half_y {
                // Half-pixel in X direction only - horizontal 6-tap
                let samples = [
                    get_pixel_from_blocks(reference, dimensions, pixel_x - 2, pixel_y),
                    get_pixel_from_blocks(reference, dimensions, pixel_x - 1, pixel_y),
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y),
                    get_pixel_from_blocks(reference, dimensions, pixel_x + 1, pixel_y),
                    get_pixel_from_blocks(reference, dimensions, pixel_x + 2, pixel_y),
                    get_pixel_from_blocks(reference, dimensions, pixel_x + 3, pixel_y),
                ];
                apply_6tap_filter(samples)
            } else if !mv_half_x && mv_half_y {
                // Half-pixel in Y direction only - vertical 6-tap
                let samples = [
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y - 2),
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y - 1),
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y),
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y + 1),
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y + 2),
                    get_pixel_from_blocks(reference, dimensions, pixel_x, pixel_y + 3),
                ];
                apply_6tap_filter(samples)
            } else {
                // Half-pixel in both X and Y - 2D interpolation
                // First apply horizontal 6-tap to get 6 intermediate values
                let mut temp = [0i16; 6];
                for (i, temp_val) in temp.iter_mut().enumerate() {
                    let y_offset = pixel_y + i as isize - 2;
                    let samples = [
                        get_pixel_from_blocks(reference, dimensions, pixel_x - 2, y_offset),
                        get_pixel_from_blocks(reference, dimensions, pixel_x - 1, y_offset),
                        get_pixel_from_blocks(reference, dimensions, pixel_x, y_offset),
                        get_pixel_from_blocks(reference, dimensions, pixel_x + 1, y_offset),
                        get_pixel_from_blocks(reference, dimensions, pixel_x + 2, y_offset),
                        get_pixel_from_blocks(reference, dimensions, pixel_x + 3, y_offset),
                    ];
                    *temp_val = apply_6tap_filter(samples);
                }
                // Then apply vertical 6-tap on the intermediate values
                apply_6tap_filter(temp)
            };

            result.set(r, c, value);
        }
    }

    result
}
