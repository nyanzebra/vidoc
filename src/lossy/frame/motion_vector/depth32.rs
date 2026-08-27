use super::{MotionVector, LARGE_DIAMOND, SMALL_DIAMOND};
use crate::{block::Block, dimensions::BlockDimensions, point::Point};

/// Large Diamond Search Pattern for i32 blocks (integer-only, no sub-pixel)
#[allow(dead_code)]
pub(crate) fn ldsp_blocks(
    current: &Block<i32>,
    reference: &[Block<i32>],
    dimensions: &BlockDimensions,
    point: Point,
) -> (MotionVector, i32) {
    // Stage 1: Integer-pixel search with large diamond
    let mut best = MotionVector { x: 0, y: 0 };
    let mut best_score = sum_of_abs_diff_block(
        current,
        reference,
        dimensions,
        point,
        MotionVector { x: 0, y: 0 },
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
    exhaustive_refine(current, reference, dimensions, point, best, best_score)
}

/// Small Diamond Search Pattern for i32 blocks
#[allow(dead_code)]
pub(crate) fn sdsp_blocks(
    current: &Block<i32>,
    reference: &[Block<i32>],
    dimensions: &BlockDimensions,
    point: Point,
    best_mv: Option<MotionVector>,
    best_score: Option<i32>,
) -> (MotionVector, i32) {
    let mut best = best_mv.unwrap_or_default();
    let mut best_score = best_score.unwrap_or_else(|| {
        sum_of_abs_diff_block(
            current,
            reference,
            dimensions,
            point,
            MotionVector { x: 0, y: 0 },
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
        );
        if score < best_score {
            best = MotionVector { x: dx, y: dy };
            best_score = score;
        }
    }

    (best, best_score)
}

/// Exhaustive refinement in ±1 block around the best position
#[allow(dead_code)]
fn exhaustive_refine(
    current: &Block<i32>,
    reference: &[Block<i32>],
    dimensions: &BlockDimensions,
    point: Point,
    mut best: MotionVector,
    mut best_score: i32,
) -> (MotionVector, i32) {
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

            let score = sum_of_abs_diff_block(current, reference, dimensions, point, test_mv);
            if score < best_score {
                best = test_mv;
                best_score = score;
            }
        }
    }

    (best, best_score)
}

/// Sum of Absolute Differences for integer block positions
#[allow(dead_code)]
pub(crate) fn sum_of_abs_diff_block(
    current: &Block<i32>,
    reference: &[Block<i32>],
    dimensions: &BlockDimensions,
    point: Point,
    mv: MotionVector,
) -> i32 {
    let (row, col) = (point.row as isize + mv.y, point.col as isize + mv.x);
    if row >= dimensions.height as isize || row < 0 || col >= dimensions.width as isize || col < 0 {
        return i32::MAX;
    }

    let idx = row * dimensions.width as isize + col;
    if idx as usize >= reference.len() {
        return i32::MAX;
    }

    current.sum_of_abs_difference(&reference[idx as usize])
}
