use std::f64::consts::SQRT_2;

use crate::block::Block;

// https://en.wikipedia.org/wiki/Discrete_cosine_transform
// Allow identity_op and erasing_op for clarity in matrix indexing
#[allow(clippy::identity_op, clippy::erasing_op)]
impl Block<f64> {
    pub fn dct(self) -> Self {
        let mut data = self.0;

        // horizontal - process each row
        for r in 0..8 {
            let row_start = r * 8;
            let row = data[row_start..row_start + 8].try_into().unwrap();
            let transformed_row = dct1_fast(row);
            data[row_start..row_start + 8].copy_from_slice(&transformed_row);
        }

        // vertical - process each column
        for c in 0..8 {
            let line = [
                data[0 * 8 + c],
                data[1 * 8 + c],
                data[2 * 8 + c],
                data[3 * 8 + c],
                data[4 * 8 + c],
                data[5 * 8 + c],
                data[6 * 8 + c],
                data[7 * 8 + c],
            ];
            let col = dct1_fast(line);
            data[0 * 8 + c] = col[0];
            data[1 * 8 + c] = col[1];
            data[2 * 8 + c] = col[2];
            data[3 * 8 + c] = col[3];
            data[4 * 8 + c] = col[4];
            data[5 * 8 + c] = col[5];
            data[6 * 8 + c] = col[6];
            data[7 * 8 + c] = col[7];
        }

        Self(data)
    }

    pub fn idct(self) -> Self {
        let mut data = self.0;

        // horizontal - process each row
        for r in 0..Block::<f64>::rows() {
            let row_start = r * 8;
            let row = data[row_start..row_start + 8].try_into().unwrap();
            let transformed_row = idct1_fast(row);
            data[row_start..row_start + 8].copy_from_slice(&transformed_row);
        }

        // vertical - process each column
        for c in 0..Block::<f64>::cols() {
            let line = [
                data[0 * 8 + c],
                data[1 * 8 + c],
                data[2 * 8 + c],
                data[3 * 8 + c],
                data[4 * 8 + c],
                data[5 * 8 + c],
                data[6 * 8 + c],
                data[7 * 8 + c],
            ];
            let col = idct1_fast(line);
            data[0 * 8 + c] = col[0];
            data[1 * 8 + c] = col[1];
            data[2 * 8 + c] = col[2];
            data[3 * 8 + c] = col[3];
            data[4 * 8 + c] = col[4];
            data[5 * 8 + c] = col[5];
            data[6 * 8 + c] = col[6];
            data[7 * 8 + c] = col[7];
        }

        Self(data)
    }
}

// 8.sqrt()
const SQRT_8: f64 = 2.8284271247461903;
const LLM_C1_COSINE: f64 = 0.9807852804032304;
const LLM_C1_SINE: f64 = 0.19509032201612825;
const LLM_C3_COSINE: f64 = 0.8314696123025452;
const LLM_C3_SINE: f64 = 0.5555702330196022;
const LLM_C6_COSINE: f64 = 0.38268343236508984;
const LLM_C6_SINE: f64 = 0.9238795325112867;

/// REF:
/// https://unix4lyfe.org/dct-1d/
/// See LLM section
#[inline]
fn dct1_fast(line: [f64; 8]) -> [f64; 8] {
    let s1 = dct1_fast_stage1(line);
    let s2 = dct1_fast_stage2(s1);
    let s3 = dct1_fast_stage3(s2);
    let s4 = dct1_fast_stage4(s3);
    dct1_fast_shuffle(s4)
}

#[inline]
fn dct1_fast_stage1(line: [f64; 8]) -> [f64; 8] {
    let c0 = f64::algebraic_add(line[0], line[7]);
    let c1 = f64::algebraic_add(line[1], line[6]);
    let c2 = f64::algebraic_add(line[2], line[5]);
    let c3 = f64::algebraic_add(line[3], line[4]);
    let c4 = f64::algebraic_sub(line[3], line[4]);
    let c5 = f64::algebraic_sub(line[2], line[5]);
    let c6 = f64::algebraic_sub(line[1], line[6]);
    let c7 = f64::algebraic_sub(line[0], line[7]);
    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn dct1_fast_stage2(line: [f64; 8]) -> [f64; 8] {
    let c0 = f64::algebraic_add(line[0], line[3]);
    let c1 = f64::algebraic_add(line[1], line[2]);
    let c2 = f64::algebraic_sub(line[1], line[2]);
    let c3 = f64::algebraic_sub(line[0], line[3]);

    // c4 and c7 are pairs
    let c4 = twist1(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);
    let c7 = twist2(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);
    // c5 and c6 are pairs
    let c5 = twist1(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);
    let c6 = twist2(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);
    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn dct1_fast_stage3(line: [f64; 8]) -> [f64; 8] {
    let c0 = f64::algebraic_add(line[0], line[1]);
    let c1 = f64::algebraic_sub(line[0], line[1]);
    let c2 = twist1(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);
    let c3 = twist2(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);
    let c4 = f64::algebraic_add(line[4], line[6]);
    let c5 = f64::algebraic_sub(line[7], line[5]);
    let c6 = f64::algebraic_sub(line[4], line[6]);
    let c7 = f64::algebraic_add(line[7], line[5]);
    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn dct1_fast_stage4(line: [f64; 8]) -> [f64; 8] {
    let c0 = line[0];
    let c1 = line[1];
    let c2 = line[2];
    let c3 = line[3];
    let c4 = f64::algebraic_sub(line[7], line[4]);
    let c5 = f64::algebraic_mul(line[5], SQRT_2);
    let c6 = f64::algebraic_mul(line[6], SQRT_2);
    let c7 = f64::algebraic_add(line[7], line[4]);

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn dct1_fast_shuffle(line: [f64; 8]) -> [f64; 8] {
    let inv = f64::algebraic_div(1.0, SQRT_8);
    [
        f64::algebraic_mul(line[0], inv), // 0
        f64::algebraic_mul(line[7], inv), // 1
        f64::algebraic_mul(line[2], inv), // 2
        f64::algebraic_mul(line[5], inv), // 3
        f64::algebraic_mul(line[1], inv), // 4
        f64::algebraic_mul(line[6], inv), // 5
        f64::algebraic_mul(line[3], inv), // 6
        f64::algebraic_mul(line[4], inv), // 7
    ]
}

#[inline]
fn twist1(x: f64, y: f64, c: f64, s: f64, scale: f64) -> f64 {
    f64::algebraic_mul(
        scale,
        f64::algebraic_add(f64::algebraic_mul(x, c), f64::algebraic_mul(y, s)),
    )
}

#[inline]
fn twist2(x: f64, y: f64, c: f64, s: f64, scale: f64) -> f64 {
    f64::algebraic_mul(
        scale,
        f64::algebraic_add(f64::algebraic_mul(-x, s), f64::algebraic_mul(y, c)),
    )
}

/// This is just the reverse of `dct1_fast`
fn idct1_fast(line: [f64; 8]) -> [f64; 8] {
    let s0 = idct1_fast_unshuffle(line);
    let s1 = idct1_fast_stage1(s0);
    let s2 = idct1_fast_stage2(s1);
    let s3 = idct1_fast_stage3(s2);
    idct1_fast_stage4(s3)
}

#[inline]
fn idct1_fast_unshuffle(line: [f64; 8]) -> [f64; 8] {
    [
        line[0] * SQRT_8,
        line[4] * SQRT_8,
        line[2] * SQRT_8,
        line[6] * SQRT_8,
        line[7] * SQRT_8,
        line[3] * SQRT_8,
        line[5] * SQRT_8,
        line[1] * SQRT_8,
    ]
}

#[inline]
fn idct1_fast_stage1(line: [f64; 8]) -> [f64; 8] {
    let c0 = line[0];
    let c1 = line[1];
    let c2 = line[2];
    let c3 = line[3];
    let c4 = (line[7] - line[4]) / 2.0;
    let c5 = line[5] / SQRT_2;
    let c6 = line[6] / SQRT_2;
    let c7 = (line[7] + line[4]) / 2.0;

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn idct1_fast_stage2(line: [f64; 8]) -> [f64; 8] {
    let c0 = (line[0] + line[1]) / 2.0;
    let c1 = (-line[1] + line[0]) / 2.0;
    let c2 = untwist1(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);
    let c3 = untwist2(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);
    let c4 = (line[4] + line[6]) / 2.0;
    let c5 = (line[7] - line[5]) / 2.0;
    let c6 = (line[4] - line[6]) / 2.0;
    let c7 = (line[7] + line[5]) / 2.0;

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn idct1_fast_stage3(line: [f64; 8]) -> [f64; 8] {
    let c0 = (line[0] + line[3]) / 2.0;
    let c1 = (line[1] + line[2]) / 2.0;
    let c2 = (line[1] - line[2]) / 2.0;
    let c3 = (line[0] - line[3]) / 2.0;

    // c4 and c7 are pairs
    let c4 = untwist1(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);
    let c7 = untwist2(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);
    // c5 and c6 are pairs
    let c5 = untwist1(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);
    let c6 = untwist2(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn idct1_fast_stage4(line: [f64; 8]) -> [f64; 8] {
    let c0 = (line[0] + line[7]) / 2.0;
    let c1 = (line[1] + line[6]) / 2.0;
    let c2 = (line[2] + line[5]) / 2.0;
    let c3 = (line[3] + line[4]) / 2.0;
    let c4 = (line[3] - line[4]) / 2.0;
    let c5 = (line[2] - line[5]) / 2.0;
    let c6 = (line[1] - line[6]) / 2.0;
    let c7 = (line[0] - line[7]) / 2.0;

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

#[inline]
fn untwist1(x: f64, y: f64, c: f64, s: f64, scale: f64) -> f64 {
    // x and y are the results of a twist 1 or 2
    // x = (a * c) + (b * s)
    // y = (-a * s) + (b * c)
    // ->
    // x / s = (a * c / s) + b
    // y / c = (-a * s / c) + b
    // ->
    // (x / s) - (y / c) = (a * c / s) + (a * s / c)
    // ... = a * ((c / s) + (s / c))
    let x = x / (s * scale);
    let y = y / (c * scale);
    (x - y) / ((s / c) + (c / s))
}

#[inline]
fn untwist2(x: f64, y: f64, c: f64, s: f64, scale: f64) -> f64 {
    // same as `untwist1` but solve for b
    let x = x / (c * scale);
    let y = y / (s * scale);
    (x + y) / ((s / c) + (c / s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dct_idct() {
        let block = Block([
            52.0, 55.0, 61.0, 66.0, 70.0, 61.0, 64.0, 73.0, 63.0, 59.0, 55.0, 90.0, 109.0, 85.0,
            69.0, 72.0, 62.0, 59.0, 68.0, 113.0, 144.0, 104.0, 66.0, 73.0, 63.0, 58.0, 71.0, 122.0,
            154.0, 106.0, 70.0, 69.0, 67.0, 61.0, 68.0, 104.0, 126.0, 88.0, 68.0, 70.0, 79.0, 65.0,
            60.0, 70.0, 77.0, 68.0, 58.0, 75.0, 85.0, 71.0, 64.0, 59.0, 55.0, 61.0, 65.0, 83.0,
            87.0, 79.0, 69.0, 68.0, 65.0, 76.0, 78.0, 94.0,
        ]);
        println!("{:?}", block.dct().idct());
    }

    #[test]
    fn dct_idct_is_orig() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_f64s_eq(&idct1_fast(dct1_fast(test)), &test);
    }

    #[test]
    fn check_dct_idct_is_orig_by_summation() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let s1 = dct1_fast_stage1(test);
        let s2 = dct1_fast_stage2(s1);
        let s3 = dct1_fast_stage3(s2);
        let s4 = dct1_fast_stage4(s3);
        let is1 = idct1_fast_stage1(s4);
        let is2 = idct1_fast_stage2(is1);
        let is3 = idct1_fast_stage3(is2);
        let is4 = idct1_fast_stage4(is3);
        assert_f64s_eq(&test, &is4);
    }

    #[test]
    fn shuffle_unshuffle() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            idct1_fast_unshuffle(dct1_fast_shuffle(test))
                .iter()
                .map(|x| x.round())
                .collect::<Vec<_>>(),
            test
        );
    }

    #[test]
    fn twist() {
        let t1 = twist1(2.0, 4.5, LLM_C3_COSINE, LLM_C3_SINE, 2.0);
        let t2 = twist2(2.0, 4.5, LLM_C3_COSINE, LLM_C3_SINE, 2.0);
        let u1 = untwist1(t1, t2, LLM_C3_COSINE, LLM_C3_SINE, 2.0);
        let u2 = untwist2(t1, t2, LLM_C3_COSINE, LLM_C3_SINE, 2.0);
        assert_f64_eq(u1, 2.0);
        assert_f64_eq(u2, 4.5);
    }

    #[test]
    fn stage1() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_f64s_eq(&idct1_fast_stage1(dct1_fast_stage4(test)), &test);
    }

    #[test]
    fn stage2() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_f64s_eq(&idct1_fast_stage2(dct1_fast_stage3(test)), &test);
    }

    #[test]
    fn stage3() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_f64s_eq(&idct1_fast_stage3(dct1_fast_stage2(test)), &test);
    }

    #[test]
    fn stage4() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_f64s_eq(&idct1_fast_stage4(dct1_fast_stage1(test)), &test);
    }

    fn assert_f64_eq(x: f64, y: f64) {
        assert!((x - y).abs() < (f64::EPSILON * 10.0), "{x} != {y}");
    }

    fn assert_f64s_eq(x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len(), "{x:?} != {y:?}");
        for i in 0..x.len() {
            assert_f64_eq(x[i], y[i]);
        }
    }
}
