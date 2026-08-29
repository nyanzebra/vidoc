use wide::f32x4;

use crate::block::Block;

// https://en.wikipedia.org/wiki/Discrete_cosine_transform

const SQRT_8: f32 = 2.828_427;
const LLM_C1_COSINE: f32 = 0.980_785_25;
const LLM_C1_SINE: f32 = 0.195_090_32;
const LLM_C3_COSINE: f32 = 0.831_469_6;
const LLM_C3_SINE: f32 = 0.555_570_24;
const LLM_C6_COSINE: f32 = 0.382_683_43;
const LLM_C6_SINE: f32 = 0.923_879_5;

// -----------------------------------------------------------------------------
// SIMD 2-D DCT / IDCT
// -----------------------------------------------------------------------------
//
// We use f32x4 to process four independent 1-D transforms simultaneously.
//
// For example, during the row pass:
//
//   x0 = [row0[0], row1[0], row2[0], row3[0]]
//   x1 = [row0[1], row1[1], row2[1], row3[1]]
//   ...
//   x7 = [row0[7], row1[7], row2[7], row3[7]]
//
// The SIMD lanes therefore represent four independent DCTs.
//
// `wide` provides the appropriate implementation for the target platform,
// including x86/x86-64, ARM/NEON, and other supported SIMD targets.
// -----------------------------------------------------------------------------

impl Block<i16> {
    #[inline]
    pub fn dct(self) -> Self {
        Self::from(Block::<f32>::from(self).dct())
    }

    #[inline]
    pub fn idct(self) -> Self {
        Self::from(Block::<f32>::from(self).idct())
    }
}

impl Block<i32> {
    #[inline]
    pub fn dct(self) -> Self {
        Self::from(Block::<f32>::from(self).dct())
    }

    #[inline]
    pub fn idct(self) -> Self {
        Self::from(Block::<f32>::from(self).idct())
    }
}

impl Block<f32> {
    #[inline]
    fn dct(self) -> Self {
        let mut data = self.0;

        // Horizontal pass: four rows at a time.
        for row_group in 0..2 {
            let r = row_group * 4;

            let input = [
                f32x4::new([
                    data[(r) * 8],
                    data[(r + 1) * 8],
                    data[(r + 2) * 8],
                    data[(r + 3) * 8],
                ]),
                f32x4::new([
                    data[(r) * 8 + 1],
                    data[(r + 1) * 8 + 1],
                    data[(r + 2) * 8 + 1],
                    data[(r + 3) * 8 + 1],
                ]),
                f32x4::new([
                    data[(r) * 8 + 2],
                    data[(r + 1) * 8 + 2],
                    data[(r + 2) * 8 + 2],
                    data[(r + 3) * 8 + 2],
                ]),
                f32x4::new([
                    data[(r) * 8 + 3],
                    data[(r + 1) * 8 + 3],
                    data[(r + 2) * 8 + 3],
                    data[(r + 3) * 8 + 3],
                ]),
                f32x4::new([
                    data[(r) * 8 + 4],
                    data[(r + 1) * 8 + 4],
                    data[(r + 2) * 8 + 4],
                    data[(r + 3) * 8 + 4],
                ]),
                f32x4::new([
                    data[(r) * 8 + 5],
                    data[(r + 1) * 8 + 5],
                    data[(r + 2) * 8 + 5],
                    data[(r + 3) * 8 + 5],
                ]),
                f32x4::new([
                    data[(r) * 8 + 6],
                    data[(r + 1) * 8 + 6],
                    data[(r + 2) * 8 + 6],
                    data[(r + 3) * 8 + 6],
                ]),
                f32x4::new([
                    data[(r) * 8 + 7],
                    data[(r + 1) * 8 + 7],
                    data[(r + 2) * 8 + 7],
                    data[(r + 3) * 8 + 7],
                ]),
            ];

            let output = dct::dct1_simd(input);

            for i in 0..8 {
                let values = output[i].to_array();

                data[r * 8 + i] = values[0];
                data[(r + 1) * 8 + i] = values[1];
                data[(r + 2) * 8 + i] = values[2];
                data[(r + 3) * 8 + i] = values[3];
            }
        }

        // Vertical pass.
        //
        // We process four columns simultaneously. Each vector contains
        // four independent column values.
        for column_group in 0..2 {
            let c = column_group * 4;

            let input = [
                f32x4::new([data[c], data[c + 1], data[c + 2], data[c + 3]]),
                f32x4::new([
                    data[8 + c],
                    data[8 + c + 1],
                    data[8 + c + 2],
                    data[8 + c + 3],
                ]),
                f32x4::new([
                    data[16 + c],
                    data[16 + c + 1],
                    data[16 + c + 2],
                    data[16 + c + 3],
                ]),
                f32x4::new([
                    data[24 + c],
                    data[24 + c + 1],
                    data[24 + c + 2],
                    data[24 + c + 3],
                ]),
                f32x4::new([
                    data[32 + c],
                    data[32 + c + 1],
                    data[32 + c + 2],
                    data[32 + c + 3],
                ]),
                f32x4::new([
                    data[40 + c],
                    data[40 + c + 1],
                    data[40 + c + 2],
                    data[40 + c + 3],
                ]),
                f32x4::new([
                    data[48 + c],
                    data[48 + c + 1],
                    data[48 + c + 2],
                    data[48 + c + 3],
                ]),
                f32x4::new([
                    data[56 + c],
                    data[56 + c + 1],
                    data[56 + c + 2],
                    data[56 + c + 3],
                ]),
            ];

            let output = dct::dct1_simd(input);

            for i in 0..8 {
                let values = output[i].to_array();

                data[i * 8 + c] = values[0];
                data[i * 8 + c + 1] = values[1];
                data[i * 8 + c + 2] = values[2];
                data[i * 8 + c + 3] = values[3];
            }
        }

        Self(data)
    }

    #[inline]
    fn idct(self) -> Self {
        let mut data = self.0;

        // Horizontal pass.
        for row_group in 0..2 {
            let r = row_group * 4;

            let input = [
                f32x4::new([
                    data[r * 8],
                    data[(r + 1) * 8],
                    data[(r + 2) * 8],
                    data[(r + 3) * 8],
                ]),
                f32x4::new([
                    data[r * 8 + 1],
                    data[(r + 1) * 8 + 1],
                    data[(r + 2) * 8 + 1],
                    data[(r + 3) * 8 + 1],
                ]),
                f32x4::new([
                    data[r * 8 + 2],
                    data[(r + 1) * 8 + 2],
                    data[(r + 2) * 8 + 2],
                    data[(r + 3) * 8 + 2],
                ]),
                f32x4::new([
                    data[r * 8 + 3],
                    data[(r + 1) * 8 + 3],
                    data[(r + 2) * 8 + 3],
                    data[(r + 3) * 8 + 3],
                ]),
                f32x4::new([
                    data[r * 8 + 4],
                    data[(r + 1) * 8 + 4],
                    data[(r + 2) * 8 + 4],
                    data[(r + 3) * 8 + 4],
                ]),
                f32x4::new([
                    data[r * 8 + 5],
                    data[(r + 1) * 8 + 5],
                    data[(r + 2) * 8 + 5],
                    data[(r + 3) * 8 + 5],
                ]),
                f32x4::new([
                    data[r * 8 + 6],
                    data[(r + 1) * 8 + 6],
                    data[(r + 2) * 8 + 6],
                    data[(r + 3) * 8 + 6],
                ]),
                f32x4::new([
                    data[r * 8 + 7],
                    data[(r + 1) * 8 + 7],
                    data[(r + 2) * 8 + 7],
                    data[(r + 3) * 8 + 7],
                ]),
            ];

            let output = idct::idct1_simd(input);

            for i in 0..8 {
                let values = output[i].to_array();

                data[r * 8 + i] = values[0];
                data[(r + 1) * 8 + i] = values[1];
                data[(r + 2) * 8 + i] = values[2];
                data[(r + 3) * 8 + i] = values[3];
            }
        }

        // Vertical pass.
        for column_group in 0..2 {
            let c = column_group * 4;

            let input = [
                f32x4::new([data[c], data[c + 1], data[c + 2], data[c + 3]]),
                f32x4::new([
                    data[8 + c],
                    data[8 + c + 1],
                    data[8 + c + 2],
                    data[8 + c + 3],
                ]),
                f32x4::new([
                    data[16 + c],
                    data[16 + c + 1],
                    data[16 + c + 2],
                    data[16 + c + 3],
                ]),
                f32x4::new([
                    data[24 + c],
                    data[24 + c + 1],
                    data[24 + c + 2],
                    data[24 + c + 3],
                ]),
                f32x4::new([
                    data[32 + c],
                    data[32 + c + 1],
                    data[32 + c + 2],
                    data[32 + c + 3],
                ]),
                f32x4::new([
                    data[40 + c],
                    data[40 + c + 1],
                    data[40 + c + 2],
                    data[40 + c + 3],
                ]),
                f32x4::new([
                    data[48 + c],
                    data[48 + c + 1],
                    data[48 + c + 2],
                    data[48 + c + 3],
                ]),
                f32x4::new([
                    data[56 + c],
                    data[56 + c + 1],
                    data[56 + c + 2],
                    data[56 + c + 3],
                ]),
            ];

            let output = idct::idct1_simd(input);

            for i in 0..8 {
                let values = output[i].to_array();

                data[i * 8 + c] = values[0];
                data[i * 8 + c + 1] = values[1];
                data[i * 8 + c + 2] = values[2];
                data[i * 8 + c + 3] = values[3];
            }
        }

        Self(data)
    }
}

mod dct {
    use std::f32::consts::SQRT_2;

    use wide::f32x4;

    use super::{
        LLM_C1_COSINE, LLM_C1_SINE, LLM_C3_COSINE, LLM_C3_SINE, LLM_C6_COSINE, LLM_C6_SINE, SQRT_8,
    };

    #[inline]
    pub fn dct1_simd(line: [f32x4; 8]) -> [f32x4; 8] {
        let s1 = dct_stage1(line);
        let s2 = dct_stage2(s1);
        let s3 = dct_stage3(s2);
        let s4 = dct_stage4(s3);
        dct_shuffle(s4)
    }

    #[inline]
    pub fn dct_stage1(line: [f32x4; 8]) -> [f32x4; 8] {
        let c0 = line[0] + line[7];
        let c1 = line[1] + line[6];
        let c2 = line[2] + line[5];
        let c3 = line[3] + line[4];

        let c4 = line[3] - line[4];
        let c5 = line[2] - line[5];
        let c6 = line[1] - line[6];
        let c7 = line[0] - line[7];

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn dct_stage2(line: [f32x4; 8]) -> [f32x4; 8] {
        let c0 = line[0] + line[3];
        let c1 = line[1] + line[2];
        let c2 = line[1] - line[2];
        let c3 = line[0] - line[3];

        let c4 = twist1(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);

        let c7 = twist2(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);

        let c5 = twist1(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);

        let c6 = twist2(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn dct_stage3(line: [f32x4; 8]) -> [f32x4; 8] {
        let c0 = line[0] + line[1];
        let c1 = line[0] - line[1];

        let c2 = twist1(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);

        let c3 = twist2(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);

        let c4 = line[4] + line[6];
        let c5 = line[7] - line[5];
        let c6 = line[4] - line[6];
        let c7 = line[7] + line[5];

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn dct_stage4(line: [f32x4; 8]) -> [f32x4; 8] {
        let c0 = line[0];
        let c1 = line[1];
        let c2 = line[2];
        let c3 = line[3];

        let c4 = line[7] - line[4];
        let c5 = line[5] * SQRT_2;
        let c6 = line[6] * SQRT_2;
        let c7 = line[7] + line[4];

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn dct_shuffle(line: [f32x4; 8]) -> [f32x4; 8] {
        let inv = f32x4::splat(1.0 / SQRT_8);

        [
            line[0] * inv,
            line[7] * inv,
            line[2] * inv,
            line[5] * inv,
            line[1] * inv,
            line[6] * inv,
            line[3] * inv,
            line[4] * inv,
        ]
    }

    #[inline]
    fn twist1(x: f32x4, y: f32x4, c: f32, s: f32, scale: f32) -> f32x4 {
        f32x4::splat(scale) * (x * f32x4::splat(c) + y * f32x4::splat(s))
    }

    #[inline]
    fn twist2(x: f32x4, y: f32x4, c: f32, s: f32, scale: f32) -> f32x4 {
        f32x4::splat(scale) * (-x * f32x4::splat(s) + y * f32x4::splat(c))
    }
}

mod idct {
    use std::f32::consts::SQRT_2;

    use wide::f32x4;

    use super::{
        LLM_C1_COSINE, LLM_C1_SINE, LLM_C3_COSINE, LLM_C3_SINE, LLM_C6_COSINE, LLM_C6_SINE, SQRT_8,
    };

    #[inline]
    pub fn idct1_simd(line: [f32x4; 8]) -> [f32x4; 8] {
        let s0 = idct_unshuffle(line);
        let s1 = idct_stage1(s0);
        let s2 = idct_stage2(s1);
        let s3 = idct_stage3(s2);
        idct_stage4(s3)
    }

    #[inline]
    pub fn idct_unshuffle(line: [f32x4; 8]) -> [f32x4; 8] {
        let scale = f32x4::splat(SQRT_8);

        [
            line[0] * scale,
            line[4] * scale,
            line[2] * scale,
            line[6] * scale,
            line[7] * scale,
            line[3] * scale,
            line[5] * scale,
            line[1] * scale,
        ]
    }

    #[inline]
    pub fn idct_stage1(line: [f32x4; 8]) -> [f32x4; 8] {
        let half = f32x4::splat(0.5);
        let inv_sqrt_2 = f32x4::splat(1.0 / SQRT_2);

        let c0 = line[0];
        let c1 = line[1];
        let c2 = line[2];
        let c3 = line[3];

        let c4 = (line[7] - line[4]) * half;
        let c5 = line[5] * inv_sqrt_2;
        let c6 = line[6] * inv_sqrt_2;
        let c7 = (line[7] + line[4]) * half;

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn idct_stage2(line: [f32x4; 8]) -> [f32x4; 8] {
        let half = f32x4::splat(0.5);

        let c0 = (line[0] + line[1]) * half;
        let c1 = (-line[1] + line[0]) * half;

        let c2 = untwist1(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);

        let c3 = untwist2(line[2], line[3], LLM_C6_COSINE, LLM_C6_SINE, SQRT_2);

        let c4 = (line[4] + line[6]) * half;
        let c5 = (line[7] - line[5]) * half;
        let c6 = (line[4] - line[6]) * half;
        let c7 = (line[7] + line[5]) * half;

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn idct_stage3(line: [f32x4; 8]) -> [f32x4; 8] {
        let half = f32x4::splat(0.5);

        let c0 = (line[0] + line[3]) * half;
        let c1 = (line[1] + line[2]) * half;
        let c2 = (line[1] - line[2]) * half;
        let c3 = (line[0] - line[3]) * half;

        let c4 = untwist1(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);

        let c7 = untwist2(line[4], line[7], LLM_C3_COSINE, LLM_C3_SINE, 1.0);

        let c5 = untwist1(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);

        let c6 = untwist2(line[5], line[6], LLM_C1_COSINE, LLM_C1_SINE, 1.0);

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn idct_stage4(line: [f32x4; 8]) -> [f32x4; 8] {
        let half = f32x4::splat(0.5);

        let c0 = (line[0] + line[7]) * half;
        let c1 = (line[1] + line[6]) * half;
        let c2 = (line[2] + line[5]) * half;
        let c3 = (line[3] + line[4]) * half;

        let c4 = (line[3] - line[4]) * half;
        let c5 = (line[2] - line[5]) * half;
        let c6 = (line[1] - line[6]) * half;
        let c7 = (line[0] - line[7]) * half;

        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline]
    pub fn untwist1(x: f32x4, y: f32x4, c: f32, s: f32, scale: f32) -> f32x4 {
        let c = f32x4::splat(c);
        let s = f32x4::splat(s);
        let scale = f32x4::splat(scale);

        let x = x / (s * scale);
        let y = y / (c * scale);

        (x - y) / (s / c + c / s)
    }

    #[inline]
    pub fn untwist2(x: f32x4, y: f32x4, c: f32, s: f32, scale: f32) -> f32x4 {
        let c = f32x4::splat(c);
        let s = f32x4::splat(s);
        let scale = f32x4::splat(scale);

        let x = x / (c * scale);
        let y = y / (s * scale);

        (x + y) / (s / c + c / s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dct_idct() {
        let block = Block::<i32>([
            52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 109, 85, 69, 72, 62, 59, 68, 113, 144,
            104, 66, 73, 63, 58, 71, 122, 154, 106, 70, 69, 67, 61, 68, 104, 126, 88, 68, 70, 79,
            65, 60, 70, 77, 68, 58, 75, 85, 71, 64, 59, 55, 61, 65, 83, 87, 79, 69, 68, 65, 76, 78,
            94,
        ]);

        let result = block.dct().idct();

        for (actual, expected) in result.0.iter().zip(block.0.iter()) {
            assert!(
                (*actual as f32 - *expected as f32).abs() < 0.001,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    pub fn dct_idct_is_orig() {
        let test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let input = [
            f32x4::splat(test[0]),
            f32x4::splat(test[1]),
            f32x4::splat(test[2]),
            f32x4::splat(test[3]),
            f32x4::splat(test[4]),
            f32x4::splat(test[5]),
            f32x4::splat(test[6]),
            f32x4::splat(test[7]),
        ];

        let output = idct::idct1_simd(dct::dct1_simd(input));

        for (i, value) in output.iter().enumerate() {
            let values = value.to_array();

            for actual in values {
                assert!((actual - test[i]).abs() < 0.001, "{actual} != {}", test[i]);
            }
        }
    }
}
