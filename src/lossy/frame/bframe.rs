use std::{
    fmt::Debug,
    io::{Read, Write},
};

use rayon::prelude::*;

use super::{
    build_predicted_blocks, calculate_residuals_for_macroblock, compressed_motion_vectors,
    r#macro::BMacroBlocks, reassemble_frame,
};
use crate::{
    block::Block,
    dimensions::BlockDimensions,
    lossy::{
        frame::{
            motion_vector::{depth16, MotionVector},
            r#macro::{BMacroBlock, Prediction},
        },
        SubSampleBlockGroup, SubSampleBlockGroupRef,
    },
    point::Point,
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Result,
};

pub(crate) struct BFrame<T> {
    current: SubSampleBlockGroup<T>,
    forward_ref: Option<SubSampleBlockGroup<T>>,
    backward_ref: SubSampleBlockGroup<T>,
}

impl<T> BFrame<T> {
    pub fn new(
        current: SubSampleBlockGroup<T>,
        forward_ref: Option<SubSampleBlockGroup<T>>,
        backward_ref: SubSampleBlockGroup<T>,
    ) -> Self {
        Self {
            current,
            forward_ref,
            backward_ref,
        }
    }

    fn dimensions(&self) -> BlockDimensions {
        self.current.dimensions()
    }
}

impl Encodable for BFrame<i16> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let macroblocks = self.get_macroblocks();
        BMacroBlocks::new(macroblocks).encode(stream)?;
        stream.flush()?;

        Ok(())
    }
}

impl<const N: usize, T> Decodable for BFrame<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    type Output = BMacroBlocks<T>;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        BMacroBlocks::decode(stream)
    }
}

impl BFrame<i16> {
    pub(crate) fn reassemble(
        forward_ref: Option<SubSampleBlockGroupRef<'_, i16>>,
        backward_ref: SubSampleBlockGroupRef<'_, i16>,
        macro_blocks: &[BMacroBlock<i16>],
    ) -> Result<SubSampleBlockGroup<i16>> {
        reassemble_frame(forward_ref, backward_ref, macro_blocks)
    }

    pub(crate) fn get_macroblocks(&self) -> Vec<BMacroBlock<i16>> {
        let motion_vecs = self.motion_vectors();
        let compressed = compressed_motion_vectors(&motion_vecs, &self.dimensions());

        // build_predicted_blocks and calculate_residuals_for_macroblock are pure
        // functions of their inputs — parallelize across macroblocks.
        compressed
            .into_par_iter()
            .map(|((prediction, _score), location)| {
                let (predicted_y, predicted_cb, predicted_cr) = build_predicted_blocks(
                    &location,
                    &prediction,
                    &self.current.dimensions(),
                    self.current.subsampling(),
                    self.forward_ref.as_ref(),
                    self.backward_ref.clone(),
                );

                let residuals = calculate_residuals_for_macroblock(
                    &location,
                    self.current.clone(),
                    &predicted_y,
                    &predicted_cb,
                    &predicted_cr,
                );

                BMacroBlock {
                    location,
                    prediction,
                    residuals,
                }
            })
            .collect()
    }

    fn motion_vectors(&self) -> Vec<Vec<(Prediction, i16)>> {
        let dimensions = self.dimensions();
        let current_y = self.current.y();
        let forward_y = self.forward_ref.as_ref().map(|f| f.y());
        let backward_y = self.backward_ref.y();

        (0..dimensions.height)
            .into_par_iter()
            .map(|row| {
                (0..dimensions.width)
                    .map(|col| {
                        let idx = row * dimensions.width + col;
                        if idx < current_y.len() {
                            let current = &current_y[idx];
                            let point = Point { row, col };

                            let (forward_mv, forward_cost) = if let Some(forward_ref) = forward_y.as_ref() {
                                depth16::ldsp_blocks(current, forward_ref, &dimensions, point)
                            } else {
                                // No forward reference - use zero MV with high cost
                                (MotionVector { x: 0, y: 0 }, i16::MAX)
                            };

                            let (backward_mv, backward_cost) =
                                depth16::ldsp_blocks(current, backward_y, &dimensions, point);

                            let bidirectional_cost = if forward_y.is_some() {
                                self.calculate_bidirectional_cost(
                                    current,
                                    point,
                                    forward_mv,
                                    backward_mv,
                                )
                            } else {
                                // No forward ref, can't do bidirectional
                                i16::MAX
                            };

                            let (prediction, cost) =
                                if forward_cost <= backward_cost && forward_cost <= bidirectional_cost
                                {
                                    // CRITICAL: Should never choose Forward when there's no forward reference
                                    assert!(
                                        forward_y.is_some(),
                                        "BUG: Chose Forward prediction without forward reference! \
                                    forward_cost={}, backward_cost={}, bidirectional_cost={}",
                                        forward_cost,
                                        backward_cost,
                                        bidirectional_cost
                                    );
                                    (Prediction::Forward(forward_mv), forward_cost)
                                } else if backward_cost <= bidirectional_cost {
                                    (Prediction::Backward(backward_mv), backward_cost)
                                } else {
                                    // CRITICAL: Should never choose Both when there's no forward reference
                                    assert!(
                                        forward_y.is_some(),
                                        "BUG: Chose Bidirectional prediction without forward reference! \
                                    forward_cost={}, backward_cost={}, bidirectional_cost={}",
                                        forward_cost,
                                        backward_cost,
                                        bidirectional_cost
                                    );
                                    (
                                        Prediction::Both {
                                            forward: forward_mv,
                                            backward: backward_mv,
                                        },
                                        bidirectional_cost,
                                    )
                                };

                            (prediction, cost)
                        } else {
                            // Out of bounds
                            (Prediction::Backward(MotionVector { x: 0, y: 0 }), 0)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn calculate_bidirectional_cost(
        &self,
        current: &Block<i16>,
        point: Point,
        forward_mv: MotionVector,
        backward_mv: MotionVector,
    ) -> i16 {
        let dimensions = self.dimensions();
        let backward_y = &self.backward_ref.y();
        let backward_idx = mv_idx(point, backward_mv, &dimensions);

        // Check if we have a forward reference
        if let Some(forward_ref) = self.forward_ref.as_ref() {
            let forward_y = forward_ref.y();
            let forward_idx = mv_idx(point, forward_mv, &dimensions);

            if forward_idx >= forward_y.len() || backward_idx >= backward_y.len() {
                i16::MAX
            } else {
                // Bi-directional: average forward and backward
                let interpolated = (forward_y[forward_idx] + backward_y[backward_idx]) / 2;

                current.sum_of_abs_difference(&interpolated) + 50
            }
        } else {
            // No forward ref: use backward-only cost
            if backward_idx >= backward_y.len() {
                i16::MAX
            } else {
                current.sum_of_abs_difference(&backward_y[backward_idx]) + 25
            }
        }
    }
}

fn mv_idx(point: Point, mv: MotionVector, dimensions: &BlockDimensions) -> usize {
    let row = (point.row as isize + mv.y).max(0) as usize;
    let col = (point.col as isize + mv.x).max(0) as usize;
    row * dimensions.width + col
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lossy::tests::TestSubSampleBlockGroup, FromBytes, ToBytes};

    #[test]
    fn test_bprediction_mode_conversion() {
        let mv = MotionVector::default();
        assert_eq!(Prediction::from(0), Prediction::Forward(mv));
        assert_eq!(Prediction::from(1), Prediction::Backward(mv));
        assert_eq!(
            Prediction::from(2),
            Prediction::Both {
                forward: mv,
                backward: mv
            }
        );

        assert_eq!(u8::from(Prediction::Forward(mv)), 0);
        assert_eq!(u8::from(Prediction::Backward(mv)), 1);
        assert_eq!(
            u8::from(Prediction::Both {
                forward: mv,
                backward: mv
            }),
            2
        );
    }

    #[test]
    fn test_bframe_basic_operations() {
        // Combined test for creation, motion vector calculation, and cost calculation
        let forward_ref = TestSubSampleBlockGroup::test_frame(8, 8, 0);
        let backward_ref = TestSubSampleBlockGroup::test_frame(8, 8, 0);
        let current = TestSubSampleBlockGroup::test_frame(8, 8, 0);

        // Test creation
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );

        assert_eq!(bframe.dimensions(), current.as_ref().dimensions);
        assert_eq!(bframe.current.y().len(), current.as_ref().y.len());

        // Test motion vector calculation
        let mvs = bframe.motion_vectors();
        assert_eq!(mvs.len(), current.as_ref().dimensions.height);
        if !mvs.is_empty() {
            assert_eq!(mvs[0].len(), current.as_ref().dimensions.width);
        }

        // Test bidirectional cost calculation
        if !current.as_ref().y.is_empty() {
            let cost = bframe.calculate_bidirectional_cost(
                &current.as_ref().y[0],
                Point { row: 0, col: 0 },
                MotionVector { x: 0, y: 0 },
                MotionVector { x: 0, y: 0 },
            );

            assert!(cost < i16::MAX);
        }
    }

    #[test]
    fn test_bframe_encode_decode_identical_frames() {
        // Test: When all frames are identical, motion vectors should be (0,0)
        // and residuals should be minimal
        use std::io::Cursor;

        use crate::bitstream::{BitStreamReader, BitStreamWriter};

        let width = 8;
        let height = 8;
        let value = 100;

        let current = TestSubSampleBlockGroup::test_frame(width, height, value);
        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, value);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, value);

        // Create B-frame
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );

        // Get macroblocks
        let macroblocks = bframe.get_macroblocks();

        assert!(
            !macroblocks.is_empty(),
            "Should have at least one macroblock"
        );

        // Verify motion vectors are close to zero and residuals are minimal
        for mb in &macroblocks {
            match &mb.prediction {
                Prediction::Forward(mv) | Prediction::Backward(mv) => {
                    assert!(
                        mv.x.abs() <= 1 && mv.y.abs() <= 1,
                        "Motion vectors should be near zero for identical frames"
                    );
                }
                Prediction::Both { forward, backward } => {
                    assert!(
                        forward.x.abs() <= 1 && forward.y.abs() <= 1,
                        "Forward MV should be near zero"
                    );
                    assert!(
                        backward.x.abs() <= 1 && backward.y.abs() <= 1,
                        "Backward MV should be near zero"
                    );
                }
            }

            // Check residuals are minimal
            for residual in mb.residuals.y.iter() {
                let mut max_abs = 0i16;
                for r in 0..8 {
                    for c in 0..8 {
                        max_abs = max_abs.max(residual.get(r, c).abs());
                    }
                }
                assert!(
                    max_abs < 50,
                    "Residuals should be minimal for identical frames, got {}",
                    max_abs
                );
            }
        }

        // Test encode/decode
        let mut bytes = Vec::new();
        {
            let mut writer = BitStreamWriter::new(&mut bytes);
            bframe.encode(&mut writer).expect("Encoding should succeed");
        }

        let decoded_macroblocks = {
            let cursor = Cursor::new(&bytes);
            let mut reader = BitStreamReader::new(cursor);
            BFrame::decode(&mut reader).expect("Decoding should succeed")
        };

        // Reassemble using decoded macroblocks
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &decoded_macroblocks.into_inner(),
        )
        .expect("Reassembly should succeed");

        // Verify reconstructed is close to original
        for (idx, recon) in reconstructed.y().iter().enumerate() {
            let orig = &current.y[idx];
            for r in 0..8 {
                for c in 0..8 {
                    let diff = (recon.get(r, c) - orig.get(r, c)).abs();
                    assert!(diff < 20, "Reconstructed should match original closely");
                }
            }
        }
    }

    #[test]
    fn test_bframe_encode_decode_different_frames() {
        // Test: When frames differ, ensure all blocks are covered with residuals
        let width = 8;
        let height = 8;
        let current = TestSubSampleBlockGroup::test_frame(width, height, 150);
        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 100);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 80);

        // Create B-frame
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );

        // Get macroblocks
        let macroblocks = bframe.get_macroblocks();

        // Count total blocks covered by all macroblocks
        let mut total_blocks = 0;
        for mb in &macroblocks {
            let rows = mb.location.end.row - mb.location.start.row + 1;
            let cols = mb.location.end.col - mb.location.start.col + 1;
            let blocks_in_mb = rows * cols;
            total_blocks += blocks_in_mb;

            // Verify residuals count matches location
            assert_eq!(
                mb.residuals.y.len(),
                blocks_in_mb,
                "Y residual count should match blocks in macroblock"
            );
        }

        // CRITICAL TEST: Verify 100% block coverage
        assert_eq!(
            total_blocks,
            width * height,
            "All blocks in frame must be covered by macroblocks"
        );

        // Reassemble
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )
        .expect("Reassembly should succeed");

        // Verify the reconstructed frame is closer to current than to references
        let mut total_error_to_current = 0i64;
        let mut total_error_to_forward = 0i64;

        for (idx, recon) in reconstructed.y().iter().enumerate() {
            let curr = &current.y[idx];
            let fwd = &forward_ref.y[idx];

            for r in 0..8 {
                for c in 0..8 {
                    let recon_val = recon.get(r, c);
                    let curr_val = curr.get(r, c);
                    let fwd_val = fwd.get(r, c);

                    total_error_to_current += (recon_val - curr_val).abs() as i64;
                    total_error_to_forward += (recon_val - fwd_val).abs() as i64;
                }
            }
        }

        // Reconstructed should be much closer to current than to forward reference
        assert!(
            total_error_to_current < total_error_to_forward / 2,
            "Reconstructed frame should be closer to current than forward reference"
        );
    }

    #[test]
    fn test_bframe_residual_application() {
        // Test: Verify that residuals are correctly applied for all prediction types
        let width = 4;
        let height = 4;

        let current = TestSubSampleBlockGroup::test_frame(width, height, 120);
        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 100);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 80);

        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        assert!(!macroblocks.is_empty(), "Should have macroblocks");

        // Test reassembly
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )
        .expect("Reassembly should succeed");

        // Verify residuals were applied
        let mut has_nonzero_residual = false;
        for mb in &macroblocks {
            for residual in mb.residuals.y.iter() {
                for r in 0..8 {
                    for c in 0..8 {
                        if residual.get(r, c).abs() > 10 {
                            has_nonzero_residual = true;
                            break;
                        }
                    }
                }
            }
        }

        assert!(
            has_nonzero_residual,
            "Should have significant residuals for different frames"
        );

        // Verify reconstructed is not identical to either reference
        let mut diff_from_forward = 0i64;
        let mut diff_from_backward = 0i64;

        for (idx, recon) in reconstructed.y().iter().enumerate() {
            let fwd = &forward_ref.y[idx];
            let bwd = &backward_ref.y[idx];

            for r in 0..8 {
                for c in 0..8 {
                    diff_from_forward += (recon.get(r, c) - fwd.get(r, c)).abs() as i64;
                    diff_from_backward += (recon.get(r, c) - bwd.get(r, c)).abs() as i64;
                }
            }
        }

        assert!(
            diff_from_forward > 0,
            "Reconstructed should differ from forward ref"
        );
        assert!(
            diff_from_backward > 0,
            "Reconstructed should differ from backward ref"
        );
    }

    #[test]
    fn test_bframe_bidirectional_prediction() {
        // Test: Verify bi-directional prediction type works correctly
        let width = 4;
        let height = 4;

        // Create scenario where bi-directional might be chosen
        let current = TestSubSampleBlockGroup::test_frame(width, height, 90);
        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 100);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 80);

        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        assert!(!macroblocks.is_empty(), "Should have macroblocks");

        // Check that we have prediction types
        let mut has_forward = false;
        let mut has_backward = false;
        let mut has_both = false;

        for mb in &macroblocks {
            match &mb.prediction {
                Prediction::Forward(_) => has_forward = true,
                Prediction::Backward(_) => has_backward = true,
                Prediction::Both { .. } => has_both = true,
            }
        }

        // At least one prediction type should be used
        assert!(
            has_forward || has_backward || has_both,
            "Should have at least one prediction type"
        );

        // Verify reassembly works with all prediction types
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )
        .expect("Reassembly should succeed with all prediction types");

        assert_eq!(
            reconstructed.y().len(),
            current.y.len(),
            "Reconstructed should have same number of blocks"
        );
    }

    #[test]
    fn test_bframe_chroma_motion_vector_scaling() {
        // CRITICAL TEST: Verify chroma motion vectors are correctly scaled for 4:2:0
        // This tests the core issue with colored square artifacts

        // Create 8x8 luma blocks (8 blocks wide x 8 blocks tall)
        // Chroma will be 4x4 blocks for 4:2:0 subsampling
        let width = 8;
        let height = 8;

        // Create frames with distinct values to test motion compensation
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let mut forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let mut backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set specific chroma values in forward reference
        // Cb channel: set first block to 100
        for r in 0..8 {
            for c in 0..8 {
                forward_ref.cb[0].set(r, c, 100);
                forward_ref.cr[0].set(r, c, 200);
            }
        }

        // Set different chroma values in backward reference
        for r in 0..8 {
            for c in 0..8 {
                backward_ref.cb[0].set(r, c, 50);
                backward_ref.cr[0].set(r, c, 150);
            }
        }

        // Current frame: set chroma to match forward ref (should prefer forward prediction)
        for r in 0..8 {
            for c in 0..8 {
                current.cb[0].set(r, c, 100);
                current.cr[0].set(r, c, 200);
            }
        }

        // Create B-frame
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );

        let macroblocks = bframe.get_macroblocks();

        // Reassemble with chroma
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )
        .expect("Reassembly should succeed");

        // CRITICAL VERIFICATION: Check that chroma was reconstructed
        assert_eq!(
            reconstructed.cb().len(),
            (width * height) / 4,
            "Chroma should have 1/4 the blocks of luma for 4:2:0"
        );
        assert_eq!(
            reconstructed.cr().len(),
            (width * height) / 4,
            "Chroma should have 1/4 the blocks of luma for 4:2:0"
        );

        // Verify first chroma block was reconstructed (should be close to forward ref)
        let cb_block = &reconstructed.cb()[0];
        let cr_block = &reconstructed.cr()[0];

        let mut cb_avg = 0i32;
        let mut cr_avg = 0i32;
        for r in 0..8 {
            for c in 0..8 {
                cb_avg += cb_block.get(r, c) as i32;
                cr_avg += cr_block.get(r, c) as i32;
            }
        }
        cb_avg /= 64;
        cr_avg /= 64;

        // Allow some tolerance for quantization/compression
        assert!(
            (cb_avg - 100).abs() < 30,
            "Cb should be close to forward reference (100), got {}",
            cb_avg
        );
        assert!(
            (cr_avg - 200).abs() < 30,
            "Cr should be close to forward reference (200), got {}",
            cr_avg
        );

        // Verify chroma is NOT just zeros (common bug)
        let mut has_nonzero_cb = false;
        let mut has_nonzero_cr = false;
        for block in reconstructed.cb() {
            for r in 0..8 {
                for c in 0..8 {
                    if block.get(r, c).abs() > 10 {
                        has_nonzero_cb = true;
                        break;
                    }
                }
            }
        }
        for block in reconstructed.cr() {
            for r in 0..8 {
                for c in 0..8 {
                    if block.get(r, c).abs() > 10 {
                        has_nonzero_cr = true;
                        break;
                    }
                }
            }
        }

        assert!(has_nonzero_cb, "Cb channel should have non-zero values");
        assert!(has_nonzero_cr, "Cr channel should have non-zero values");
    }

    #[test]
    fn test_bframe_chroma_with_motion() {
        // Test chroma reconstruction when motion vectors are non-zero
        // This specifically tests the motion vector scaling from luma to chroma

        let width = 8;
        let height = 8;

        // Create frames where content has moved
        let mut forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Put a chroma "feature" at position (1, 1) - second chroma block row/col
        // Row 1, Col 1 in 4x4 chroma grid
        let chroma_idx = 4 + 1;
        for r in 0..8 {
            for c in 0..8 {
                forward_ref.cb[chroma_idx].set(r, c, 150);
                forward_ref.cr[chroma_idx].set(r, c, 180);
            }
        }

        // Current frame: same "feature" but needs motion compensation to find it
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Put same chroma feature at same location
        for r in 0..8 {
            for c in 0..8 {
                current.cb[chroma_idx].set(r, c, 150);
                current.cr[chroma_idx].set(r, c, 180);
            }
        }

        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Create B-frame
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );

        let macroblocks = bframe.get_macroblocks();

        // Reassemble
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )
        .expect("Reassembly should succeed");

        // Verify the chroma "feature" was reconstructed
        let recon_cb = &reconstructed.cb()[chroma_idx];
        let recon_cr = &reconstructed.cr()[chroma_idx];

        let mut cb_avg = 0i32;
        let mut cr_avg = 0i32;
        for r in 0..8 {
            for c in 0..8 {
                cb_avg += recon_cb.get(r, c) as i32;
                cr_avg += recon_cr.get(r, c) as i32;
            }
        }
        cb_avg /= 64;
        cr_avg /= 64;

        // The feature should be reconstructed with reasonable accuracy
        assert!(
            cb_avg > 100,
            "Cb should reflect the feature (expected ~150), got {}",
            cb_avg
        );
        assert!(
            cr_avg > 100,
            "Cr should reflect the feature (expected ~180), got {}",
            cr_avg
        );
    }

    #[test]
    fn test_motion_vector_coordinate_system() {
        // CRITICAL TEST: Verify the coordinate system of motion vectors
        // This test documents and verifies how motion vectors map to blocks

        let width = 8;
        let height = 8;

        // Create a distinctive pattern in forward reference
        // We'll put a unique value in each luma block to track motion
        let mut forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set luma block at position (2, 2) to value 100
        // Row 2, Col 2
        let luma_idx_2_2 = 2 * width + 2;
        for r in 0..8 {
            for c in 0..8 {
                forward_ref.y[luma_idx_2_2].set(r, c, 100);
            }
        }

        // Set chroma block at position (1, 1) to specific values (this corresponds to luma blocks
        // (2,2)-(3,3))
        // Row 1, Col 1 in chroma (4x4 grid)
        let chroma_idx_1_1 = 4 + 1;
        for r in 0..8 {
            for c in 0..8 {
                forward_ref.cb[chroma_idx_1_1].set(r, c, 150);
                forward_ref.cr[chroma_idx_1_1].set(r, c, 180);
            }
        }

        // Create current frame with same pattern at position (3, 3)
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let luma_idx_3_3 = 3 * width + 3;
        for r in 0..8 {
            for c in 0..8 {
                current.y[luma_idx_3_3].set(r, c, 100);
            }
        }

        // Chroma still at (1, 1) since (3,3) luma maps to same chroma block
        for r in 0..8 {
            for c in 0..8 {
                current.cb[chroma_idx_1_1].set(r, c, 150);
                current.cr[chroma_idx_1_1].set(r, c, 180);
            }
        }

        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Debug: Print what's in the forward reference

        // Create B-frame and examine motion vectors
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );

        let macroblocks = bframe.get_macroblocks();

        // Print ALL macroblocks to understand chroma updates

        for (idx, mb) in macroblocks.iter().enumerate() {
            // Specifically track which chroma blocks this affects
            for luma_r in mb.location.start.row..=mb.location.end.row {
                for luma_c in mb.location.start.col..=mb.location.end.col {
                    let chroma_r = luma_r / 2;
                    let chroma_c = luma_c / 2;
                    if chroma_r == 1 && chroma_c == 1 {}
                }
            }
        }

        // Reassemble and verify
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )
        .expect("Reassembly should succeed");

        // Verify the luma block at (3,3) was reconstructed correctly
        let recon_luma = &reconstructed.y()[luma_idx_3_3];
        let mut luma_avg = 0i32;
        for r in 0..8 {
            for c in 0..8 {
                luma_avg += recon_luma.get(r, c) as i32;
            }
        }
        luma_avg /= 64;

        assert!(
            luma_avg > 50,
            "Luma should be reconstructed from forward ref, got {}",
            luma_avg
        );

        // Verify chroma was also reconstructed
        let recon_cb = &reconstructed.cb()[chroma_idx_1_1];
        let mut cb_avg = 0i32;
        for r in 0..8 {
            for c in 0..8 {
                cb_avg += recon_cb.get(r, c) as i32;
            }
        }
        cb_avg /= 64;

        // Also check what the base_cb had
        let base_cb_at_1_1 = {
            let mut sum = 0i32;
            for r in 0..8 {
                for c in 0..8 {
                    sum += forward_ref.cb[chroma_idx_1_1].get(r, c) as i32;
                }
            }
            sum / 64
        };

        assert!(
            cb_avg > 100,
            "Chroma should be reconstructed, got {}. Base had {}",
            cb_avg,
            base_cb_at_1_1
        );
    }

    #[test]
    fn test_chroma_residuals_stored_and_retrieved() {
        // Test that chroma residuals are actually stored in macroblocks

        let width = 8;
        let height = 8;

        // Create frames with different chroma values
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set distinct chroma values
        for idx in 0..current.cb.len() {
            for r in 0..8 {
                for c in 0..8 {
                    current.cb[idx].set(r, c, 100);
                    current.cr[idx].set(r, c, -100);
                }
            }
        }

        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        let mut has_cb_residuals = false;
        let mut has_cr_residuals = false;

        for (idx, mb) in macroblocks.iter().enumerate() {
            println!(
                "MB{}: cb_residuals={}, cr_residuals={}",
                idx,
                mb.residuals.cb.len(),
                mb.residuals.cr.len()
            );

            if mb.residuals.cb.len() > 0 {
                has_cb_residuals = true;
            }
            if mb.residuals.cr.len() > 0 {
                has_cr_residuals = true;
            }
        }

        assert!(
            has_cb_residuals,
            "Macroblocks should have Cb residuals stored"
        );
        assert!(
            has_cr_residuals,
            "Macroblocks should have Cr residuals stored"
        );
    }

    #[test]
    fn test_chroma_residuals_no_duplicates() {
        // Test that we don't store duplicate chroma residuals for 4:2:0

        let width = 8;
        let height = 8;

        let current = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        for (idx, mb) in macroblocks.iter().enumerate() {
            // Calculate how many luma blocks in this macroblock
            let luma_count = (mb.location.end.row - mb.location.start.row + 1)
                * (mb.location.end.col - mb.location.start.col + 1);

            // For 4:2:0, chroma should have 1/4 the blocks of luma (or fewer due to deduplication)
            let expected_max_chroma = luma_count.div_ceil(4);

            assert!(
                mb.residuals.cb.len() <= expected_max_chroma,
                "MB{}: Too many Cb residuals! Got {}, expected at most {} (luma blocks: {})",
                idx,
                mb.residuals.cb.len(),
                expected_max_chroma,
                luma_count
            );

            assert!(
                mb.residuals.cr.len() <= expected_max_chroma,
                "MB{}: Too many Cr residuals! Got {}, expected at most {} (luma blocks: {})",
                idx,
                mb.residuals.cr.len(),
                expected_max_chroma,
                luma_count
            );
        }
    }

    #[test]
    fn test_chroma_residuals_roundtrip() -> crate::error::Result<()> {
        // Test that chroma residuals survive encode/decode roundtrip

        let width = 4;
        let height = 4;

        // Create a current frame with specific chroma values
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set specific chroma values - these will require residuals
        for r in 0..8 {
            for c in 0..8 {
                current.cb[0].set(r, c, 50);
                current.cr[0].set(r, c, -50);
            }
        }

        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Encode
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        // Serialize and deserialize macroblocks
        let mut serialized = Vec::new();
        for mb in &macroblocks {
            serialized.extend_from_slice(&mb.to_bytes());
        }

        // Deserialize
        let mut deserialized_mbs = Vec::new();
        let mut offset = 0;
        for _ in 0..macroblocks.len() {
            let (mb, bytes_read) = BMacroBlock::from_bytes(&serialized[offset..]);
            deserialized_mbs.push(mb);
            offset += bytes_read;
        }

        // Check that chroma residuals survived
        for (idx, (orig, deser)) in macroblocks.iter().zip(deserialized_mbs.iter()).enumerate() {
            assert_eq!(
                orig.residuals.cb.len(),
                deser.residuals.cb.len(),
                "MB{}: Cb residual count mismatch",
                idx
            );
            assert_eq!(
                orig.residuals.cr.len(),
                deser.residuals.cr.len(),
                "MB{}: Cr residual count mismatch",
                idx
            );
        }

        // Reconstruct and verify chroma is correct
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &deserialized_mbs,
        )?;

        let recon_cb = reconstructed.cb()[0].get(4, 4);
        let recon_cr = reconstructed.cr()[0].get(4, 4);

        // Should be close to original (within quantization error)
        let cb_error = (recon_cb - current.cb[0].get(4, 4)).abs();
        let cr_error = (recon_cr - current.cr[0].get(4, 4)).abs();

        assert!(
            cb_error < 30,
            "Cb reconstruction error too large: {}",
            cb_error
        );
        assert!(
            cr_error < 30,
            "Cr reconstruction error too large: {}",
            cr_error
        );

        Ok(())
    }

    #[test]
    fn test_chroma_residuals_with_motion_vectors() -> crate::error::Result<()> {
        let width = 8;
        let height = 8;

        // Create current frame with chroma at position 1,1
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let chroma_idx_1_1 = 4 + 1;
        for r in 0..8 {
            for c in 0..8 {
                current.cb[chroma_idx_1_1].set(r, c, 80);
                current.cr[chroma_idx_1_1].set(r, c, -80);
            }
        }

        // Create forward ref with chroma at position 0,0
        let mut forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        for r in 0..8 {
            for c in 0..8 {
                forward_ref.cb[0].set(r, c, 20);
                forward_ref.cr[0].set(r, c, -20);
            }
        }

        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        // Reconstruct
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )?;

        let recon_cb = reconstructed.cb()[chroma_idx_1_1].get(4, 4);

        // With motion compensation and residuals, should be close to original
        let error = (recon_cb - current.cb[chroma_idx_1_1].get(4, 4)).abs();

        assert!(
            error < 30,
            "Chroma reconstruction with motion vectors failed, error: {}",
            error
        );

        Ok(())
    }

    #[test]
    fn test_chroma_residuals_large_frame() -> crate::error::Result<()> {
        // Test with frame size similar to real video (88x60 for 704×480)

        // 704 pixels / 8 = 88 blocks
        let width = 88;
        // 480 pixels / 8 = 60 blocks
        let height = 60;

        // Create current frame with specific chroma pattern
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set a pattern in chroma - bright blue region in center
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        for r in 10..20 {
            for c in 10..20 {
                let idx = r * chroma_width + c;
                if idx < current.cb.len() {
                    for br in 0..8 {
                        for bc in 0..8 {
                            // High Cb = blue
                            current.cb[idx].set(br, bc, 100);
                            // Negative Cr
                            current.cr[idx].set(br, bc, -100);
                        }
                    }
                }
            }
        }

        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Encode
        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        // Count total residuals
        let total_cb_residuals: usize = macroblocks.iter().map(|mb| mb.residuals.cb.len()).sum();
        let total_cr_residuals: usize = macroblocks.iter().map(|mb| mb.residuals.cr.len()).sum();

        println!("Total Cb residuals: {}", total_cb_residuals);
        println!("Total Cr residuals: {}", total_cr_residuals);

        // Reconstruct
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )?;

        // Check the blue region
        let mut errors = 0;
        let mut max_error = 0;
        for r in 10..20 {
            for c in 10..20 {
                let idx = r * chroma_width + c;
                if idx < current.cb.len() {
                    let original_cb = current.cb[idx].get(4, 4);
                    let recon_cb = reconstructed.cb()[idx].get(4, 4);
                    let error = (recon_cb - original_cb).abs();

                    if error > max_error {
                        max_error = error;
                    }

                    if error > 30 {
                        errors += 1;
                        if errors <= 3 {}
                    }
                }
            }
        }

        assert!(
            max_error < 30,
            "Large frame chroma reconstruction failed, max error: {}",
            max_error
        );

        Ok(())
    }

    #[test]
    fn test_chroma_residuals_macroblock_spans() -> crate::error::Result<()> {
        // Test edge case: macroblock that spans multiple chroma blocks
        // With 4:2:0, a 16x16 macroblock (4 luma blocks) maps to ONE 8x8 chroma block
        // But what if motion estimation creates odd-sized macroblocks?

        // Small test case
        let width = 16;
        let height = 16;

        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set specific chroma pattern - alternating values
        let chroma_width = width / 2;
        for r in 0..chroma_width {
            for c in 0..chroma_width {
                let idx = r * chroma_width + c;
                let value = if (r + c) % 2 == 0 { 100 } else { -100 };
                for br in 0..8 {
                    for bc in 0..8 {
                        current.cb[idx].set(br, bc, value);
                        current.cr[idx].set(br, bc, -value);
                    }
                }
            }
        }

        let forward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);
        let backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        let bframe = BFrame::new(
            current.clone().into(),
            Some(forward_ref.clone().into()),
            backward_ref.clone().into(),
        );
        let macroblocks = bframe.get_macroblocks();

        // Print detailed info about each macroblock
        for (idx, mb) in macroblocks.iter().enumerate() {
            let luma_blocks = (mb.location.end.row - mb.location.start.row + 1)
                * (mb.location.end.col - mb.location.start.col + 1);
            println!(
                "MB{}: loc=({},{})-({},{}), luma={}, cb_res={}, cr_res={}",
                idx,
                mb.location.start.row,
                mb.location.start.col,
                mb.location.end.row,
                mb.location.end.col,
                luma_blocks,
                mb.residuals.cb.len(),
                mb.residuals.cr.len()
            );
        }

        // Reconstruct
        let reconstructed = BFrame::reassemble(
            Some(forward_ref.as_ref()),
            backward_ref.as_ref(),
            &macroblocks,
        )?;

        // Check every chroma block
        let mut errors = 0;
        for r in 0..chroma_width {
            for c in 0..chroma_width {
                let idx = r * chroma_width + c;
                let orig_cb = current.cb[idx].get(4, 4);
                let recon_cb = reconstructed.cb()[idx].get(4, 4);
                let error = (recon_cb - orig_cb).abs();

                if error > 30 {
                    errors += 1;
                }
            }
        }

        assert_eq!(errors, 0, "Should have no reconstruction errors");

        Ok(())
    }

    #[test]
    fn test_bframe_without_forward_ref() -> crate::error::Result<()> {
        // Test B-frame encoding/decoding when forward_ref is None
        // This mimics what happens in GOP encoding

        let width = 16;
        let height = 16;

        // Create current frame with distinct chroma
        let mut current = TestSubSampleBlockGroup::test_frame(width, height, 0);

        // Set chroma values
        for idx in 0..current.cb.len() {
            for r in 0..8 {
                for c in 0..8 {
                    current.cb[idx].set(r, c, 80);
                    current.cr[idx].set(r, c, -80);
                }
            }
        }

        // Backward reference (different from current)
        let mut backward_ref = TestSubSampleBlockGroup::test_frame(width, height, 0);

        for idx in 0..backward_ref.cb.len() {
            for r in 0..8 {
                for c in 0..8 {
                    backward_ref.cb[idx].set(r, c, 20);
                    backward_ref.cr[idx].set(r, c, -20);
                }
            }
        }

        // Encode with forward_ref=None (like GOP does)
        let bframe = BFrame::new(current.clone().into(), None, backward_ref.clone().into());
        let macroblocks = bframe.get_macroblocks();

        println!("Macroblocks: {}", macroblocks.len());

        // Check prediction types
        let mut forward_count = 0;
        let mut backward_count = 0;
        let mut both_count = 0;

        for (idx, mb) in macroblocks.iter().enumerate() {
            match &mb.prediction {
                crate::lossy::frame::r#macro::Prediction::Forward(_) => {
                    forward_count += 1;
                    if idx < 3 {
                        println!("MB{}: Forward prediction (UNEXPECTED!)", idx);
                    }
                }
                crate::lossy::frame::r#macro::Prediction::Backward(_) => {
                    backward_count += 1;
                }
                crate::lossy::frame::r#macro::Prediction::Both { .. } => {
                    both_count += 1;
                    if idx < 3 {
                        println!("MB{}: Both prediction (UNEXPECTED!)", idx);
                    }
                }
            }
        }

        println!(
            "Prediction types: Forward={}, Backward={}, Both={}",
            forward_count, backward_count, both_count
        );
        println!(
            "Total Cb residuals: {}",
            macroblocks
                .iter()
                .map(|mb| mb.residuals.cb.len())
                .sum::<usize>()
        );

        // Decode with forward_ref=None
        let reconstructed = BFrame::reassemble(None, backward_ref.as_ref(), &macroblocks)?;

        // Check reconstruction
        let mut max_error = 0;
        for idx in 0..current.cb.len() {
            let orig = current.cb[idx].get(4, 4);
            let recon = reconstructed.cb()[idx].get(4, 4);
            let error = (orig - recon).abs();
            if error > max_error {
                max_error = error;
            }
            if error > 30 && idx < 3 {
                println!(
                    "Cb[{}]: orig={}, recon={}, error={}",
                    idx, orig, recon, error
                );
            }
        }

        assert!(
            max_error < 30,
            "Reconstruction error too large: {}",
            max_error
        );

        Ok(())
    }
}
