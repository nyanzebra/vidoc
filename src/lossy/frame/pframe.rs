use std::io::{Read, Write};

use rayon::prelude::*;

use crate::{
    block::Block,
    dimensions::BlockDimensions,
    lossy::{frame::motion_vector::depth16, SubSampleBlockGroup, SubSampleBlockGroupRef},
    point::Point,
    BitStreamReader, BitStreamWriter, Decodable, Encodable, Result,
};

use super::{
    build_predicted_blocks, calculate_residuals_for_macroblock, compressed_motion_vectors,
    motion_vector::MotionVector,
    r#macro::{PMacroBlock, PMacroBlocks, Prediction},
    reassemble_frame,
};

pub struct PFrame<'a, T> {
    current: SubSampleBlockGroupRef<'a, T>,
    previous: SubSampleBlockGroupRef<'a, T>,
}

impl<'a, T> PFrame<'a, T> {
    pub(crate) fn new(
        current: SubSampleBlockGroupRef<'a, T>,
        previous: SubSampleBlockGroupRef<'a, T>,
    ) -> Self {
        PFrame { current, previous }
    }
}

impl Encodable for PFrame<'_, i16> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let macroblocks = self.get_macroblocks();
        PMacroBlocks::new(macroblocks).encode(stream)?;

        Ok(())
    }
}

impl PFrame<'_, i16> {
    pub(crate) fn reassemble(
        previous_frame: &SubSampleBlockGroupRef<'_, i16>,
        macro_blocks: &[PMacroBlock<i16>],
    ) -> Result<SubSampleBlockGroup<i16>> {
        reassemble_frame(None, previous_frame, macro_blocks)
    }

    pub(crate) fn get_macroblocks(&self) -> Vec<PMacroBlock<i16>> {
        let motion_vecs = self.motion_vectors(&self.current.dimensions, self.previous.y);
        let compressed = compressed_motion_vectors(&motion_vecs, &self.current.dimensions);

        // build_predicted_blocks and calculate_residuals_for_macroblock are pure
        // functions of their inputs — parallelize across macroblocks.
        compressed
            .into_par_iter()
            .map(|((prediction, _score), location)| {
                let mv = match prediction {
                    Prediction::Backward(mv) => mv,
                    _ => unreachable!("P-frames should only have Backward prediction"),
                };

                let (predicted_y, predicted_cb, predicted_cr) = build_predicted_blocks(
                    &location,
                    &prediction,
                    &self.current.dimensions,
                    self.current.subsampling,
                    None,
                    &self.previous,
                );

                let residuals = calculate_residuals_for_macroblock(
                    &location,
                    &self.current,
                    &predicted_y,
                    &predicted_cb,
                    &predicted_cr,
                );

                PMacroBlock {
                    location,
                    mv,
                    residuals,
                }
            })
            .collect()
    }

    fn motion_vectors(
        &self,
        dimensions: &BlockDimensions,
        channel: &[Block<i16>],
    ) -> Vec<Vec<(Prediction, i16)>> {
        (0..dimensions.height)
            .into_par_iter()
            .map(|row| {
                (0..dimensions.width)
                    .map(|col| {
                        let idx = row * dimensions.width + col;
                        if idx < self.current.y.len() && idx < channel.len() {
                            let current = &self.current.y[idx];
                            let (mv, cost) = depth16::ldsp_blocks(
                                current,
                                channel,
                                dimensions,
                                Point { row, col },
                            );
                            (Prediction::Backward(mv), cost)
                        } else {
                            (Prediction::Backward(MotionVector { x: 0, y: 0 }), 0)
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

impl Decodable for PFrame<'_, i16> {
    type Output = PMacroBlocks<i16>;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        PMacroBlocks::decode(stream)
    }
}

impl PFrame<'_, i16> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        block::Block, color::Subsampling, dimensions::BlockDimensions, lossy::SubSampleBlockGroup,
    };

    fn create_test_frame(width: usize, height: usize, fill_value: i16) -> SubSampleBlockGroup<i16> {
        let block_dims = BlockDimensions { width, height };

        let num_blocks = width * height;
        let mut blocks = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            let mut block = Block::<i16>::default();
            for r in 0..8 {
                for c in 0..8 {
                    block.set(r, c, fill_value);
                }
            }
            blocks.push(block);
        }

        // For 4:2:0, chroma is half resolution in both dimensions
        let chroma_blocks = (width / 2) * (height / 2);
        let mut cb_blocks = Vec::with_capacity(chroma_blocks);
        let mut cr_blocks = Vec::with_capacity(chroma_blocks);

        for _ in 0..chroma_blocks {
            cb_blocks.push(Block::<i16>::default());
            cr_blocks.push(Block::<i16>::default());
        }

        SubSampleBlockGroup {
            dimensions: block_dims,
            subsampling: Subsampling::Sample420,
            y: blocks,
            cb: cb_blocks,
            cr: cr_blocks,
        }
    }

    #[test]
    fn test_pframe_encode_decode_identical_frames() {
        // Test: When current and previous frames are identical,
        // motion vectors should be (0,0) and residuals should be near zero
        let width = 8;
        let height = 8;
        let current = create_test_frame(width, height, 100);
        let previous = create_test_frame(width, height, 100);

        // Create P-frame
        let pframe = PFrame::new(current.as_ref(), previous.as_ref());

        // Get macroblocks
        let macroblocks = pframe.get_macroblocks();

        println!("Test: Identical frames");
        println!("Number of macroblocks: {}", macroblocks.len());

        // Verify we have macroblocks
        assert!(
            !macroblocks.is_empty(),
            "Should have at least one macroblock"
        );

        // Check that motion vectors are (0,0) for identical frames
        for (idx, mb) in macroblocks.iter().enumerate() {
            println!(
                "Macroblock {}: MV=({}, {}), Y Residuals={}",
                idx,
                mb.mv.x,
                mb.mv.y,
                mb.residuals.y.len()
            );
            assert_eq!(
                mb.mv.x, 0,
                "Motion vector X should be 0 for identical frames"
            );
            assert_eq!(
                mb.mv.y, 0,
                "Motion vector Y should be 0 for identical frames"
            );
        }

        // Reassemble and verify
        let reconstructed = PFrame::reassemble(&previous.as_ref(), &macroblocks)
            .expect("Reassembly should succeed");

        println!(
            "Reconstructed: {}x{} blocks",
            reconstructed.dimensions.width, reconstructed.dimensions.height
        );

        // Verify dimensions match
        assert_eq!(reconstructed.dimensions.width, current.dimensions.width);
        assert_eq!(reconstructed.dimensions.height, current.dimensions.height);

        // Verify Y blocks are close to original (allowing for quantization error)
        for (idx, (orig, recon)) in current.y.iter().zip(reconstructed.y.iter()).enumerate() {
            for r in 0..8 {
                for c in 0..8 {
                    let orig_val = orig.get(r, c);
                    let recon_val = recon.get(r, c);
                    let diff = (orig_val - recon_val).abs();
                    assert!(
                        diff < 10,
                        "Block {} at ({},{}) differs too much: {} vs {} (diff={})",
                        idx,
                        r,
                        c,
                        orig_val,
                        recon_val,
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_pframe_encode_decode_different_frames() {
        // Test: When frames differ, residuals should capture the difference
        let width = 8;
        let height = 8;
        let current = create_test_frame(width, height, 150);
        let previous = create_test_frame(width, height, 100);

        // Create P-frame
        let pframe = PFrame::new(current.as_ref(), previous.as_ref());

        // Get macroblocks
        let macroblocks = pframe.get_macroblocks();

        assert!(
            !macroblocks.is_empty(),
            "Should have at least one macroblock"
        );

        // Reassemble
        let reconstructed = PFrame::reassemble(&previous.as_ref(), &macroblocks)
            .expect("Reassembly should succeed");

        // Verify the reconstructed frame is closer to current than to previous
        let mut total_error_to_current = 0i64;
        let mut total_error_to_previous = 0i64;

        for (idx, recon) in reconstructed.y.iter().enumerate() {
            let curr = &current.y[idx];
            let prev = &previous.y[idx];

            for r in 0..8 {
                for c in 0..8 {
                    let recon_val = recon.get(r, c);
                    let curr_val = curr.get(r, c);
                    let prev_val = prev.get(r, c);

                    total_error_to_current += (recon_val - curr_val).abs() as i64;
                    total_error_to_previous += (recon_val - prev_val).abs() as i64;
                }
            }
        }

        // Reconstructed should be much closer to current than to previous
        assert!(
            total_error_to_current < total_error_to_previous / 2,
            "Reconstructed frame should be closer to current than previous"
        );
    }

    #[test]
    fn test_pframe_residual_application() {
        // Test: Verify that residuals are correctly applied during reassembly
        let width = 4;
        let height = 4;

        // Create a simple test case where we can manually verify
        let mut current = create_test_frame(width, height, 0);
        let mut previous = create_test_frame(width, height, 0);

        // Set specific values in first block
        for r in 0..8 {
            for c in 0..8 {
                current.y[0].set(r, c, 100);
                previous.y[0].set(r, c, 50);
            }
        }

        println!("\nTest: Residual application");
        println!("Current block[0]: all 100");
        println!("Previous block[0]: all 50");
        println!("Expected residual: ~50 (after DCT/IDCT)");

        let pframe = PFrame::new(current.as_ref(), previous.as_ref());
        let macroblocks = pframe.get_macroblocks();

        // Check that we have residuals
        assert!(!macroblocks.is_empty(), "Should have macroblocks");
        assert!(
            macroblocks[0].residuals.y.len() > 0,
            "Should have Y residuals"
        );

        println!(
            "First macroblock has {} Y residual blocks",
            macroblocks[0].residuals.y.len()
        );

        // Reassemble
        let reconstructed = PFrame::reassemble(&previous.as_ref(), &macroblocks)
            .expect("Reassembly should succeed");

        // Check first block is closer to 100 than to 50
        let mut sum = 0i32;
        for r in 0..8 {
            for c in 0..8 {
                sum += reconstructed.y[0].get(r, c) as i32;
            }
        }
        let recon_avg = (sum / 64) as i16;

        println!("Reconstructed block[0] average: {}", recon_avg);

        let error_to_current = (recon_avg - 100).abs();
        let error_to_previous = (recon_avg - 50).abs();

        println!("Error to current (100): {}", error_to_current);
        println!("Error to previous (50): {}", error_to_previous);

        assert!(
            error_to_current < error_to_previous,
            "Reconstructed should be closer to current than previous"
        );
    }

    #[test]
    fn test_pframe_motion_vector_calculation() {
        // Test: Verify motion vectors are calculated when content moves
        let width = 8;
        let height = 8;

        let current = create_test_frame(width, height, 100);
        let previous = create_test_frame(width, height, 50);

        let pframe = PFrame::new(current.as_ref(), previous.as_ref());
        let macroblocks = pframe.get_macroblocks();

        println!("\nTest: Motion vector calculation");

        for (idx, mb) in macroblocks.iter().enumerate() {
            println!(
                "Macroblock {}: Location ({},{}) to ({},{}), MV=({},{}), Y Residuals={}",
                idx,
                mb.location.start.row,
                mb.location.start.col,
                mb.location.end.row,
                mb.location.end.col,
                mb.mv.x,
                mb.mv.y,
                mb.residuals.y.len()
            );

            // Verify residuals exist
            assert!(
                mb.residuals.y.len() > 0,
                "Macroblock {} should have Y residuals",
                idx
            );

            // Verify location is within bounds
            assert!(mb.location.end.row < height, "End row out of bounds");
            assert!(mb.location.end.col < width, "End col out of bounds");
        }
    }
}
