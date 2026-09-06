use std::{
    fmt::Debug,
    io::{Read, Write},
};

use super::{AssemblableMacroBlock, BlockLocation};
use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    encoders::ans,
    error::Result,
    lossy::frame::{
        motion_vector::MOTION_VECTOR_SIZE,
        r#macro::{block_location::BLOCK_LOCATION_SIZE, Residuals},
        MotionVector,
    },
    Decodable, Encodable, FromBytes, ToBytes,
};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Prediction {
    Forward(MotionVector),
    Backward(MotionVector),
    Both {
        forward: MotionVector,
        backward: MotionVector,
    },
}

impl From<Prediction> for u8 {
    fn from(prediction: Prediction) -> Self {
        match prediction {
            Prediction::Forward(_) => 0,
            Prediction::Backward(_) => 1,
            Prediction::Both { .. } => 2,
        }
    }
}

impl From<u8> for Prediction {
    fn from(value: u8) -> Self {
        match value {
            0 => Prediction::Forward(MotionVector::default()),
            1 => Prediction::Backward(MotionVector::default()),
            2 => Prediction::Both {
                forward: MotionVector::default(),
                backward: MotionVector::default(),
            },
            _ => Prediction::Forward(MotionVector::default()), // Default fallback
        }
    }
}

pub(crate) struct BMacroBlock {
    pub(crate) location: BlockLocation,
    pub(crate) prediction: Prediction,
    pub(crate) residuals: Residuals,
}

impl AssemblableMacroBlock for BMacroBlock {
    fn location(&self) -> &BlockLocation {
        &self.location
    }

    fn prediction(&self) -> Prediction {
        self.prediction
    }

    fn residuals(&self) -> &Residuals {
        &self.residuals
    }
}

impl ToBytes for BMacroBlock {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize location
        let location: [u8; _] = self.location.into();
        bytes.extend(location);

        let code = u8::from(self.prediction);
        match self.prediction {
            Prediction::Forward(mv) => {
                bytes.push(code);
                let mv_bytes: [u8; _] = mv.into();
                bytes.extend(mv_bytes);
            }
            Prediction::Backward(mv) => {
                bytes.push(code);
                let mv_bytes: [u8; _] = mv.into();
                bytes.extend(mv_bytes);
            }
            Prediction::Both { forward, backward } => {
                bytes.push(code);
                let forward_bytes: [u8; _] = forward.into();
                bytes.extend(forward_bytes);
                let backward_bytes: [u8; _] = backward.into();
                bytes.extend(backward_bytes);
            }
        }

        bytes.extend(self.residuals.to_bytes());

        bytes
    }
}

impl FromBytes for BMacroBlock {
    fn from_bytes(bytes: &[u8]) -> (Self, usize) {
        let mut offset = 0;

        let location = BlockLocation::try_from(&bytes[offset..offset + BLOCK_LOCATION_SIZE])
            .expect("block location");
        offset += BLOCK_LOCATION_SIZE;

        let prediction_type = bytes[offset];
        offset += 1;

        let (prediction, bytes_consumed) = match prediction_type {
            0 => {
                assert!(bytes.len() >= offset + MOTION_VECTOR_SIZE);
                let mv = MotionVector::try_from(&bytes[offset..offset + MOTION_VECTOR_SIZE])
                    .expect("motion vector");
                (Prediction::Forward(mv), MOTION_VECTOR_SIZE)
            }
            1 => {
                // Backward prediction - read one motion vector
                assert!(bytes.len() >= offset + MOTION_VECTOR_SIZE);
                let mv = MotionVector::try_from(&bytes[offset..offset + MOTION_VECTOR_SIZE])
                    .expect("motion vector");
                (Prediction::Backward(mv), MOTION_VECTOR_SIZE)
            }
            2 => {
                // Both prediction - read two motion vectors
                assert!(bytes.len() >= offset + 2 * MOTION_VECTOR_SIZE);
                let forward = MotionVector::try_from(&bytes[offset..offset + MOTION_VECTOR_SIZE])
                    .expect("forward motion vector");
                let backward = MotionVector::try_from(
                    &bytes[offset + MOTION_VECTOR_SIZE..offset + 2 * MOTION_VECTOR_SIZE],
                )
                .expect("backward motion vector");
                (
                    Prediction::Both { forward, backward },
                    2 * MOTION_VECTOR_SIZE,
                )
            }
            _ => panic!("invalid prediction type"),
        };

        offset += bytes_consumed;

        let (residuals, size) = Residuals::from_bytes(&bytes[offset..]);
        offset += size;

        (
            Self {
                location,
                prediction,
                residuals,
            },
            offset,
        )
    }
}

pub(crate) struct BMacroBlocks(Vec<BMacroBlock>);

impl BMacroBlocks {
    pub(crate) fn new(blocks: Vec<BMacroBlock>) -> Self {
        Self(blocks)
    }

    pub(crate) fn into_inner(self) -> Vec<BMacroBlock> {
        self.0
    }
}

impl Decodable for BMacroBlocks {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self>
    where
        R: Read,
    {
        Ok(Self(ans::decode(stream)?))
    }
}

impl Encodable for BMacroBlocks {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        ans::encode(&self.0, stream)
    }
}

pub(crate) struct BMacroBlocksRef<'a>(&'a [BMacroBlock]);

impl<'a> BMacroBlocksRef<'a> {
    pub(crate) fn new(blocks: &'a [BMacroBlock]) -> Self {
        Self(blocks)
    }
}

impl Encodable for BMacroBlocksRef<'_> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        ans::encode(self.0, stream)
    }
}
