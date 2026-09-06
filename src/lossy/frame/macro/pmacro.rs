use std::io::{Read, Write};

use super::{AssemblableMacroBlock, BlockLocation, Prediction, Residuals};
use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    encoders::ans,
    error::Result,
    lossy::frame::{
        motion_vector::MOTION_VECTOR_SIZE, r#macro::block_location::BLOCK_LOCATION_SIZE,
        MotionVector,
    },
    Decodable, Encodable, FromBytes, ToBytes,
};

#[repr(C)]
pub(crate) struct PMacroBlock {
    pub(crate) location: BlockLocation,
    pub(crate) mv: MotionVector,
    pub(crate) residuals: Residuals,
}

impl AssemblableMacroBlock for PMacroBlock {
    fn location(&self) -> &BlockLocation {
        &self.location
    }

    fn prediction(&self) -> Prediction {
        // P-frames always use Backward prediction
        Prediction::Backward(self.mv)
    }

    fn residuals(&self) -> &Residuals {
        &self.residuals
    }
}

impl Encodable for PMacroBlock {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.location.encode(stream)?;
        self.mv.encode(stream)?;
        self.residuals.encode(stream)?;

        Ok(())
    }
}

impl Decodable for PMacroBlock {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let location = BlockLocation::decode(stream)?;
        let mv = MotionVector::decode(stream)?;
        let residuals = Residuals::decode(stream)?;
        Ok(Self {
            location,
            mv,
            residuals,
        })
    }
}

impl ToBytes for PMacroBlock {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        let location: [u8; _] = self.location.into();
        bytes.extend(location);
        let mv: [u8; _] = self.mv.into();
        bytes.extend(mv);
        bytes.extend_from_slice(&self.residuals.to_bytes());

        bytes
    }
}

impl FromBytes for PMacroBlock {
    fn from_bytes(bytes: &[u8]) -> (Self, usize) {
        let location =
            BlockLocation::try_from(&bytes[0..BLOCK_LOCATION_SIZE]).expect("block location");
        let mv = MotionVector::try_from(
            &bytes[BLOCK_LOCATION_SIZE..BLOCK_LOCATION_SIZE + MOTION_VECTOR_SIZE],
        )
        .expect("motion vector");

        let offset = BLOCK_LOCATION_SIZE + MOTION_VECTOR_SIZE;
        let (residuals, size) = Residuals::from_bytes(&bytes[offset..]);

        (
            Self {
                location,
                mv,
                residuals,
            },
            offset + size,
        )
    }
}

pub(crate) struct PMacroBlocks(Vec<PMacroBlock>);

impl PMacroBlocks {
    pub(crate) fn new(blocks: Vec<PMacroBlock>) -> Self {
        Self(blocks)
    }

    pub(crate) fn into_inner(self) -> Vec<PMacroBlock> {
        self.0
    }
}

impl Decodable for PMacroBlocks {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        Ok(Self(ans::decode(stream)?))
    }
}

impl Encodable for PMacroBlocks {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        ans::encode(&self.0, stream)
    }
}

pub(crate) struct PMacroBlocksRef<'a>(&'a [PMacroBlock]);

impl<'a> PMacroBlocksRef<'a> {
    pub(crate) fn new(blocks: &'a [PMacroBlock]) -> Self {
        Self(blocks)
    }
}

impl Encodable for PMacroBlocksRef<'_> {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        ans::encode(self.0, stream)
    }
}
