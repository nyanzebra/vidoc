use std::{
    fmt::Debug,
    io::{Read, Write},
};

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
pub(crate) struct PMacroBlock<T> {
    pub(crate) location: BlockLocation,
    pub(crate) mv: MotionVector,
    pub(crate) residuals: Residuals<T>,
}

impl AssemblableMacroBlock for PMacroBlock<i16> {
    fn location(&self) -> &BlockLocation {
        &self.location
    }

    fn prediction(&self) -> Prediction {
        // P-frames always use Backward prediction
        Prediction::Backward(self.mv)
    }

    fn residuals(&self) -> &Residuals<i16> {
        &self.residuals
    }
}

impl<const N: usize, T> Encodable for PMacroBlock<T>
where
    T: Sync + num_traits::ToBytes<Bytes = [u8; N]>,
{
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

impl<const N: usize, T> Decodable for PMacroBlock<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
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

impl<const N: usize, T> ToBytes for PMacroBlock<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
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

impl<const N: usize, T> FromBytes for PMacroBlock<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
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

pub(crate) struct PMacroBlocks<T>(Vec<PMacroBlock<T>>);

impl<T> PMacroBlocks<T> {
    pub(crate) fn new(blocks: Vec<PMacroBlock<T>>) -> Self {
        Self(blocks)
    }

    pub(crate) fn into_inner(self) -> Vec<PMacroBlock<T>> {
        self.0
    }
}

impl<const N: usize, T> Decodable for PMacroBlocks<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        Ok(Self(ans::decode(stream)?))
    }
}

impl<const N: usize, T> Encodable for PMacroBlocks<T>
where
    T: Sync + num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        ans::encode(&self.0, stream)
    }
}
