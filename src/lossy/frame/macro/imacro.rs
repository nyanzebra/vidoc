use std::{
    fmt::Debug,
    io::{Read, Write},
    mem::size_of,
};

use rayon::{
    iter::{IntoParallelRefIterator as _, ParallelIterator as _},
    slice::ParallelSlice as _,
};

use super::BlockLocation;
use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    block::Block,
    encoders::ans,
    error::{Error, Result},
    lossy::frame::r#macro::block_location::BLOCK_LOCATION_SIZE,
    Decodable, Encodable, FromBytes, ToBytes,
};

pub(crate) struct IMacroBlock<T> {
    pub(crate) location: BlockLocation,
    pub(crate) blocks: Vec<Block<T>>,
}

impl<const N: usize, T> Encodable for IMacroBlock<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.location.encode(stream)?;
        stream.write(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.encode(stream)?;
        }
        Ok(())
    }
}

impl<const N: usize, T> Decodable for IMacroBlock<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let location = BlockLocation::decode(stream)?;
        let len = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("len".to_owned()))? as usize;
        let mut blocks = Vec::with_capacity(len);
        for _ in 0..len {
            blocks.push(Block::decode(stream)?);
        }
        Ok(Self { location, blocks })
    }
}

impl<const N: usize, T> ToBytes for IMacroBlock<T>
where
    T: Sync + num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let slice: [u8; _] = self.location.into();
            bytes.extend(slice);
        }

        bytes.extend_from_slice(&self.blocks.len().to_be_bytes());
        bytes.extend(
            self.blocks
                .par_iter()
                .flat_map(|x| x.to_bytes())
                .collect::<Vec<u8>>(),
        );
        bytes
    }
}

impl<const N: usize, T> FromBytes for IMacroBlock<T>
where
    T: Debug + Send + num_traits::FromBytes<Bytes = [u8; N]>,
{
    fn from_bytes(bytes: &[u8]) -> (Self, usize) {
        let location =
            BlockLocation::try_from(&bytes[..BLOCK_LOCATION_SIZE]).expect("block location");

        let start = BLOCK_LOCATION_SIZE;
        let end = start + size_of::<usize>();

        let blocks = usize::from_be_bytes(bytes[start..end].try_into().expect("usize"));

        let start = end;
        let end = start + (blocks * Block::<T>::size() * size_of::<T>());

        let blocks = bytes[start..end]
            .par_chunks_exact(Block::<T>::size() * size_of::<T>())
            .map(Block::<T>::from_bytes)
            .map(|x| x.0)
            .collect::<Vec<_>>();

        (Self { location, blocks }, end)
    }
}

pub(crate) struct IMacroBlocks<T>(Vec<IMacroBlock<T>>);

impl<T> IMacroBlocks<T> {
    pub(crate) fn new(blocks: Vec<IMacroBlock<T>>) -> Self {
        Self(blocks)
    }

    pub(crate) fn into_inner(self) -> Vec<IMacroBlock<T>> {
        self.0
    }
}

impl<T> Decodable for IMacroBlocks<T>
where
    IMacroBlock<T>: FromBytes,
{
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        Ok(Self(ans::decode(stream)?))
    }
}

impl<T> Encodable for IMacroBlocks<T>
where
    T: Sync + num_traits::ToBytes,
    IMacroBlock<T>: ToBytes,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        ans::encode(&self.0, stream)
    }
}
