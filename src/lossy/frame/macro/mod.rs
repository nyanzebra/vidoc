use std::{
    fmt::Debug,
    io::{Read, Write},
};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    block::{Block, Blocks},
    error::{Error, Result},
    Decodable, Encodable, FromBytes, ToBytes,
};

mod block_location;
pub use block_location::BlockLocation;

mod bmacro;
pub(crate) use bmacro::{BMacroBlock, BMacroBlocks, Prediction};

mod imacro;
pub(crate) use imacro::{IMacroBlock, IMacroBlocks};

mod pmacro;
pub(crate) use pmacro::{PMacroBlock, PMacroBlocks};

/// Trait for macroblocks that can be reassembled into frames.
/// Implemented by both PMacroBlock and BMacroBlock to provide
/// a unified interface for frame reassembly.
pub(crate) trait AssemblableMacroBlock {
    /// Get the block location (spatial extent) of this macroblock
    fn location(&self) -> &BlockLocation;

    /// Get the prediction type (Forward, Backward, or Both)
    fn prediction(&self) -> Prediction;

    /// Get the residuals (Y, Cb, Cr)
    fn residuals(&self) -> &Residuals<i16>;
}

pub(crate) struct Residuals<T> {
    pub(crate) y: Blocks<T>,
    pub(crate) cb: Blocks<T>,
    pub(crate) cr: Blocks<T>,
}

impl<const N: usize, T> ToBytes for Residuals<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];

        bytes.extend(self.y.to_bytes());
        bytes.extend(self.cb.to_bytes());
        bytes.extend(self.cr.to_bytes());

        bytes
    }
}

impl<const N: usize, T> FromBytes for Residuals<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    fn from_bytes(bytes: &[u8]) -> (Self, usize)
    where
        Self: Sized,
    {
        let mut offset = 0;
        let (y, size) = Blocks::from_bytes(bytes);
        offset += size;
        let (cb, size) = Blocks::from_bytes(&bytes[offset..]);
        offset += size;
        let (cr, size) = Blocks::from_bytes(&bytes[offset..]);
        offset += size;
        (Self { y, cb, cr }, offset)
    }
}

impl<const N: usize, T> Decodable for Residuals<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let mut y = Vec::new();
        let mut cb = Vec::new();
        let mut cr = Vec::new();

        let len = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("y len".to_owned()))? as usize;
        for _ in 0..len {
            y.push(Block::<T>::decode(stream)?);
        }

        let len = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("cb len".to_owned()))? as usize;
        for _ in 0..len {
            cb.push(Block::<T>::decode(stream)?);
        }

        let len = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("cr len".to_owned()))? as usize;
        for _ in 0..len {
            cr.push(Block::<T>::decode(stream)?);
        }

        Ok(Self {
            y: Blocks::new(y),
            cb: Blocks::new(cb),
            cr: Blocks::new(cr),
        })
    }
}

impl<const N: usize, T> Encodable for Residuals<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        stream.write(self.y.len() as u32)?;
        for block in self.y.iter() {
            block.encode(stream)?;
        }
        stream.write(self.cb.len() as u32)?;
        for block in self.cb.iter() {
            block.encode(stream)?;
        }
        stream.write(self.cr.len() as u32)?;
        for block in self.cr.iter() {
            block.encode(stream)?;
        }
        Ok(())
    }
}
