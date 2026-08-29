use std::io::{Read, Write};

use num_traits::{Bounded, FromPrimitive, ToPrimitive};

pub mod bitstream;
// mod bitvec;
pub mod block;
pub mod color;
mod dct;
pub mod dimensions;
mod encoders;
pub mod error;
pub mod image;
pub mod lossless;
pub mod lossy;
pub mod pixels;
mod point;
mod rice;

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    error::{Error, Result},
};

pub(crate) trait Encodable {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write;
}

pub(crate) trait Decodable {
    type Output;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read;
}

pub(crate) trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

pub(crate) trait FromBytes {
    fn from_bytes(bytes: &[u8]) -> (Self, usize)
    where
        Self: Sized;
}

pub(crate) fn clamp<T>(val: f32) -> T
where
    T: Bounded + FromPrimitive + ToPrimitive,
{
    T::from_f32(val.clamp(0.0, T::max_value().to_f32().expect("to f32"))).expect("from f32")
}
