use std::{
    io::{Read, Write},
    mem::size_of,
};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    error::{Error, Result},
    Decodable, Encodable,
};

pub(crate) const POINT_SIZE: usize = size_of::<Point>();

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
#[repr(C)]
pub struct Point {
    pub row: usize,
    pub col: usize,
}

impl Decodable for Point {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        Ok(Self {
            row: stream
                .read::<u32>()?
                .ok_or(Error::FailedToDecode("row".to_owned()))? as usize,
            col: stream
                .read::<u32>()?
                .ok_or(Error::FailedToDecode("col".to_owned()))? as usize,
        })
    }
}

impl Encodable for Point {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        stream.write(self.row as u32)?;
        stream.write(self.col as u32)?;
        Ok(())
    }
}

impl From<(usize, usize)> for Point {
    fn from((row, col): (usize, usize)) -> Self {
        Self { row, col }
    }
}

impl From<Point> for (usize, usize) {
    fn from(point: Point) -> Self {
        (point.row, point.col)
    }
}

impl From<Point> for [u8; POINT_SIZE] {
    fn from(point: Point) -> Self {
        let mut bytes = [0u8; POINT_SIZE];
        bytes[0..size_of::<usize>()].copy_from_slice(&point.row.to_be_bytes());
        bytes[size_of::<usize>()..].copy_from_slice(&point.col.to_be_bytes());
        bytes
    }
}

impl From<[u8; POINT_SIZE]> for Point {
    fn from(bytes: [u8; POINT_SIZE]) -> Self {
        let row = usize::from_be_bytes(bytes[0..size_of::<usize>()].try_into().expect("usize"));
        let col = usize::from_be_bytes(bytes[size_of::<usize>()..].try_into().expect("usize"));
        Self { row, col }
    }
}

impl TryFrom<&[u8]> for Point {
    type Error = Error;

    fn try_from(bytes: &[u8]) -> Result<Self> {
        let slice: [u8; POINT_SIZE] = bytes.try_into()?;
        Ok(Self::from(slice))
    }
}
