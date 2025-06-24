use std::io::{Read, Write};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    color::Subsampling,
    error::{Error, Result},
    point::{Point, POINT_SIZE},
    Decodable, Encodable,
};

pub(crate) const BLOCK_LOCATION_SIZE: usize = size_of::<BlockLocation>();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct BlockLocation {
    pub start: Point,
    pub end: Point,
}

impl Decodable for BlockLocation {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        Ok(Self {
            start: Point::decode(stream)?,
            end: Point::decode(stream)?,
        })
    }
}

impl Encodable for BlockLocation {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.start.encode(stream)?;
        self.end.encode(stream)?;
        Ok(())
    }
}

impl From<[u8; BLOCK_LOCATION_SIZE]> for BlockLocation {
    fn from(bytes: [u8; BLOCK_LOCATION_SIZE]) -> Self {
        Self {
            start: Point::try_from(&bytes[..POINT_SIZE]).expect("start point"),
            end: Point::try_from(&bytes[POINT_SIZE..]).expect("end point"),
        }
    }
}

impl TryFrom<&[u8]> for BlockLocation {
    type Error = Error;

    fn try_from(bytes: &[u8]) -> Result<Self> {
        let slice: [u8; BLOCK_LOCATION_SIZE] = bytes.try_into()?;
        Ok(Self::from(slice))
    }
}

impl From<[usize; 4]> for BlockLocation {
    fn from([start_row, start_col, end_row, end_col]: [usize; 4]) -> Self {
        BlockLocation {
            start: Point {
                row: start_row,
                col: start_col,
            },
            end: Point {
                row: end_row,
                col: end_col,
            },
        }
    }
}

impl TryFrom<&[usize]> for BlockLocation {
    type Error = Error;

    fn try_from(bytes: &[usize]) -> Result<Self> {
        let slice: [usize; 4] = bytes.try_into()?;
        Ok(Self::from(slice))
    }
}

impl From<BlockLocation> for [usize; 4] {
    fn from(value: BlockLocation) -> Self {
        [
            value.start.row,
            value.start.col,
            value.end.row,
            value.end.col,
        ]
    }
}

impl From<BlockLocation> for [u8; BLOCK_LOCATION_SIZE] {
    fn from(value: BlockLocation) -> Self {
        let mut bytes = [0u8; BLOCK_LOCATION_SIZE];

        {
            let slice: [u8; _] = value.start.into();
            bytes[0..POINT_SIZE].copy_from_slice(&slice);
        }
        {
            let slice: [u8; POINT_SIZE] = value.end.into();
            bytes[POINT_SIZE..].copy_from_slice(&slice);
        }

        bytes
    }
}

impl BlockLocation {
    pub(crate) fn map_to_chroma(self, subsampling: Subsampling) -> Self {
        BlockLocation {
            start: Point {
                row: match subsampling {
                    Subsampling::Sample420 | Subsampling::Sample411 => self.start.row / 2,
                    _ => self.start.row,
                },
                col: match subsampling {
                    Subsampling::Sample420 | Subsampling::Sample422 => self.start.col / 2,
                    Subsampling::Sample411 => self.start.col / 4,
                    _ => self.start.col,
                },
            },
            end: Point {
                row: match subsampling {
                    Subsampling::Sample420 | Subsampling::Sample411 => self.end.row / 2,
                    _ => self.end.row,
                },
                col: match subsampling {
                    Subsampling::Sample420 | Subsampling::Sample422 => self.end.col / 2,
                    Subsampling::Sample411 => self.end.col / 4,
                    _ => self.end.col,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_location() {
        let block = BlockLocation {
            start: Point { row: 1, col: 2 },
            end: Point { row: 3, col: 4 },
        };
        assert_eq!(block.start.row, 1);
        assert_eq!(block.start.col, 2);
        assert_eq!(block.end.row, 3);
        assert_eq!(block.end.col, 4);
    }

    #[test]
    fn test_block_location_from_array() {
        let array = [1, 2, 3, 4];
        let block = BlockLocation::from(array);
        assert_eq!(block.start.row, 1);
        assert_eq!(block.start.col, 2);
        assert_eq!(block.end.row, 3);
        assert_eq!(block.end.col, 4);
    }

    #[test]
    fn test_block_to_array_and_back() {
        let block = BlockLocation {
            start: Point { row: 1, col: 2 },
            end: Point { row: 3, col: 4 },
        };
        let array: [u8; _] = block.into();
        let block2 = BlockLocation::from(array);
        assert_eq!(block, block2);
    }
}
