use std::io::{Read, Write};

use crate::{
    color::Subsampling, BitStreamReader, BitStreamWriter, Decodable, Encodable, Error, Result,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockDimensions {
    pub width: usize,
    pub height: usize,
}

impl BlockDimensions {
    pub(crate) fn subsample(self, subsampling: Subsampling) -> Self {
        match subsampling {
            Subsampling::Sample444 => self,
            Subsampling::Sample422 => Self {
                width: self.width.div_ceil(2),
                height: self.height,
            },
            Subsampling::Sample420 => Self {
                width: self.width.div_ceil(2),
                height: self.height.div_ceil(2),
            },
            Subsampling::Sample411 => Self {
                width: self.width.div_ceil(4),
                height: self.height,
            },
        }
    }
}

impl Encodable for BlockDimensions {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        stream.write(self.width as u32)?;
        stream.write(self.height as u32)?;
        Ok(())
    }
}

impl Decodable for BlockDimensions {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let width = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("width".to_owned()))? as usize;
        let height = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("height".to_owned()))? as usize;
        Ok(Self { width, height })
    }
}

impl From<(u32, u32)> for BlockDimensions {
    fn from((width, height): (u32, u32)) -> Self {
        Self {
            width: width as usize,
            height: height as usize,
        }
    }
}

impl From<(usize, usize)> for BlockDimensions {
    fn from((width, height): (usize, usize)) -> Self {
        Self { width, height }
    }
}

impl From<PixelDimensions> for BlockDimensions {
    fn from(dimensions: PixelDimensions) -> Self {
        // Round up to cover all pixels
        Self {
            width: dimensions.width.div_ceil(8),
            height: dimensions.height.div_ceil(8),
        }
    }
}

impl From<&PixelDimensions> for BlockDimensions {
    fn from(dimensions: &PixelDimensions) -> Self {
        Self::from(*dimensions)
    }
}

impl From<BlockDimensions> for (usize, usize) {
    fn from(dimensions: BlockDimensions) -> Self {
        (dimensions.width, dimensions.height)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelDimensions {
    pub width: usize,
    pub height: usize,
}

impl PixelDimensions {
    pub(crate) fn subsample(self, subsampling: Subsampling) -> Self {
        match subsampling {
            Subsampling::Sample444 => self,
            Subsampling::Sample422 => Self {
                width: self.width.div_ceil(2),
                height: self.height,
            },
            Subsampling::Sample420 => Self {
                width: self.width.div_ceil(2),
                height: self.height.div_ceil(2),
            },
            Subsampling::Sample411 => Self {
                width: self.width.div_ceil(4),
                height: self.height,
            },
        }
    }
}

impl Encodable for PixelDimensions {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        stream.write(self.width as u32)?;
        stream.write(self.height as u32)?;
        Ok(())
    }
}

impl Decodable for PixelDimensions {
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: Read,
    {
        let width = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("width".to_owned()))? as usize;
        let height = stream
            .read::<u32>()?
            .ok_or(Error::FailedToDecode("height".to_owned()))? as usize;
        Ok(Self { width, height })
    }
}

impl From<(u32, u32)> for PixelDimensions {
    fn from((width, height): (u32, u32)) -> Self {
        Self {
            width: width as usize,
            height: height as usize,
        }
    }
}

impl From<(usize, usize)> for PixelDimensions {
    fn from((width, height): (usize, usize)) -> Self {
        Self { width, height }
    }
}

impl From<PixelDimensions> for (usize, usize) {
    fn from(dimensions: PixelDimensions) -> Self {
        (dimensions.width, dimensions.height)
    }
}

impl From<BlockDimensions> for PixelDimensions {
    fn from(dimensions: BlockDimensions) -> Self {
        Self {
            width: dimensions.width * 8,
            height: dimensions.height * 8,
        }
    }
}

impl From<&BlockDimensions> for PixelDimensions {
    fn from(dimensions: &BlockDimensions) -> Self {
        Self::from(*dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_dimensions_from_tuple_u32() {
        let dims = BlockDimensions::from((640u32, 480u32));
        assert_eq!(dims.width, 640);
        assert_eq!(dims.height, 480);
    }

    #[test]
    fn test_block_dimensions_from_tuple_usize() {
        let dims = BlockDimensions::from((1920usize, 1080usize));
        assert_eq!(dims.width, 1920);
        assert_eq!(dims.height, 1080);
    }

    #[test]
    fn test_block_dimensions_to_tuple() {
        let dims = BlockDimensions {
            width: 100,
            height: 75,
        };
        let tuple: (usize, usize) = dims.into();
        assert_eq!(tuple, (100, 75));
    }

    #[test]
    fn test_block_dimensions_from_pixel_dimensions() {
        let pixel_dims = PixelDimensions {
            width: 640,
            height: 480,
        };
        let block_dims = BlockDimensions::from(pixel_dims);
        assert_eq!(block_dims.width, 80); // 640 / 8
        assert_eq!(block_dims.height, 60); // 480 / 8
    }

    #[test]
    fn test_block_dimensions_from_pixel_dimensions_rounds_up() {
        let pixel_dims = PixelDimensions {
            width: 641,
            height: 481,
        };
        let block_dims = BlockDimensions::from(pixel_dims);
        assert_eq!(block_dims.width, 81); // div_ceil(641 / 8) = 81
        assert_eq!(block_dims.height, 61); // div_ceil(481 / 8) = 61
    }

    #[test]
    fn test_block_dimensions_subsample_444() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };
        let subsampled = dims.subsample(Subsampling::Sample444);
        assert_eq!(subsampled.width, 80);
        assert_eq!(subsampled.height, 60);
    }

    #[test]
    fn test_block_dimensions_subsample_422() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };
        let subsampled = dims.subsample(Subsampling::Sample422);
        assert_eq!(subsampled.width, 40); // div_ceil(80 / 2)
        assert_eq!(subsampled.height, 60);
    }

    #[test]
    fn test_block_dimensions_subsample_420() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };
        let subsampled = dims.subsample(Subsampling::Sample420);
        assert_eq!(subsampled.width, 40); // div_ceil(80 / 2)
        assert_eq!(subsampled.height, 30); // div_ceil(60 / 2)
    }

    #[test]
    fn test_block_dimensions_subsample_411() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };
        let subsampled = dims.subsample(Subsampling::Sample411);
        assert_eq!(subsampled.width, 20); // div_ceil(80 / 4)
        assert_eq!(subsampled.height, 60);
    }

    #[test]
    fn test_block_dimensions_encode_decode() {
        let dims = BlockDimensions {
            width: 100,
            height: 75,
        };

        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);
        dims.encode(&mut writer).unwrap();
        writer.flush().unwrap();

        let mut reader = BitStreamReader::new(buffer.as_slice());
        let decoded = BlockDimensions::decode(&mut reader).unwrap();

        assert_eq!(decoded.width, 100);
        assert_eq!(decoded.height, 75);
    }

    #[test]
    fn test_pixel_dimensions_from_tuple_u32() {
        let dims = PixelDimensions::from((1920u32, 1080u32));
        assert_eq!(dims.width, 1920);
        assert_eq!(dims.height, 1080);
    }

    #[test]
    fn test_pixel_dimensions_from_tuple_usize() {
        let dims = PixelDimensions::from((640usize, 480usize));
        assert_eq!(dims.width, 640);
        assert_eq!(dims.height, 480);
    }

    #[test]
    fn test_pixel_dimensions_to_tuple() {
        let dims = PixelDimensions {
            width: 1920,
            height: 1080,
        };
        let tuple: (usize, usize) = dims.into();
        assert_eq!(tuple, (1920, 1080));
    }

    #[test]
    fn test_pixel_dimensions_from_block_dimensions() {
        let block_dims = BlockDimensions {
            width: 80,
            height: 60,
        };
        let pixel_dims = PixelDimensions::from(block_dims);
        assert_eq!(pixel_dims.width, 640); // 80 * 8
        assert_eq!(pixel_dims.height, 480); // 60 * 8
    }

    #[test]
    fn test_pixel_dimensions_from_block_dimensions_ref() {
        let block_dims = BlockDimensions {
            width: 100,
            height: 75,
        };
        let pixel_dims = PixelDimensions::from(&block_dims);
        assert_eq!(pixel_dims.width, 800); // 100 * 8
        assert_eq!(pixel_dims.height, 600); // 75 * 8
    }

    #[test]
    fn test_pixel_dimensions_subsample_444() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };
        let subsampled = dims.subsample(Subsampling::Sample444);
        assert_eq!(subsampled.width, 640);
        assert_eq!(subsampled.height, 480);
    }

    #[test]
    fn test_pixel_dimensions_subsample_422() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };
        let subsampled = dims.subsample(Subsampling::Sample422);
        assert_eq!(subsampled.width, 320); // div_ceil(640 / 2)
        assert_eq!(subsampled.height, 480);
    }

    #[test]
    fn test_pixel_dimensions_subsample_420() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };
        let subsampled = dims.subsample(Subsampling::Sample420);
        assert_eq!(subsampled.width, 320); // div_ceil(640 / 2)
        assert_eq!(subsampled.height, 240); // div_ceil(480 / 2)
    }

    #[test]
    fn test_pixel_dimensions_subsample_411() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };
        let subsampled = dims.subsample(Subsampling::Sample411);
        assert_eq!(subsampled.width, 160); // div_ceil(640 / 4)
        assert_eq!(subsampled.height, 480);
    }

    #[test]
    fn test_pixel_dimensions_encode_decode() {
        let dims = PixelDimensions {
            width: 1920,
            height: 1080,
        };

        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);
        dims.encode(&mut writer).unwrap();
        writer.flush().unwrap();

        let mut reader = BitStreamReader::new(buffer.as_slice());
        let decoded = PixelDimensions::decode(&mut reader).unwrap();

        assert_eq!(decoded.width, 1920);
        assert_eq!(decoded.height, 1080);
    }

    #[test]
    fn test_dimensions_equality() {
        let dims1 = PixelDimensions {
            width: 640,
            height: 480,
        };
        let dims2 = PixelDimensions {
            width: 640,
            height: 480,
        };
        let dims3 = PixelDimensions {
            width: 1920,
            height: 1080,
        };

        assert_eq!(dims1, dims2);
        assert_ne!(dims1, dims3);
    }
}
