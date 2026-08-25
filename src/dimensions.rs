use std::io::{Read, Write};

use crate::{
    color::Subsampling, BitStreamReader, BitStreamWriter, Decodable, Encodable, Error, Result,
};

/// Maximum dimension representable by the on-disk format.
///
/// This is deliberately a codec-format limit rather than an architecture
/// limit. Internally dimensions remain usize for efficient indexing.
pub const MAX_DIMENSION: usize = u32::MAX as usize;

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

    fn validate(&self) -> Result<()> {
        validate_dimension(self.width, "block width")?;
        validate_dimension(self.height, "block height")?;
        Ok(())
    }
}

impl Encodable for BlockDimensions {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.validate()?;

        stream.write_u32(u32::try_from(self.width).map_err(|_| Error::InvalidData)?)?;

        stream.write_u32(u32::try_from(self.height).map_err(|_| Error::InvalidData)?)?;

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
            .read_u32()?
            .ok_or_else(|| Error::FailedToDecode("block width".to_owned()))?;

        let height = stream
            .read_u32()?
            .ok_or_else(|| Error::FailedToDecode("block height".to_owned()))?;

        let dimensions = Self {
            width: width as usize,
            height: height as usize,
        };

        dimensions.validate()?;

        Ok(dimensions)
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

    fn validate(&self) -> Result<()> {
        validate_dimension(self.width, "pixel width")?;
        validate_dimension(self.height, "pixel height")?;

        // Zero-sized images are not useful codec frames and tend to create
        // division/iteration corner cases throughout the block pipeline.
        if self.width == 0 || self.height == 0 {
            return Err(Error::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }

        Ok(())
    }
}

impl Encodable for PixelDimensions {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        self.validate()?;

        stream.write_u32(u32::try_from(self.width).map_err(|_| Error::InvalidData)?)?;

        stream.write_u32(u32::try_from(self.height).map_err(|_| Error::InvalidData)?)?;

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
            .read_u32()?
            .ok_or_else(|| Error::FailedToDecode("pixel width".to_owned()))?;

        let height = stream
            .read_u32()?
            .ok_or_else(|| Error::FailedToDecode("pixel height".to_owned()))?;

        let dimensions = Self {
            width: width as usize,
            height: height as usize,
        };

        dimensions.validate()?;

        Ok(dimensions)
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
            width: dimensions.width.saturating_mul(8),
            height: dimensions.height.saturating_mul(8),
        }
    }
}

impl From<&BlockDimensions> for PixelDimensions {
    fn from(dimensions: &BlockDimensions) -> Self {
        Self::from(*dimensions)
    }
}

fn validate_dimension(value: usize, _name: &str) -> Result<()> {
    if value > MAX_DIMENSION {
        return Err(Error::InvalidData);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dimensions_from_tuple_u32() {
        let dims = BlockDimensions::from((640u32, 480u32));

        assert_eq!(dims.width, 640);
        assert_eq!(dims.height, 480);
    }

    #[test]
    fn block_dimensions_from_tuple_usize() {
        let dims = BlockDimensions::from((1920usize, 1080usize));

        assert_eq!(dims.width, 1920);
        assert_eq!(dims.height, 1080);
    }

    #[test]
    fn block_dimensions_to_tuple() {
        let dims = BlockDimensions {
            width: 100,
            height: 75,
        };

        let tuple: (usize, usize) = dims.into();

        assert_eq!(tuple, (100, 75));
    }

    #[test]
    fn block_dimensions_from_pixel_dimensions() {
        let pixel_dims = PixelDimensions {
            width: 640,
            height: 480,
        };

        let block_dims = BlockDimensions::from(pixel_dims);

        assert_eq!(block_dims.width, 80);
        assert_eq!(block_dims.height, 60);
    }

    #[test]
    fn block_dimensions_from_pixel_dimensions_rounds_up() {
        let pixel_dims = PixelDimensions {
            width: 641,
            height: 481,
        };

        let block_dims = BlockDimensions::from(pixel_dims);

        assert_eq!(block_dims.width, 81);
        assert_eq!(block_dims.height, 61);
    }

    #[test]
    fn block_dimensions_subsample_444() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample444),
            BlockDimensions {
                width: 80,
                height: 60
            }
        );
    }

    #[test]
    fn block_dimensions_subsample_422() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample422),
            BlockDimensions {
                width: 40,
                height: 60
            }
        );
    }

    #[test]
    fn block_dimensions_subsample_420() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample420),
            BlockDimensions {
                width: 40,
                height: 30
            }
        );
    }

    #[test]
    fn block_dimensions_subsample_411() {
        let dims = BlockDimensions {
            width: 80,
            height: 60,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample411),
            BlockDimensions {
                width: 20,
                height: 60
            }
        );
    }

    #[test]
    fn block_dimensions_encode_decode() {
        let dims = BlockDimensions {
            width: 100,
            height: 75,
        };

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            dims.encode(&mut writer).unwrap();
            writer.flush().unwrap();
        }

        // Two u32 values = exactly eight bytes in the format.
        assert_eq!(buffer.len(), 8);

        let mut reader = BitStreamReader::new(buffer.as_slice());

        let decoded = BlockDimensions::decode(&mut reader).unwrap();

        assert_eq!(decoded, dims);
    }

    #[test]
    fn pixel_dimensions_from_tuple_u32() {
        let dims = PixelDimensions::from((1920u32, 1080u32));

        assert_eq!(dims.width, 1920);
        assert_eq!(dims.height, 1080);
    }

    #[test]
    fn pixel_dimensions_from_tuple_usize() {
        let dims = PixelDimensions::from((640usize, 480usize));

        assert_eq!(dims.width, 640);
        assert_eq!(dims.height, 480);
    }

    #[test]
    fn pixel_dimensions_to_tuple() {
        let dims = PixelDimensions {
            width: 1920,
            height: 1080,
        };

        let tuple: (usize, usize) = dims.into();

        assert_eq!(tuple, (1920, 1080));
    }

    #[test]
    fn pixel_dimensions_from_block_dimensions() {
        let block_dims = BlockDimensions {
            width: 80,
            height: 60,
        };

        let pixel_dims = PixelDimensions::from(block_dims);

        assert_eq!(pixel_dims.width, 640);
        assert_eq!(pixel_dims.height, 480);
    }

    #[test]
    fn pixel_dimensions_from_block_dimensions_ref() {
        let block_dims = BlockDimensions {
            width: 100,
            height: 75,
        };

        let pixel_dims = PixelDimensions::from(&block_dims);

        assert_eq!(pixel_dims.width, 800);
        assert_eq!(pixel_dims.height, 600);
    }

    #[test]
    fn pixel_dimensions_subsample_444() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample444),
            PixelDimensions {
                width: 640,
                height: 480
            }
        );
    }

    #[test]
    fn pixel_dimensions_subsample_422() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample422),
            PixelDimensions {
                width: 320,
                height: 480
            }
        );
    }

    #[test]
    fn pixel_dimensions_subsample_420() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample420),
            PixelDimensions {
                width: 320,
                height: 240
            }
        );
    }

    #[test]
    fn pixel_dimensions_subsample_411() {
        let dims = PixelDimensions {
            width: 640,
            height: 480,
        };

        assert_eq!(
            dims.subsample(Subsampling::Sample411),
            PixelDimensions {
                width: 160,
                height: 480
            }
        );
    }

    #[test]
    fn pixel_dimensions_encode_decode() {
        let dims = PixelDimensions {
            width: 1920,
            height: 1080,
        };

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);

            dims.encode(&mut writer).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer.len(), 8);

        let mut reader = BitStreamReader::new(buffer.as_slice());

        let decoded = PixelDimensions::decode(&mut reader).unwrap();

        assert_eq!(decoded, dims);
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        let dims = PixelDimensions {
            width: 0,
            height: 1080,
        };

        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);

        assert!(dims.encode(&mut writer).is_err());
    }

    #[test]
    fn dimensions_are_fixed_width_on_disk() {
        let dims = PixelDimensions {
            width: 1920,
            height: 1080,
        };

        let mut buffer = Vec::new();

        {
            let mut writer = BitStreamWriter::new(&mut buffer);
            dims.encode(&mut writer).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(
            buffer,
            [
                0x00, 0x00, 0x07, 0x80, // 1920
                0x00, 0x00, 0x04, 0x38, // 1080
            ]
        );
    }
}
