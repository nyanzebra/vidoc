use std::{io::Write, mem::size_of};

use crate::{
    bitstream::BitStreamWriter,
    color::Subsampling,
    error::{Error, Result},
    Decodable, Encodable,
};

pub(crate) mod depth16;
pub(crate) mod depth32;

/// SAD threshold below which sub-pixel refinement is skipped.
pub(crate) const SUBPIXEL_SAD_THRESHOLD: i16 = 1024;

/// Large diamond: ±4 blocks (32 pixels). Covers typical motion at 24fps.
/// The old ±8 / 20-point pattern was overkill for 24fps content.
pub(crate) const LARGE_DIAMOND: [(isize, isize); 12] = [
    (-4, 0),
    (4, 0),
    (0, -4),
    (0, 4),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-2, 2),
    (-2, -2),
    (2, 2),
    (2, -2),
];

pub(crate) const SMALL_DIAMOND: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

pub(crate) const HALF_PIXEL_OFFSETS: [(isize, isize); 8] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (0, -1),
    (-1, 0),
    (-1, -1),
    (1, -1),
    (-1, 1),
];

/// Stored in half-pixel units: x=2 means 1 full pixel, x=1 means 0.5 pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct MotionVector {
    pub(crate) x: isize,
    pub(crate) y: isize,
}

impl MotionVector {
    #[inline]
    pub(crate) fn integer_x(&self) -> isize {
        self.x / 2
    }
    #[inline]
    pub(crate) fn integer_y(&self) -> isize {
        self.y / 2
    }
    #[inline]
    pub(crate) fn has_half_pixel_x(&self) -> bool {
        self.x % 2 != 0
    }
    #[inline]
    pub(crate) fn has_half_pixel_y(&self) -> bool {
        self.y % 2 != 0
    }

    pub(crate) fn scale_for_chroma(&self, subsampling: Subsampling) -> Self {
        Self {
            x: match subsampling {
                Subsampling::Sample420 | Subsampling::Sample422 => self.x / 2,
                Subsampling::Sample411 => self.x / 4,
                _ => self.x,
            },
            y: match subsampling {
                Subsampling::Sample420 | Subsampling::Sample411 => self.y / 2,
                _ => self.y,
            },
        }
    }
}

impl Decodable for MotionVector {
    type Output = Self;

    fn decode<R>(stream: &mut crate::BitStreamReader<R>) -> crate::Result<Self>
    where
        R: std::io::Read,
    {
        Ok(Self {
            x: stream
                .read::<u32>()?
                .ok_or(Error::FailedToDecode("mv x".to_owned()))? as isize,
            y: stream
                .read::<u32>()?
                .ok_or(Error::FailedToDecode("mv y".to_owned()))? as isize,
        })
    }
}

impl Encodable for MotionVector {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        stream.write(self.x as u32)?;
        stream.write(self.y as u32)?;
        Ok(())
    }
}

pub(crate) const MOTION_VECTOR_SIZE: usize = 2 * size_of::<isize>();

impl From<[u8; MOTION_VECTOR_SIZE]> for MotionVector {
    fn from(bytes: [u8; MOTION_VECTOR_SIZE]) -> Self {
        let x = isize::from_ne_bytes(bytes[..size_of::<isize>()].try_into().unwrap());
        let y = isize::from_ne_bytes(bytes[size_of::<isize>()..].try_into().unwrap());
        Self { x, y }
    }
}

impl TryFrom<&[u8]> for MotionVector {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self> {
        let slice: [u8; MOTION_VECTOR_SIZE] = value.try_into()?;
        Ok(Self::from(slice))
    }
}

impl From<MotionVector> for [u8; MOTION_VECTOR_SIZE] {
    fn from(vector: MotionVector) -> Self {
        let mut bytes = [0u8; 2 * size_of::<isize>()];
        bytes[..size_of::<isize>()].copy_from_slice(&vector.x.to_ne_bytes());
        bytes[size_of::<isize>()..].copy_from_slice(&vector.y.to_ne_bytes());
        bytes
    }
}

impl From<(isize, isize)> for MotionVector {
    fn from((x, y): (isize, isize)) -> Self {
        Self { x, y }
    }
}

impl From<MotionVector> for (isize, isize) {
    fn from(vector: MotionVector) -> Self {
        (vector.x, vector.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_parts() {
        let mv = MotionVector { x: 10, y: -20 };
        assert_eq!(mv.integer_x(), 5);
        assert_eq!(mv.integer_y(), -10);
    }

    #[test]
    fn half_pixel_detection() {
        assert!(!MotionVector { x: 10, y: 20 }.has_half_pixel_x());
        assert!(MotionVector { x: 11, y: 20 }.has_half_pixel_x());
        assert!(!MotionVector { x: 10, y: 20 }.has_half_pixel_y());
        assert!(MotionVector { x: 10, y: 21 }.has_half_pixel_y());
    }

    #[test]
    fn chroma_scaling() {
        let mv = MotionVector { x: 16, y: 16 };
        let s420 = mv.scale_for_chroma(Subsampling::Sample420);
        assert_eq!((s420.x, s420.y), (8, 8));
        let s422 = mv.scale_for_chroma(Subsampling::Sample422);
        assert_eq!((s422.x, s422.y), (8, 16));
        let s411 = mv.scale_for_chroma(Subsampling::Sample411);
        assert_eq!((s411.x, s411.y), (4, 8));
        let s444 = mv.scale_for_chroma(Subsampling::Sample444);
        assert_eq!((s444.x, s444.y), (16, 16));
    }

    #[test]
    fn encode_decode_roundtrip() {
        use crate::{BitStreamReader, BitStreamWriter};
        let mv = MotionVector { x: 123, y: 456 };
        let mut buf = Vec::new();
        let mut w = BitStreamWriter::new(&mut buf);
        mv.encode(&mut w).unwrap();
        w.flush().unwrap();
        let decoded = MotionVector::decode(&mut BitStreamReader::new(buf.as_slice())).unwrap();
        assert_eq!(decoded, mv);
    }
}
