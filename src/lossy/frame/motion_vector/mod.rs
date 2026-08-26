use std::{io::Write, mem::size_of};

use crate::{
    bitstream::BitStreamWriter,
    color::Subsampling,
    error::{Error, Result},
    Decodable, Encodable,
};

pub(crate) mod depth16;
pub(crate) mod depth32;

// SAD threshold for early termination of sub-pixel search
// Blocks with SAD below this threshold skip expensive interpolation
// Higher = test more blocks (slower, better quality)
// Lower = skip more blocks (faster, slightly worse quality)
pub(crate) const SUBPIXEL_SAD_THRESHOLD: i16 = 1024;

// Expanded search pattern for better motion estimation
// Searches ±8 blocks (64 pixels) for very fast motion and sudden movements
pub(crate) const LARGE_DIAMOND: [(isize, isize); 20] = [
    // ±8 cardinal directions (very far points for extreme motion)
    (-8, 0),
    (8, 0),
    (0, -8),
    (0, 8),
    // ±6 cardinal directions (far points)
    (-6, 0),
    (6, 0),
    (0, -6),
    (0, 6),
    // ±4 cardinal directions (mid-far points)
    (-4, 0),
    (4, 0),
    (0, -4),
    (0, 4),
    // ±2 cardinal directions (mid points)
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    // ±4 diagonal corners (for diagonal fast motion)
    (-4, 4),
    (-4, -4),
    (4, 4),
    (4, -4),
];

pub(crate) const SMALL_DIAMOND: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

// Half-pixel offsets for sub-pixel motion estimation
pub(crate) const HALF_PIXEL_OFFSETS: [(isize, isize); 8] = [
    (0, 1),   // half-pixel down
    (1, 0),   // half-pixel right
    (1, 1),   // half-pixel diagonal
    (0, -1),  // half-pixel up
    (-1, 0),  // half-pixel left
    (-1, -1), // half-pixel diagonal
    (1, -1),  // half-pixel diagonal
    (-1, 1),  // half-pixel diagonal
];

/// Stored in quarter-pixel units (x=4 means 1 full pixel)
/// This allows half-pixel precision: x=2 means 0.5 pixels
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct MotionVector {
    pub(crate) x: isize,
    pub(crate) y: isize,
}

impl Decodable for MotionVector {
    type Output = Self;

    fn decode<R>(stream: &mut crate::BitStreamReader<R>) -> crate::Result<Self>
    where
        R: std::io::Read,
    {
        Ok(Self {
            x: stream
                .read::<usize>()?
                .ok_or(Error::FailedToDecode("mv x".to_owned()))? as isize,
            y: stream
                .read::<usize>()?
                .ok_or(Error::FailedToDecode("mv y".to_owned()))? as isize,
        })
    }
}

impl Encodable for MotionVector {
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        stream.write(self.x as usize)?;
        stream.write(self.y as usize)?;
        Ok(())
    }
}

impl MotionVector {
    // Get integer part (in blocks)
    pub(crate) fn integer_x(&self) -> isize {
        self.x / 2
    }

    pub(crate) fn integer_y(&self) -> isize {
        self.y / 2
    }

    // Check if has fractional (half-pixel) component
    pub(crate) fn has_half_pixel_x(&self) -> bool {
        self.x % 2 != 0
    }

    pub(crate) fn has_half_pixel_y(&self) -> bool {
        self.y % 2 != 0
    }

    /// Scale this motion vector for chroma based on subsampling mode.
    ///
    /// Returns a new MotionVector with scaled coordinates.
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

pub(crate) const MOTION_VECTOR_SIZE: usize = 2 * size_of::<isize>();

impl From<[u8; MOTION_VECTOR_SIZE]> for MotionVector {
    fn from(bytes: [u8; MOTION_VECTOR_SIZE]) -> Self {
        let x = isize::from_ne_bytes(bytes[0..size_of::<isize>()].try_into().unwrap());
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
        let mut bytes = [0; 2 * size_of::<isize>()];
        bytes[0..size_of::<isize>()].copy_from_slice(&vector.x.to_ne_bytes());
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
    fn test_motion_vector_default() {
        let mv = MotionVector::default();
        assert_eq!(mv.x, 0);
        assert_eq!(mv.y, 0);
    }

    #[test]
    fn test_motion_vector_from_tuple() {
        let mv = MotionVector::from((10, 20));
        assert_eq!(mv.x, 10);
        assert_eq!(mv.y, 20);
    }

    #[test]
    fn test_motion_vector_to_tuple() {
        let mv = MotionVector { x: 15, y: 25 };
        let tuple: (isize, isize) = mv.into();
        assert_eq!(tuple, (15, 25));
    }

    #[test]
    fn test_motion_vector_integer_x() {
        let mv = MotionVector { x: 10, y: 20 };
        assert_eq!(mv.integer_x(), 5); // 10 / 2
    }

    #[test]
    fn test_motion_vector_integer_y() {
        let mv = MotionVector { x: 10, y: 20 };
        assert_eq!(mv.integer_y(), 10); // 20 / 2
    }

    #[test]
    fn test_motion_vector_has_half_pixel_x() {
        let mv1 = MotionVector { x: 10, y: 20 };
        assert!(!mv1.has_half_pixel_x()); // 10 % 2 == 0

        let mv2 = MotionVector { x: 11, y: 20 };
        assert!(mv2.has_half_pixel_x()); // 11 % 2 != 0
    }

    #[test]
    fn test_motion_vector_has_half_pixel_y() {
        let mv1 = MotionVector { x: 10, y: 20 };
        assert!(!mv1.has_half_pixel_y()); // 20 % 2 == 0

        let mv2 = MotionVector { x: 10, y: 21 };
        assert!(mv2.has_half_pixel_y()); // 21 % 2 != 0
    }

    #[test]
    fn test_motion_vector_scale_for_chroma_420() {
        let mv = MotionVector { x: 16, y: 16 };
        let scaled = mv.scale_for_chroma(Subsampling::Sample420);
        assert_eq!(scaled.x, 8); // 16 / 2
        assert_eq!(scaled.y, 8); // 16 / 2
    }

    #[test]
    fn test_motion_vector_scale_for_chroma_422() {
        let mv = MotionVector { x: 16, y: 16 };
        let scaled = mv.scale_for_chroma(Subsampling::Sample422);
        assert_eq!(scaled.x, 8); // 16 / 2
        assert_eq!(scaled.y, 16); // unchanged
    }

    #[test]
    fn test_motion_vector_scale_for_chroma_411() {
        let mv = MotionVector { x: 16, y: 16 };
        let scaled = mv.scale_for_chroma(Subsampling::Sample411);
        assert_eq!(scaled.x, 4); // 16 / 4
        assert_eq!(scaled.y, 8); // 16 / 2
    }

    #[test]
    fn test_motion_vector_scale_for_chroma_444() {
        let mv = MotionVector { x: 16, y: 16 };
        let scaled = mv.scale_for_chroma(Subsampling::Sample444);
        assert_eq!(scaled.x, 16); // unchanged
        assert_eq!(scaled.y, 16); // unchanged
    }

    #[test]
    fn test_motion_vector_from_bytes() {
        let x = 100isize;
        let y = 200isize;
        let mut bytes = [0u8; MOTION_VECTOR_SIZE];
        bytes[0..size_of::<isize>()].copy_from_slice(&x.to_ne_bytes());
        bytes[size_of::<isize>()..].copy_from_slice(&y.to_ne_bytes());

        let mv = MotionVector::from(bytes);
        assert_eq!(mv.x, 100);
        assert_eq!(mv.y, 200);
    }

    #[test]
    fn test_motion_vector_to_bytes() {
        let mv = MotionVector { x: 150, y: 250 };
        let bytes: [u8; MOTION_VECTOR_SIZE] = mv.into();

        let x = isize::from_ne_bytes(bytes[0..size_of::<isize>()].try_into().unwrap());
        let y = isize::from_ne_bytes(bytes[size_of::<isize>()..].try_into().unwrap());

        assert_eq!(x, 150);
        assert_eq!(y, 250);
    }

    #[test]
    fn test_motion_vector_try_from_slice() {
        let x = 300isize;
        let y = 400isize;
        let mut bytes = vec![0u8; MOTION_VECTOR_SIZE];
        bytes[0..size_of::<isize>()].copy_from_slice(&x.to_ne_bytes());
        bytes[size_of::<isize>()..].copy_from_slice(&y.to_ne_bytes());

        let mv = MotionVector::try_from(bytes.as_slice()).unwrap();
        assert_eq!(mv.x, 300);
        assert_eq!(mv.y, 400);
    }

    #[test]
    fn test_motion_vector_try_from_slice_wrong_size() {
        let bytes = vec![0u8; 4]; // too small
        let result = MotionVector::try_from(bytes.as_slice());
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_vector_encode_decode() {
        let mv = MotionVector { x: 123, y: 456 };

        let mut buffer = Vec::new();
        let mut writer = BitStreamWriter::new(&mut buffer);
        mv.encode(&mut writer).unwrap();
        writer.flush().unwrap();

        let mut reader = crate::BitStreamReader::new(buffer.as_slice());
        let decoded = MotionVector::decode(&mut reader).unwrap();

        assert_eq!(decoded.x, 123);
        assert_eq!(decoded.y, 456);
    }

    #[test]
    fn test_motion_vector_equality() {
        let mv1 = MotionVector { x: 10, y: 20 };
        let mv2 = MotionVector { x: 10, y: 20 };
        let mv3 = MotionVector { x: 15, y: 25 };

        assert_eq!(mv1, mv2);
        assert_ne!(mv1, mv3);
    }

    #[test]
    fn test_motion_vector_negative_values() {
        let mv = MotionVector { x: -10, y: -20 };
        assert_eq!(mv.integer_x(), -5); // -10 / 2
        assert_eq!(mv.integer_y(), -10); // -20 / 2
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector { x: 0, y: 0 };
        assert_eq!(mv.integer_x(), 0);
        assert_eq!(mv.integer_y(), 0);
        assert!(!mv.has_half_pixel_x());
        assert!(!mv.has_half_pixel_y());
    }
}
