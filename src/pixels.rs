use rayon::prelude::*;

use crate::color::{rgba_to_ycbcr, Rgba, Ycbcr};

pub trait Depth {
    fn depth() -> u8;
}

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub struct Pixels<const DEPTH: usize, T> {
    data: Vec<T>,
}

impl<const DEPTH: usize, T> Depth for Pixels<DEPTH, T> {
    fn depth() -> u8 {
        DEPTH as u8
    }
}

impl<const DEPTH: usize, T> Pixels<DEPTH, T>
where
    T: Copy,
{
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn borrow(&self) -> PixelsRef<'_, DEPTH, T> {
        PixelsRef {
            data: self.data.as_slice(),
        }
    }

    pub const fn depth() -> usize {
        DEPTH
    }

    pub fn pixel(&self, idx: usize) -> T {
        self.data[idx]
    }

    pub fn update(&mut self, idx: usize, value: T) {
        self.data[idx] = value;
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into(self) -> Vec<T> {
        self.data
    }
}

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub struct PixelsRef<'a, const DEPTH: usize, T> {
    data: &'a [T],
}

impl<'a, const DEPTH: usize, T> Depth for PixelsRef<'a, DEPTH, T> {
    fn depth() -> u8 {
        DEPTH as u8
    }
}

impl<'a, const DEPTH: usize, T> PixelsRef<'a, DEPTH, T>
where
    T: Copy,
{
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    pub fn pixel(&self, idx: usize) -> T {
        self.data[idx]
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}

pub type Grey = Pixels<1, u8>;
pub type Rgb8 = Pixels<3, u8>;
pub type Rgb16 = Pixels<3, u16>;
pub type Rgba8 = Pixels<4, u8>;
pub type Rgba16 = Pixels<4, u16>;
pub type Yuv8 = Pixels<3, u8>;
pub type Yuv16 = Pixels<3, u16>;
pub type Ayuv8 = Pixels<4, u8>;
pub type Ayuv16 = Pixels<4, u16>;

pub type GreyRef<'a> = PixelsRef<'a, 1, u8>;
pub type Rgb8Ref<'a> = PixelsRef<'a, 3, u8>;
pub type Rgb16Ref<'a> = PixelsRef<'a, 3, u16>;
pub type Rgba8Ref<'a> = PixelsRef<'a, 4, u8>;
pub type Rgba16Ref<'a> = PixelsRef<'a, 4, u16>;
pub type Yuv8Ref<'a> = PixelsRef<'a, 3, u8>;
pub type Yuv16Ref<'a> = PixelsRef<'a, 3, u16>;
pub type Ayuv8Ref<'a> = PixelsRef<'a, 4, u8>;
pub type Ayuv16Ref<'a> = PixelsRef<'a, 4, u16>;

impl Rgb8 {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.borrow().to_ycbcr()
    }
}

impl Rgba8 {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.borrow().to_ycbcr()
    }
}

impl Rgb16 {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.borrow().to_ycbcr()
    }
}

impl Rgba16 {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.borrow().to_ycbcr()
    }
}

impl Rgb8Ref<'_> {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.data
            .par_chunks(Self::depth().into())
            .map(|rgb| {
                rgba_to_ycbcr(&Rgba {
                    r: rgb[0],
                    g: rgb[1],
                    b: rgb[2],
                    a: 0,
                })
            })
            .collect()
    }
}

impl Rgba8Ref<'_> {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.data
            .par_chunks(Self::depth().into())
            .map(|rgb| {
                rgba_to_ycbcr(&Rgba {
                    r: rgb[0],
                    g: rgb[1],
                    b: rgb[2],
                    a: rgb[3],
                })
            })
            .collect()
    }
}

impl Rgb16Ref<'_> {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.data
            .par_chunks(Self::depth().into())
            .map(|rgb| {
                rgba_to_ycbcr(&Rgba {
                    r: rgb[0],
                    g: rgb[1],
                    b: rgb[2],
                    a: 0,
                })
            })
            .collect()
    }
}

impl Rgba16Ref<'_> {
    pub fn to_ycbcr(&self) -> Vec<Ycbcr> {
        self.data
            .par_chunks(Self::depth().into())
            .map(|rgb| {
                rgba_to_ycbcr(&Rgba {
                    r: rgb[0],
                    g: rgb[1],
                    b: rgb[2],
                    a: rgb[3],
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb8_to_ycbcr_correctness() {
        // Test correctness with known values
        let rgb_data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // Red, Green, Blue
        let pixels = Pixels::<3, u8>::new(rgb_data);
        let rgb_ref = pixels.borrow();

        let ycbcr_result = rgb_ref.to_ycbcr();

        assert_eq!(ycbcr_result.len(), 3); // 3 pixels

        // Red pixel (255,0,0) should have:
        // - High Y (luminance)
        // - Low Cb (blue-yellow axis, negative for red)
        // - High Cr (red-cyan axis, positive for red)
        assert!(ycbcr_result[0].y > 50.0, "Red should have high luminance");
        assert!(
            ycbcr_result[0].cb < 128.0,
            "Red should have low Cb (less blue)"
        );
        assert!(
            ycbcr_result[0].cr > 128.0,
            "Red should have high Cr (more red)"
        );
    }

    #[test]
    fn test_rgba8_to_ycbcr_correctness() {
        // Test correctness with known values
        let rgba_data = vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]; // Red, Green, Blue
        let pixels = Pixels::<4, u8>::new(rgba_data);
        let rgba_ref = pixels.borrow();

        let ycbcr_result = rgba_ref.to_ycbcr();

        assert_eq!(ycbcr_result.len(), 3); // 3 pixels

        // Red pixel (255,0,0) should have:
        // - High Y (luminance)
        // - Low Cb (blue-yellow axis, negative for red)
        // - High Cr (red-cyan axis, positive for red)
        assert!(ycbcr_result[0].y > 50.0, "Red should have high luminance");
        assert!(
            ycbcr_result[0].cb < 128.0,
            "Red should have low Cb (less blue)"
        );
        assert!(
            ycbcr_result[0].cr > 128.0,
            "Red should have high Cr (more red)"
        );
    }
}
