// https://medium.com/@oleg.shipitko/what-does-stride-mean-in-image-processing-bba158a72bcd

use std::marker::PhantomData;

use crate::{
    color::Subsampling,
    dimensions::PixelDimensions,
    lossy::{subsample_into_block_ycbcr, SubSampleBlockGroup},
    pixels::{
        Ayuv16, Ayuv16Ref, Ayuv8, Ayuv8Ref, Depth, Grey, GreyRef, Pixels, PixelsRef, Rgb16,
        Rgb16Ref, Rgb8, Rgb8Ref, Rgba16, Rgba16Ref, Rgba8, Rgba8Ref, Yuv16, Yuv16Ref, Yuv8,
        Yuv8Ref,
    },
};

pub struct Image<T> {
    pixels: T,
    dimensions: PixelDimensions,
    subsampling: Subsampling,
    line_stride: usize,
}

pub type ImageGrey = Image<Grey>;
pub type ImageRgb8 = Image<Rgb8>;
pub type ImageRgb16 = Image<Rgb16>;
pub type ImageRgba8 = Image<Rgba8>;
pub type ImageRgba16 = Image<Rgba16>;
pub type ImageYuv8 = Image<Yuv8>;
pub type ImageYuv16 = Image<Yuv16>;
pub type ImageAyuv8 = Image<Ayuv8>;
pub type ImageAyuv16 = Image<Ayuv16>;

impl<const DEPTH: usize, T> Image<Pixels<DEPTH, T>>
where
    T: Copy,
{
    pub fn as_ref(&self) -> ImageRef<'_, PixelsRef<'_, DEPTH, T>> {
        ImageRef {
            pixels: self.pixels.borrow(),
            dimensions: self.dimensions,

            line_stride: self.line_stride,
            _phantom: PhantomData,
        }
    }

    pub fn pixel(&self, (x, y): (usize, usize)) -> T {
        let offset = self.offset((x, y));
        self.pixels.pixel(offset)
    }

    pub fn update_pixel(&mut self, (row, col): (usize, usize), value: T) {
        let offset = self.offset((row, col));
        self.pixels.update(offset, value);
    }
}

impl<T> Image<T> {
    pub fn dimensions(&self) -> PixelDimensions {
        self.dimensions
    }

    pub fn line_stride(&self) -> usize {
        self.line_stride
    }

    pub fn pixels(&self) -> &T {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut T {
        &mut self.pixels
    }

    pub fn subsampling(&self) -> Subsampling {
        self.subsampling
    }

    fn offset(&self, (row, col): (usize, usize)) -> usize {
        (row * self.line_stride) + col
    }
}

impl<T> Image<T>
where
    T: Depth,
{
    pub fn new(dimensions: PixelDimensions, pixels: T, subsampling: Subsampling) -> Self {
        let line_stride = usize::from(T::depth()) * dimensions.width;
        Self {
            pixels,
            dimensions,
            subsampling,
            line_stride,
        }
    }

    pub fn depth(&self) -> u8 {
        T::depth()
    }
}

impl ImageRgb8 {
    pub fn subsample_into_block_ycbcr(&self) -> SubSampleBlockGroup<f64> {
        subsample_into_block_ycbcr(self.dimensions, &self.pixels.to_ycbcr(), self.subsampling)
    }
}

impl ImageRgb16 {
    pub fn subsample_into_block_ycbcr(&self) -> SubSampleBlockGroup<f64> {
        subsample_into_block_ycbcr(self.dimensions, &self.pixels.to_ycbcr(), self.subsampling)
    }
}

pub type ImageRefGrey<'a> = ImageRef<'a, GreyRef<'a>>;
pub type ImageRefRgb8<'a> = ImageRef<'a, Rgb8Ref<'a>>;
pub type ImageRefRgb16<'a> = ImageRef<'a, Rgb16Ref<'a>>;
pub type ImageRefRgba8<'a> = ImageRef<'a, Rgba8Ref<'a>>;
pub type ImageRefRgba16<'a> = ImageRef<'a, Rgba16Ref<'a>>;
pub type ImageRefYuv8<'a> = ImageRef<'a, Yuv8Ref<'a>>;
pub type ImageRefYuv16<'a> = ImageRef<'a, Yuv16Ref<'a>>;
pub type ImageRefAyuv8<'a> = ImageRef<'a, Ayuv8Ref<'a>>;
pub type ImageRefAyuv16<'a> = ImageRef<'a, Ayuv16Ref<'a>>;

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub struct ImageRef<'a, T> {
    pixels: T,
    dimensions: PixelDimensions,

    line_stride: usize,
    _phantom: PhantomData<&'a ()>,
}

impl<T> ImageRef<'_, T>
where
    T: Depth,
{
    pub fn dimensions(&self) -> PixelDimensions {
        self.dimensions
    }

    pub fn depth(&self) -> u8 {
        T::depth()
    }

    pub fn line_stride(&self) -> usize {
        self.line_stride
    }

    pub fn pixels(&self) -> &T {
        &self.pixels
    }
}
