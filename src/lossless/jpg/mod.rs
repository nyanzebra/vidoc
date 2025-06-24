use crate::image::ImageRef;

mod rgb16;
mod rgb8;

pub struct Jpg<'a, T> {
    pub(crate) image: ImageRef<'a, T>,
}
