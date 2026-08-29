use crate::{color::Subsampling, image::ImageRef};

pub(crate) mod depth16;
pub(crate) mod depth8;

pub struct Jpg<'a, T> {
    pub image: ImageRef<'a, T>,
    pub subsampling: Subsampling,
}
