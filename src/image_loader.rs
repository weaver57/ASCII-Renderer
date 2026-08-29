use anyhow::{Context, Result};
use image::imageops::FilterType;
use std::path::Path;

pub struct ImageFrame {
    pub rgb_data: Vec<u8>,
}

/// Loads a static image, scales it to `(target_width, target_height)`, and returns raw RGB bytes.
pub fn load_and_resize_image<P: AsRef<Path>>(
    path: P,
    target_width: u32,
    target_height: u32,
) -> Result<ImageFrame> {
    let img = image::open(&path)
        .with_context(|| format!("Failed to open image at {:?}", path.as_ref()))?;

    let resized = img.resize_exact(target_width, target_height, FilterType::Triangle);
    let rgb_img = resized.to_rgb8();

    Ok(ImageFrame {
        rgb_data: rgb_img.into_raw(),
    })
}
