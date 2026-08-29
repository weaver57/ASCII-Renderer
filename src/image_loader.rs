use anyhow::{Context, Result};
use image::imageops::FilterType;
use std::path::Path;

/// Aspect ratio of a terminal character cell: `width / height`.
/// A monospace glyph is roughly twice as tall as it is wide (~0.5).
/// This is the factor that prevents vertically-squashed or stretched output.
pub const CHAR_ASPECT: f32 = 0.5;

pub struct ImageFrame {
    pub rgb_data: Vec<u8>,
}

/// Computes the output character-grid dimensions that preserve the image's
/// aspect ratio on screen.
///
/// A character cell is physically taller than it is wide (`CHAR_ASPECT < 1`),
/// so a naive "one source pixel per cell" mapping vertically distorts the
/// image. To compensate, we derive the row count from the *source* aspect
/// ratio rather than blindly using the terminal height, so each cell samples
/// a source block that is itself `1/CHAR_ASPECT` times as tall as it is wide.
///
/// * `custom_cols`/`custom_rows` — explicit CLI overrides (`--width`/`--height`).
/// * If `custom_rows` is `None`, rows are derived from `cols` + image aspect.
/// * Both axes are clamped to fit the terminal (`term_cols` x `term_rows`),
///   reserving one line at the bottom for a status/clean margin.
pub fn compute_image_grid_dimensions(
    img_w: u32,
    img_h: u32,
    custom_cols: Option<usize>,
    custom_rows: Option<usize>,
    term_cols: u16,
    term_rows: u16,
) -> (usize, usize) {
    let max_cols = term_cols as usize;
    let max_rows = (term_rows as usize).saturating_sub(1).max(1);

    let cols = match custom_cols {
        Some(c) => c.min(max_cols).max(1),
        None => max_cols,
    };

    let rows = match custom_rows {
        Some(r) => r.min(max_rows).max(1),
        None => {
            let w = img_w.max(1) as f32;
            let h = img_h.max(1) as f32;
            let derived = ((cols as f32) * (h / w) * CHAR_ASPECT).round() as usize;
            derived.max(1).min(max_rows)
        }
    };

    (cols, rows)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn g(img_w: u32, img_h: u32, cols: Option<usize>, rows: Option<usize>, tc: u16, tr: u16) -> (usize, usize) {
        compute_image_grid_dimensions(img_w, img_h, cols, rows, tc, tr)
    }

    #[test]
    fn square_image_preserves_aspect() {
        // 100x100 square at a 100-col terminal in a 60-row terminal.
        // Derived rows = round(100 * (100/100) * 0.5) = 50.
        let (cols, rows) = g(100, 100, None, None, 100, 60);
        assert_eq!(cols, 100);
        assert_eq!(rows, 50);
    }

    #[test]
    fn landscape_image_produces_fewer_rows_than_cols() {
        // 16:9 landscape (160x90). cols=80 -> rows = round(80*(90/160)*0.5) = round(22.5)=23.
        let (cols, rows) = g(160, 90, Some(80), None, 100, 60);
        assert_eq!(cols, 80);
        assert_eq!(rows, 23);
    }

    #[test]
    fn portrait_image_produces_more_rows_per_col() {
        // 3:4 portrait (90x120). Rows derived should exceed cols (before terminal clamp).
        let (cols, rows) = g(90, 120, None, None, 80, 200);
        // cols=80, rows = round(80*(120/90)*0.5) = round(53.33) = 53. Fits in 199.
        assert_eq!(cols, 80);
        assert_eq!(rows, 53);
    }

    #[test]
    fn explicit_size_override_wins() {
        // User explicitly requests 50x20 -> honored (20 <= max_rows).
        let (cols, rows) = g(100, 100, Some(50), Some(20), 100, 60);
        assert_eq!((cols, rows), (50, 20));
    }

    #[test]
    fn explicit_height_only_keeps_cols_default_and_honors_height() {
        let (cols, rows) = g(100, 100, None, Some(20), 100, 60);
        assert_eq!(cols, 100);
        assert_eq!(rows, 20);
    }

    #[test]
    fn clamps_to_terminal_bounds() {
        // Terminal only 40x10 -> max_rows = 9. Derived rows (50) clamp to 9, cols to 40.
        let (cols, rows) = g(100, 100, None, None, 40, 10);
        assert_eq!(cols, 40);
        assert_eq!(rows, 9);
    }

    #[test]
    fn extreme_explicit_overrides_clamp_into_terminal() {
        // Huge requested values get clamped to the terminal, never 0.
        let (cols, rows) = g(100, 100, Some(9999), Some(9999), 80, 40);
        assert_eq!(cols, 80);
        assert_eq!(rows, 39);
        assert!(cols >= 1 && rows >= 1);
    }

    #[test]
    fn row_count_never_zero_even_when_terminal_tiny() {
        let (cols, rows) = g(400, 300, Some(200), None, 10, 1);
        // max_rows = 0 saturating_sub(1)... -> max(1). derived gets clamped to 1.
        assert!(cols >= 1 && rows >= 1);
    }
}
