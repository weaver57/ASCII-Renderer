use anyhow::{Context, Result};
use image::imageops::FilterType;
use std::path::Path;

pub struct ImageFrame {
    pub rgb_data: Vec<u8>,
}

/// Computes the output character-grid dimensions that preserve the image's
/// aspect ratio on screen **and** fit inside the terminal.
///
/// A character cell is physically taller than it is wide (`CHAR_ASPECT < 1`),
/// so a naive "one source pixel per cell" mapping vertically distorts the
/// image. To compensate, we derive the row count from the *source* aspect
/// ratio rather than blindly using the terminal height, so each cell samples
/// a source block that is itself `1/CHAR_ASPECT` times as tall as it is wide.
///
/// * `custom_cols`/`custom_rows` — explicit CLI overrides (`--width`/`--height`).
/// * If `custom_rows` is `None`, rows are derived from `cols` + image aspect.
/// * If the derived `rows` would overflow the terminal, `cols` is **shrunk**
///   (not `rows` merely clamped) so the whole image still fits with its aspect
///   ratio intact. Clamping only `rows` to the terminal height while leaving
///   `cols` at full width silently stretches the image into an oval — the
///   classic "circle became an ellipse" bug for wide-and-short terminals.
///
/// One line is reserved at the bottom for a status/clean margin.
pub fn compute_image_grid_dimensions(
    img_w: u32,
    img_h: u32,
    custom_cols: Option<usize>,
    custom_rows: Option<usize>,
    term_cols: u16,
    term_rows: u16,
    char_aspect: f32,
) -> (usize, usize) {
    let max_cols = term_cols as usize;
    let max_rows = (term_rows as usize).saturating_sub(1).max(1);

    // Source aspect expressed in *cell* units: rows per column.
    // char_aspect (cell width / height) comes from terminal_size::get_char_aspect()
    // at the call site, so grid sizing never depends on a mutable global —
    // tests pass the exact aspect they want and never touch the CLI override.
    let scale = (img_h.max(1) as f32 / img_w.max(1) as f32) * char_aspect;

    let mut cols = match custom_cols {
        Some(c) => c.min(max_cols).max(1),
        None => max_cols,
    };

    let rows = match custom_rows {
        Some(r) => r.min(max_rows).max(1),
        None => {
            // If the rows derived from the full `cols` would overflow the
            // terminal, shrink `cols` to fit instead — preserving aspect.
            let derived_at_full_cols = (cols as f32 * scale).round() as usize;
            if derived_at_full_cols > max_rows {
                cols = ((max_rows as f32) / scale).floor().max(1.0) as usize;
            }
            let derived = (cols as f32 * scale).round() as usize;
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

/// Full-resolution decoded frame plus its dimensions. This is the source of
/// truth for Phase 2's edge detection, which runs Sobel at full pixel
/// resolution before aggregating into the smaller character grid (see `edge.rs`).
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB8, `width * height * 3` bytes.
    pub rgb_data: Vec<u8>,
}

/// Decodes an image at its full native resolution, without downsampling.
pub fn load_rgb_frame<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let img = image::open(&path).with_context(|| format!("Failed to open image at {:?}", path.as_ref()))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok(Frame {
        width: w,
        height: h,
        rgb_data: rgb.into_raw(),
    })
}

/// Returns the `[x0,x1) x [y0,y1)` source-pixel rectangle that cell `(col,row)`
/// covers, given a source of size `(src_w, src_h)` mapped onto a grid of
/// `(cols, rows)`.
///
/// Both the color/brightness downsampling **and** Phase 2's edge aggregation
/// must iterate over *exactly* this same rectangle — duplicating this rounding
/// logic anywhere else risks the two drifting out of sync. Never inline it.
#[inline]
pub fn cell_source_rect(
    col: usize,
    row: usize,
    cols: usize,
    rows: usize,
    src_w: u32,
    src_h: u32,
) -> (u32, u32, u32, u32) {
    let x0 = (col * src_w as usize / cols) as u32;
    let x1 = (((col + 1) * src_w as usize / cols) as u32).max(x0 + 1);
    let y0 = (row * src_h as usize / rows) as u32;
    let y1 = (((row + 1) * src_h as usize / rows) as u32).max(y0 + 1);
    (x0, x1, y0, y1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(img_w: u32, img_h: u32, cols: Option<usize>, rows: Option<usize>, tc: u16, tr: u16) -> (usize, usize) {
        compute_image_grid_dimensions(img_w, img_h, cols, rows, tc, tr, 0.5)
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
        // Terminal only 40x10 -> max_rows = 9. A square image, cell aspect 0.5,
        // needs rows = cols/2, so 9 rows only fits 18 cols. cols is SHRUNK to
        // 18 (not left at 40) to preserve a square-on-screen — otherwise the
        // circle becomes a wide oval. max_rows stays 9.
        let (cols, rows) = g(100, 100, None, None, 40, 10);
        assert_eq!(cols, 18);
        assert_eq!(rows, 9);
    }

    #[test]
    fn wide_short_terminal_shrinks_cols_to_preserve_aspect() {
        // 120x30 terminal, square image. Full-width cols (120) would want 60
        // rows but only 29 fit, so cols must drop to floor(29/0.5)=58, rows 29.
        let (cols, rows) = g(100, 100, None, None, 120, 30);
        assert_eq!(cols, 58);
        assert_eq!(rows, 29);
        // The on-screen physical size (cols wide, rows*2 tall) stays square.
        assert_eq!(cols, rows * 2);
    }

    #[test]
    fn tall_enough_terminal_keeps_full_width() {
        // Terminal tall enough to fit full-width derived rows (120 -> 60 <= 79):
        // no shrink needed, aspect still held.
        let (cols, rows) = g(100, 100, None, None, 120, 80);
        assert_eq!(cols, 120);
        assert_eq!(rows, 60);
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

    // --- Comprehensive fit & aspect-preservation sweep --------------------
    //
    // These tests verify that for ANY combination of terminal size and image
    // aspect ratio, the computed grid:
    //   1. Always fits in the terminal (cols <= max_cols, rows <= max_rows)
    //   2. Preserves the on-screen aspect ratio within 1 cell tolerance
    //      (i.e., rows ≈ cols * scale ± 1)

    #[test]
    fn fits_and_preserves_aspect_sweep_square_image() {
        // Square images at many terminal sizes
        for term_cols in [20, 40, 60, 80, 100, 120, 160, 200, 250, 300] {
            for term_rows in [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120] {
                let (cols, rows) = g(100, 100, None, None, term_cols, term_rows);
                let max_cols = term_cols as usize;
                let max_rows = (term_rows as usize).saturating_sub(1).max(1);

                // 1. Always fits
                assert!(cols <= max_cols,
                    "cols {} > max_cols {} at term {}x{}", cols, max_cols, term_cols, term_rows);
                assert!(rows <= max_rows,
                    "rows {} > max_rows {} at term {}x{}", rows, max_rows, term_cols, term_rows);

                // 2. Preserves aspect (square: scale = 0.5 -> rows ≈ cols * 0.5)
                let expected_rows = ((cols as f32) * 0.5).round() as usize;
                let diff = (rows as isize - expected_rows as isize).abs();
                assert!(diff <= 1,
                    "aspect drift: rows {} vs expected {} (diff={}) at term {}x{}, cols={}",
                    rows, expected_rows, diff, term_cols, term_rows, cols);
            }
        }
    }

    #[test]
    fn fits_and_preserves_aspect_sweep_landscape_image() {
        // 16:9 landscape images at many terminal sizes
        for term_cols in [20, 40, 60, 80, 100, 120, 160, 200] {
            for term_rows in [5, 10, 15, 20, 30, 40, 50, 60, 80] {
                let (cols, rows) = g(1920, 1080, None, None, term_cols, term_rows);
                let max_cols = term_cols as usize;
                let max_rows = (term_rows as usize).saturating_sub(1).max(1);

                // 1. Always fits
                assert!(cols <= max_cols,
                    "cols {} > max_cols {} at term {}x{}", cols, max_cols, term_cols, term_rows);
                assert!(rows <= max_rows,
                    "rows {} > max_rows {} at term {}x{}", rows, max_rows, term_cols, term_rows);

                // 2. Preserves aspect (16:9: scale = (1080/1920)*0.5 = 0.28125 -> rows ≈ cols * 0.28125)
                let scale = (1080.0_f32 / 1920.0) * 0.5;
                let expected_rows = ((cols as f32) * scale).round() as usize;
                let diff = (rows as isize - expected_rows as isize).abs();
                assert!(diff <= 1,
                    "aspect drift: rows {} vs expected {} (diff={}) at term {}x{}, cols={}, scale={}",
                    rows, expected_rows, diff, term_cols, term_rows, cols, scale);
            }
        }
    }

    #[test]
    fn fits_and_preserves_aspect_sweep_portrait_image() {
        // 3:4 portrait images at many terminal sizes
        for term_cols in [20, 40, 60, 80, 100, 120] {
            for term_rows in [10, 20, 30, 40, 50, 60, 80, 100, 150, 200] {
                let (cols, rows) = g(900, 1200, None, None, term_cols, term_rows);
                let max_cols = term_cols as usize;
                let max_rows = (term_rows as usize).saturating_sub(1).max(1);

                // 1. Always fits
                assert!(cols <= max_cols,
                    "cols {} > max_cols {} at term {}x{}", cols, max_cols, term_cols, term_rows);
                assert!(rows <= max_rows,
                    "rows {} > max_rows {} at term {}x{}", rows, max_rows, term_cols, term_rows);

                // 2. Preserves aspect (3:4: scale = (1200/900)*0.5 = 0.666... -> rows ≈ cols * 0.666)
                let scale = (1200.0_f32 / 900.0) * 0.5;
                let expected_rows = ((cols as f32) * scale).round() as usize;
                let diff = (rows as isize - expected_rows as isize).abs();
                assert!(diff <= 1,
                    "aspect drift: rows {} vs expected {} (diff={}) at term {}x{}, cols={}, scale={}",
                    rows, expected_rows, diff, term_cols, term_rows, cols, scale);
            }
        }
    }

    #[test]
    fn fits_and_preserves_aspect_extreme_terminals() {
        // Very wide, very short terminals
        let (cols, rows) = g(100, 100, None, None, 400, 5);
        assert!(cols <= 400 && rows <= 4);
        // With aspect 0.5 and max_rows=4, cols should be ~8
        assert!((cols as isize - 8).abs() <= 2);

        // Very tall, very narrow terminals
        let (cols, rows) = g(100, 100, None, None, 10, 200);
        assert!(cols <= 10 && rows <= 199);
        // With aspect 0.5 and cols=10, rows should be ~5
        assert!((rows as isize - 5).abs() <= 1);

        // Extreme landscape on short terminal
        let (cols, rows) = g(1920, 1080, None, None, 300, 8);
        assert!(cols <= 300 && rows <= 7);
        let scale = (1080.0_f32 / 1920.0) * 0.5;
        let expected_rows = ((cols as f32) * scale).round() as usize;
        assert!((rows as isize - expected_rows as isize).abs() <= 1);

        // Extreme portrait on narrow terminal
        let (cols, rows) = g(900, 1200, None, None, 12, 300);
        assert!(cols <= 12 && rows <= 299);
        let scale = (1200.0_f32 / 900.0) * 0.5;
        let expected_rows = ((cols as f32) * scale).round() as usize;
        assert!((rows as isize - expected_rows as isize).abs() <= 1);
    }

    #[test]
    fn whole_image_fits_no_cropping() {
        // This test explicitly checks that the computed grid, when used to
        // downsample the image via cell_source_rect, covers the entire source.
        //
        // The logic in cell_source_rect guarantees that every source pixel is
        // covered by at least one cell (no gaps). This test verifies the grid
        // dimensions produced by compute_image_grid_dimensions don't cause any
        // source dimension to exceed what the grid can represent.

        for (img_w, img_h) in [(100u32, 100u32), (1920, 1080), (1080, 1920), (400, 300), (300, 400)] {
            for term_cols in [10, 20, 40, 80, 120, 200] {
                for term_rows in [5, 10, 20, 40, 60, 100] {
                    let (cols, rows) = g(img_w, img_h, None, None, term_cols, term_rows);

                    // The grid covers the whole image - check by iterating all cells
                    // and verifying their union covers the full source dimensions
                    let mut covered_w = 0u32;
                    for col in 0..cols {
                        let (x0, x1, _, _) = cell_source_rect(col, 0, cols, rows, img_w, img_h);
                        covered_w = covered_w.max(x1);
                    }
                    assert_eq!(covered_w, img_w,
                        "width not fully covered: {} vs {} at term {}x{}, grid {}x{}, img {}x{}",
                        covered_w, img_w, term_cols, term_rows, cols, rows, img_w, img_h);

                    let mut covered_h = 0u32;
                    for row in 0..rows {
                        let (_, _, y0, y1) = cell_source_rect(0, row, cols, rows, img_w, img_h);
                        covered_h = covered_h.max(y1);
                    }
                    assert_eq!(covered_h, img_h,
                        "height not fully covered: {} vs {} at term {}x{}, grid {}x{}, img {}x{}",
                        covered_h, img_h, term_cols, term_rows, cols, rows, img_w, img_h);
                }
            }
        }
    }

    // --- Phase 2 refactor: cell_source_rect --------------------------------

    #[test]
    fn cell_rect_exact_grid_maps_1_to_1() {
        // 100x100 source onto a 100x100 grid: each cell maps to exactly one pixel.
        let (x0, x1, y0, y1) = cell_source_rect(5, 7, 100, 100, 100, 100);
        assert_eq!((x0, x1, y0, y1), (5, 6, 7, 8));
    }

    #[test]
    fn cell_rect_partitions_source_without_gaps_or_overlaps() {
        let cols = 40;
        let rows = 25;
        // Downsizing cases (grid <= source): rects partition the source exactly.
        for (sw, sh) in [(100u32, 480u32), (640, 480), (120, 120)] {
            let mut total_w = 0u32;
            for col in 0..cols {
                let (x0, x1, _, _) = cell_source_rect(col, 0, cols, rows, sw, sh);
                assert!(x1 > x0, "cell rect must never be empty");
                total_w += x1 - x0;
            }
            assert_eq!(total_w, sw, "downsample rects must partition source width");
            let mut total_h = 0u32;
            for row in 0..rows {
                let (_, _, y0, y1) = cell_source_rect(0, row, cols, rows, sw, sh);
                assert!(y1 > y0);
                total_h += y1 - y0;
            }
            assert_eq!(total_h, sh, "downsample rects must partition source height");
        }

        // Upscaling cases (grid > source): the empty-cell guard kicks in and may
        // overlap pixels, but every cell must still be non-empty and in-bounds.
        for sw in [1u32, 3, 40] {
            for sh in [1u32, 3, 25] {
                for col in 0..cols {
                    let (x0, x1, _, _) = cell_source_rect(col, 0, cols, rows, sw, sh);
                    assert!(x1 > x0);
                    assert!(x1 <= sw, "cell rect must stay within source");
                }
                for row in 0..rows {
                    let (_, _, y0, y1) = cell_source_rect(0, row, cols, rows, sw, sh);
                    assert!(y1 > y0);
                    assert!(y1 <= sh, "cell rect must stay within source");
                }
            }
        }
    }

    #[test]
    fn cell_rect_never_zero_even_when_grid_exceeds_source() {
        // Grid larger than source (upscaling case) must still yield non-empty rects.
        let (x0, x1, _, _) = cell_source_rect(10, 0, 200, 200, 3, 3);
        assert!(x1 > x0);
        // x0 stays clamped by the max(x0+1) guard.
        assert!(x1 >= 1);
    }

    #[test]
    fn load_rgb_frame_preserves_native_dims() {
        let path = std::path::PathBuf::from("test_circle.png");
        if !path.exists() {
            return; // fixture may be gitignored/absent; native-dims test is optional here
        }
        let f = load_rgb_frame(&path).expect("checked-in fixture should load");
        // test_circle.png is 100x100.
        assert_eq!(f.width, 100);
        assert_eq!(f.height, 100);
        assert_eq!(f.rgb_data.len(), 100 * 100 * 3);
    }
}
