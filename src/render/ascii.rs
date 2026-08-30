use super::luminance::rgb_to_luminance;
use super::{EdgeCellInfo, direction_to_char};
use std::io::Write;

pub const SHORT_RAMP: &str = " .:-=+*#%@";
pub const DETAILED_RAMP: &str = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
pub const BLOCK_RAMP: &str = " ░▒▓█";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    TrueColor,
    Grayscale,
    Monochrome,
}

/// Appends the ANSI truecolor sequence for `(r,g,b)` if it differs from the
/// last emitted color, updating `last_color`. Used by both the brightness-only
/// and edge-blended render paths so their color bytes can never drift apart.
#[inline(always)]
fn append_color_escape(
    output: &mut Vec<u8>,
    last_color: &mut Option<(u8, u8, u8)>,
    r: u8,
    g: u8,
    b: u8,
    lum: u8,
    color_mode: ColorMode,
) {
    match color_mode {
        ColorMode::TrueColor => {
            let color = (r, g, b);
            if *last_color != Some(color) {
                write!(output, "\x1b[38;2;{};{};{}m", r, g, b).unwrap();
                *last_color = Some(color);
            }
        }
        ColorMode::Grayscale => {
            let color = (lum, lum, lum);
            if *last_color != Some(color) {
                write!(output, "\x1b[38;2;{};{};{}m", lum, lum, lum).unwrap();
                *last_color = Some(color);
            }
        }
        ColorMode::Monochrome => {}
    }
}

pub struct AsciiRenderer {
    ramp: Vec<char>,
    color_mode: ColorMode,
}

impl AsciiRenderer {
    pub fn new(ramp_str: &str, color_mode: ColorMode, invert: bool) -> Self {
        let mut chars: Vec<char> = ramp_str.chars().collect();
        if chars.is_empty() {
            chars = vec![' '];
        }
        if invert {
            chars.reverse();
        }
        Self {
            ramp: chars,
            color_mode,
        }
    }

    /// Maps a luminance value [0, 255] to a character in the current ramp.
    #[inline(always)]
    pub fn luminance_to_char(&self, lum: u8) -> char {
        let max_idx = self.ramp.len() - 1;
        let idx = (lum as usize * max_idx) / 255;
        self.ramp[idx]
    }

    /// Renders a raw RGB24 frame (`width * height * 3` bytes) into a formatted
    /// ANSI terminal byte buffer, using brightness shading for every cell.
    ///
    /// This is the Phase 1 path; the video loop and the edge-free fallback use
    /// it. Edge cells are selected by `render_frame_with_edges` instead.
    ///
    /// Reuses the `output` buffer to prevent heap allocations every frame.
    pub fn render_frame(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        output: &mut Vec<u8>,
    ) {
        output.clear();

        if width == 0 || height == 0 || rgb_data.len() < width * height * 3 {
            return;
        }

        // Move cursor to top-left (0, 0)
        output.extend_from_slice(b"\x1b[H");

        let mut last_color: Option<(u8, u8, u8)> = None;
        let mut char_buf = [0u8; 4];

        for y in 0..height {
            let row_offset = y * width * 3;
            for x in 0..width {
                let pixel_offset = row_offset + x * 3;
                let r = rgb_data[pixel_offset];
                let g = rgb_data[pixel_offset + 1];
                let b = rgb_data[pixel_offset + 2];

                let lum = rgb_to_luminance(r, g, b);
                let ch = self.luminance_to_char(lum);

                append_color_escape(output, &mut last_color, r, g, b, lum, self.color_mode);

                let encoded = ch.encode_utf8(&mut char_buf);
                output.extend_from_slice(encoded.as_bytes());
            }
            if y + 1 < height {
                output.extend_from_slice(b"\r\n");
            }
        }

        // Reset color attributes at the end of the frame
        if self.color_mode != ColorMode::Monochrome {
            output.extend_from_slice(b"\x1b[0m");
        }
        // Clear any leftover content below the grid (handles terminal shrink)
        output.extend_from_slice(b"\x1b[0J");
    }

    /// Renders an RGB24 frame with Phase 2's adaptive blend: any cell with a
    /// detected edge renders a directional glyph (via `direction_to_char`),
    /// everything else falls back to brightness shading exactly as
    /// `render_frame` does.
    ///
    /// `edges` holds one `Option<EdgeCellInfo>` per character cell
    /// (`width * height` entries), produced by `compute_frame_edges` over the
    /// *full-resolution* source frame. Given a genuinely edge-free frame, the
    /// output of this method is byte-identical to `render_frame`'s — the §6.7
    /// golden test locks that in. Color assignment is untouched: every cell,
    /// edge or not, keeps its box-averaged color; only the glyph branches.
    pub fn render_frame_with_edges(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        edges: &[Option<EdgeCellInfo>],
        output: &mut Vec<u8>,
    ) {
        output.clear();

        if width == 0 || height == 0 || rgb_data.len() < width * height * 3 {
            return;
        }

        output.extend_from_slice(b"\x1b[H");

        let mut last_color: Option<(u8, u8, u8)> = None;
        let mut char_buf = [0u8; 4];
        let has_edges = edges.len() == width * height;

        for y in 0..height {
            let row_offset = y * width * 3;
            for x in 0..width {
                let cell_idx = y * width + x;
                let pixel_offset = row_offset + x * 3;
                let r = rgb_data[pixel_offset];
                let g = rgb_data[pixel_offset + 1];
                let b = rgb_data[pixel_offset + 2];

                let lum = rgb_to_luminance(r, g, b);
                let ch = match (has_edges, edges.get(cell_idx)) {
                    (true, Some(Some(e))) => direction_to_char(e.orientation_deg),
                    _ => self.luminance_to_char(lum),
                };

                append_color_escape(output, &mut last_color, r, g, b, lum, self.color_mode);

                let encoded = ch.encode_utf8(&mut char_buf);
                output.extend_from_slice(encoded.as_bytes());
            }
            if y + 1 < height {
                output.extend_from_slice(b"\r\n");
            }
        }

        if self.color_mode != ColorMode::Monochrome {
            output.extend_from_slice(b"\x1b[0m");
        }
        // Clear any leftover content below the grid (handles terminal shrink)
        output.extend_from_slice(b"\x1b[0J");
    }
}

#[cfg(test)]
mod tests {
    use crate::image_loader::cell_source_rect;
    use crate::render::edge::{EdgeCellInfo, compute_frame_edges};
    use super::*;

    #[test]
    fn test_ascii_mapping_bounds() {
        let renderer = AsciiRenderer::new(" @", ColorMode::Monochrome, false);
        assert_eq!(renderer.luminance_to_char(0), ' ');
        assert_eq!(renderer.luminance_to_char(255), '@');
    }

    #[test]
    fn test_inverted_ramp() {
        let renderer = AsciiRenderer::new(" @", ColorMode::Monochrome, true);
        assert_eq!(renderer.luminance_to_char(0), '@');
        assert_eq!(renderer.luminance_to_char(255), ' ');
    }

    #[test]
    fn test_empty_ramp_safe() {
        let renderer = AsciiRenderer::new("", ColorMode::Monochrome, false);
        assert_eq!(renderer.luminance_to_char(0), ' ');
        assert_eq!(renderer.luminance_to_char(255), ' ');
    }

    #[test]
    fn test_unicode_multibyte_ramp() {
        let renderer = AsciiRenderer::new(BLOCK_RAMP, ColorMode::Monochrome, false);
        assert_eq!(renderer.luminance_to_char(0), ' ');
        assert_eq!(renderer.luminance_to_char(255), '█');

        let mut output = Vec::new();
        let rgb_data = vec![255, 255, 255, 0, 0, 0];
        renderer.render_frame(&rgb_data, 2, 1, &mut output);

        let output_str = String::from_utf8(output).expect("Output must be valid UTF-8");
        assert!(output_str.contains('█'));
        assert!(output_str.contains(' '));
    }

    #[test]
    fn test_stress_ramp_all_luminance_values() {
        let ramps = [SHORT_RAMP, DETAILED_RAMP, BLOCK_RAMP, "ABC", "⚡🔥🚀✨", "X"];
        for ramp in ramps {
            let renderer_normal = AsciiRenderer::new(ramp, ColorMode::Monochrome, false);
            let renderer_inverted = AsciiRenderer::new(ramp, ColorMode::Monochrome, true);
            for lum in 0..=255u8 {
                let ch_norm = renderer_normal.luminance_to_char(lum);
                let ch_inv = renderer_inverted.luminance_to_char(lum);
                assert!(ramp.contains(ch_norm));
                assert!(ramp.contains(ch_inv));
            }
        }
    }

    // ── §6.7 Golden tests ──────────────────────────────────────────────────────

    /// Box-average downsamples a full-res RGB buffer to a grid of `(cols, rows)`
    /// using `cell_source_rect` — the same rounding logic the real pipeline
    /// shares between color and edge aggregation (§0 of the Phase 2 plan).
    fn downsample_box(
        full_rgb: &[u8],
        src_w: usize,
        src_h: usize,
        cols: usize,
        rows: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; cols * rows * 3];
        for r in 0..rows {
            for c in 0..cols {
                let (x0, x1, y0, y1) =
                    crate::image_loader::cell_source_rect(c, r, cols, rows, src_w as u32, src_h as u32);
                let mut acc = [0u64; 3];
                let mut n = 0u64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = (y as usize * src_w + x as usize) * 3;
                        acc[0] += full_rgb[i] as u64;
                        acc[1] += full_rgb[i + 1] as u64;
                        acc[2] += full_rgb[i + 2] as u64;
                        n += 1;
                    }
                }
                let idx = (r * cols + c) * 3;
                out[idx] = (acc[0] / n) as u8;
                out[idx + 1] = (acc[1] / n) as u8;
                out[idx + 2] = (acc[2] / n) as u8;
            }
        }
        out
    }

    /// Renders a full-res image through the Phase 2 pipeline (edge detect +
    /// adaptive blend) in Monochrome mode with ramp `" X"`, returning only the
    /// glyph matrix as a String (no color escapes in Monochrome).
    fn render_pipeline(
        full_rgb: &[u8],
        src_w: usize,
        src_h: usize,
        cols: usize,
        rows: usize,
    ) -> String {
        let edges = compute_frame_edges(full_rgb, src_w, src_h, cols, rows);
        let ds = downsample_box(full_rgb, src_w, src_h, cols, rows);
        let renderer = AsciiRenderer::new(" X", ColorMode::Monochrome, false);
        let mut out = Vec::new();
        renderer.render_frame_with_edges(&ds, cols, rows, &edges, &mut out);
        String::from_utf8(out).expect("Monochrome output must be valid UTF-8")
    }

    /// The glyph matrix portion (after `\x1b[H`) of a Monochrome render,
    /// with all ANSI escapes and newlines stripped, as a Vec<&str> per row.
    fn glyph_matrix(output: &str, cols: usize) -> Vec<&str> {
        let body = output.strip_prefix("\x1b[H").unwrap_or(output);
        body.split("\r\n")
            .map(|line| &line[..cols])
            .collect()
    }

    /// §6.7 flat-image regression: `render_frame` (Phase 1 path) and
    /// `render_frame_with_edges` on a uniform image must produce byte-identical
    /// output. This proves the adaptive blend correctly falls back to brightness
    /// shading when there are genuinely no edges.
    #[test]
    fn golden_blend_flat_matches_phase1() {
        let (w, h, cols, rows) = (32, 32, 8, 8);
        let flat = vec![128u8; w * h * 3];
        let edges = compute_frame_edges(&flat, w, h, cols, rows);
        assert!(edges.iter().all(Option::is_none), "uniform image must produce no edges");

        let ds = downsample_box(&flat, w, h, cols, rows);
        let r = AsciiRenderer::new(SHORT_RAMP, ColorMode::TrueColor, false);

        let mut buf_phase1 = Vec::new();
        let mut buf_phase2 = Vec::new();
        r.render_frame(&ds, cols, rows, &mut buf_phase1);
        r.render_frame_with_edges(&ds, cols, rows, &edges, &mut buf_phase2);
        assert_eq!(
            buf_phase1, buf_phase2,
            "flat image: render_frame and render_frame_with_edges must be byte-identical"
        );
    }

    /// §6.7 vertical boundary (left dark, right bright) → boundary cells must
    /// render `|` (vertical stroke glyph), and only boundary cells.
    ///
    /// 32×32 full-res, 8×8 grid (cell = 4×4 source px). Boundary at x=17
    /// straddles cell column 4 (x 16..20), so every cell in column 4 must be
    /// `|`, cells 0–3 must be `' '` (dark), cells 5–7 must be `'X'` (bright).
    /// The golden string is the exact expected output — pinned byte-for-byte.
    #[test]
    fn golden_blend_vertical_boundary_renders_pipe() {
        let (w, h, cols, rows) = (32, 32, 8, 8);
        let mut full_rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let c: u8 = if x < 17 { 0 } else { 255 };
                let i = (y * w + x) * 3;
                full_rgb[i] = c;
                full_rgb[i + 1] = c;
                full_rgb[i + 2] = c;
            }
        }
        let output = render_pipeline(&full_rgb, w, h, cols, rows);
        let matrix = glyph_matrix(&output, cols);
        // Expected: every row is "    |XXX" (col 4 = vertical edge, cols 0–3 dark, 5–7 bright)
        let expected_row = "    |XXX";
        assert_eq!(matrix.len(), rows);
        for (r, line) in matrix.iter().enumerate() {
            assert_eq!(
                *line,
                expected_row,
                "row {} glyph mismatch: got {:?}, expected {:?}",
                r,
                line,
                expected_row
            );
        }
    }

    /// §6.7 horizontal boundary (top dark, bottom bright) → boundary row must
    /// render `-` (horizontal stroke glyph), and only boundary cells.
    ///
    /// 32×32 full-res, 8×8 grid (cell = 4×4). Boundary at y=17 straddles row 4
    /// (y 16..20), so all cells in row 4 are `-`; rows 0–3 are `' '` (dark),
    /// rows 5–7 are `'X'` (bright).
    #[test]
    fn golden_blend_horizontal_boundary_renders_dash() {
        let (w, h, cols, rows) = (32, 32, 8, 8);
        let mut full_rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let c: u8 = if y < 17 { 0 } else { 255 };
                let i = (y * w + x) * 3;
                full_rgb[i] = c;
                full_rgb[i + 1] = c;
                full_rgb[i + 2] = c;
            }
        }
        let output = render_pipeline(&full_rgb, w, h, cols, rows);
        let matrix = glyph_matrix(&output, cols);
        // Expected: rows 0–3 all dark → "        ", row 4 all '-', rows 5–7 all bright → "XXXXXXXX"
        let expected: Vec<&str> = (0..rows)
            .map(|r| match r {
                0..=3 => "        ",
                4 => "--------",
                5..=7 => "XXXXXXXX",
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(matrix.len(), rows);
        for (r, line) in matrix.iter().enumerate() {
            assert_eq!(
                *line,
                expected[r],
                "row {} glyph mismatch: got {:?}, expected {:?}",
                r,
                line,
                expected[r]
            );
        }
    }

    /// §6.7 diagonal boundary (dark upper-left, bright lower-right) → cells on
    /// the anti-diagonal line `c+r==9` must render `/` (the '/' stroke glyph),
    /// cells above-left are `' '` (dark), cells below-right are `'X'` (bright).
    ///
    /// 40×40 full-res, 10×10 grid (cell = 4×4). Boundary `x+y==40` straddles
    /// exactly the anti-diagonal cells `c+r==9`, forming a clean `/` stroke.
    /// This pins the diagonal convention against a known-stroke shape (the same
    /// geometry as `test_diagonal_edge_maps_to_slash` but through the full
    /// pipeline including per-cell aggregation and glyph rendering).
    #[test]
    fn golden_blend_diagonal_boundary_renders_slash() {
        let (n, cols, rows) = (40, 10, 10);
        let mut full_rgb = vec![0u8; n * n * 3];
        for y in 0..n {
            for x in 0..n {
                let c: u8 = if (x + y) < 40 { 0 } else { 255 };
                let i = (y * n + x) * 3;
                full_rgb[i] = c;
                full_rgb[i + 1] = c;
                full_rgb[i + 2] = c;
            }
        }
        let output = render_pipeline(&full_rgb, n, n, cols, rows);
        let matrix = glyph_matrix(&output, cols);
        // Expected matrix: row r → (9-r) spaces + '/' + r X's
        assert_eq!(matrix.len(), rows);
        for r in 0..rows {
            let mut expected = String::with_capacity(cols);
            for c in 0..cols {
                expected.push(match (c + r).cmp(&9) {
                    std::cmp::Ordering::Less => ' ',   // dark region
                    std::cmp::Ordering::Equal => '/',   // edge stroke
                    std::cmp::Ordering::Greater => 'X', // bright region
                });
            }
            assert_eq!(
                matrix[r],
                expected,
                "row {} glyph mismatch: got {:?}, expected {:?}",
                r,
                matrix[r],
                expected
            );
        }
    }
}
