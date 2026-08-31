//! Materialized character grid — the bridge between Phase 2's per-cell
//! computations and Phase 4's diff/compression machinery.
//!
//! Phase 1–3 rendered straight to an ANSI byte stream; there was no explicit
//! `CharGrid`. Phase 4 needs to diff *this* frame's cells against *last*
//! frame's cells, so the content is first materialized into a reversible,
//! value-comparable `CharGrid` of `CharCell`s, and only *then* serialized to
//! bytes. The grid stores each cell's *content* color (the box-averaged RGB,
//! or the grayscale-mapped RGB for Grayscale mode) — pre-resolution, so that a
//! diff fires whenever the underlying content genuinely changed, independent of
//! how a later palette-resolution pass maps it (see `diff.rs`).

use crate::render::ascii::ColorMode;
use crate::render::edge::{EdgeCellInfo, direction_to_char};
use crate::render::AsciiRenderer;

/// A single 24-bit RGB color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// One character cell: a glyph plus its content color.
///
/// Equality is plain field equality (`glyph == glyph && color == color`) —
/// cheap and glyph-set-agnostic, so Phase 7's half-block/Braille modes produce
/// `CharCell`s through the same struct with no diffing changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharCell {
    pub glyph: char,
    pub color: Rgb,
}

/// The sentinel used to pre-fill a `DoubleGrid`'s buffers (§3 of the Phase 4
/// plan). The `'\0'` glyph is one no real ramp or edge-glyph set will ever
/// produce, so the very first real frame naturally diffs against it as a full
/// redraw — with no special-cased "first frame" branch anywhere.
pub const SENTINEL_CELL: CharCell = CharCell {
    glyph: '\0',
    color: Rgb::new(0, 0, 0),
};

/// A 2D grid of `CharCell`s laid out row-major.
#[derive(Debug, Clone)]
pub struct CharGrid {
    cols: usize,
    rows: usize,
    cells: Vec<CharCell>,
}

impl CharGrid {
    /// Creates a `rows`×`cols` grid pre-filled with `fill`.
    pub fn new(cols: usize, rows: usize, fill: CharCell) -> Self {
        Self {
            cols,
            rows,
            cells: vec![fill; cols * rows],
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Indexed cell (row-major index `row * cols + col`).
    #[inline]
    pub fn get(&self, idx: usize) -> CharCell {
        self.cells[idx]
    }

    #[inline]
    pub fn set(&mut self, idx: usize, cell: CharCell) {
        self.cells[idx] = cell;
    }

    /// Cell at `(row, col)`.
    #[inline]
    pub fn cell(&self, row: usize, col: usize) -> CharCell {
        self.cells[row * self.cols + col]
    }
}

impl<'a> From<&'a CharGrid> for Vec<CharCell> {
    fn from(g: &'a CharGrid) -> Self {
        g.cells.clone()
    }
}

/// Fills the write-target grid with one `CharCell` per character cell,
/// reproducing Phase 1/2's exact glyph + color computation:
///
/// - **glyph**: a directional stroke glyph if the cell has a detected edge
///   (via `direction_to_char`), else `renderer.luminance_to_char(lum)`.
/// - **content color**: the box-averaged `(r, g, b)` for TrueColor mode; the
///   grayscale-mapped `(lum, lum, lum)` for Grayscale mode; and (for
///   Monochrome) `(lum, lum, lum)` — luma is the only thing that affects a
///   monochrome glyph, so tracking it as the color keeps the diff sensitive to
///   exactly the changes that would alter the rendered output.
///
/// This mirrors the glyph/color computation the Phase 1–3 emitter did inline,
/// so switching the video loop over to the grid path causes no visual change.
pub fn build_char_grid_into(
    grid: &mut crate::render::grid::CharGrid,
    cell_luma: &[f32],
    cell_color: &[(u8, u8, u8)],
    edges: &[Option<EdgeCellInfo>],
    renderer: &AsciiRenderer,
    color_mode: ColorMode,
) {
    let n = grid.len();
    debug_assert!(n <= cell_luma.len());
    debug_assert!(n <= cell_color.len());
    for idx in 0..n {
        let lum = cell_luma[idx];
        let (r, g, b) = match cell_color.get(idx) {
            Some(&c) => c,
            None => (0, 0, 0),
        };

        let glyph = match edges.get(idx) {
            Some(Some(e)) => direction_to_char(e.orientation_deg),
            _ => renderer.luminance_to_char(lum as u8),
        };

        let color = match color_mode {
            ColorMode::TrueColor => crate::render::grid::Rgb::new(r, g, b),
            ColorMode::Grayscale => {
                let lum8 = crate::render::luminance::rgb_to_luminance(r, g, b);
                crate::render::grid::Rgb::new(lum8, lum8, lum8)
            }
            ColorMode::Monochrome => {
                let lum8 = lum as u8;
                crate::render::grid::Rgb::new(lum8, lum8, lum8)
            }
        };

        grid.set(idx, CharCell { glyph, color });
    }
}
