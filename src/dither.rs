//! Ordered (Bayer-matrix) dithering — deliberately NOT error-diffusion. See
//! design decision D5 in the Phase 4 plan: error diffusion (Floyd–Steinberg)
//! propagates each cell's quantization error into its not-yet-processed
//! neighbors, making a cell's dithered output depend on distant cells' true
//! values. Frame to frame, that dependency means a static cell registers as
//! "dirty" whenever anything else in its processing neighborhood changes —
//! silently destroying the diff optimization, exactly in the reduced-palette
//! modes that need the bandwidth savings most.
//!
//! Ordered dithering has no such dependency: a cell's dithered output depends
//! only on its own true color and its fixed (x, y) position. Identical content
//! at the same position always dithers to the identical palette entry, so the
//! diff stays stable. This is a slightly worse static-image-quality algorithm
//! than error diffusion — and the right choice for a real-time video renderer
//! whose chief concern is frame-to-frame diffability.

use crate::render::grid::Rgb;
use crate::palette::Palette;

/// The 4×4 Bayer matrix.
pub const BAYER_4X4: [[u8; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 5, 13],
];

/// The Bayer threshold at cell `(x, y)` in [-0.5, +0.5).
#[inline]
pub fn bayer_threshold(x: usize, y: usize) -> f32 {
    BAYER_4X4[y % 4][x % 4] as f32 / 16.0 - 0.5
}

/// Quantizes `color` to the nearest palette entry, nudging each channel by the
/// cell-position-dependent Bayer threshold before the redmean nearest-match.
///
/// **Correctness property (the whole reason D5 chose this):** for a fixed
/// `(x, y)`, the output depends *only* on `color` — never on any neighboring
/// cell's value, current or historical.
pub fn ordered_dither_quantize(color: Rgb, x: usize, y: usize, palette: &Palette) -> u8 {
    let threshold = bayer_threshold(x, y);
    let step = palette.approx_step;
    let nudge = |v: u8| (v as f32 + threshold * step).clamp(0.0, 255.0) as u8;
    let nudged = Rgb::new(nudge(color.r), nudge(color.g), nudge(color.b));
    palette.nearest_index(nudged) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_matrix_dims_and_coverage() {
        // 4x4, values 0..=15 each exactly once.
        let mut seen = [false; 16];
        for y in 0..4 {
            for x in 0..4 {
                let v = BAYER_4X4[y][x] as usize;
                assert!(v < 16);
                assert!(!seen[v], "duplicate Bayer value {}", v);
                seen[v] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "all 0..=15 present");
    }

    #[test]
    fn threshold_range() {
        for y in 0..4 {
            for x in 0..4 {
                let t = bayer_threshold(x, y);
                assert!((-0.5..0.5).contains(&t), "threshold {} at ({},{})", t, x, y);
            }
        }
    }

    #[test]
    fn dither_determinism_same_input_same_output() {
        let p256 = Palette::xterm256();
        let p16 = Palette::basic16();
        for color in [
            Rgb::new(0, 0, 0),
            Rgb::new(255, 255, 255),
            Rgb::new(128, 64, 32),
            Rgb::new(200, 10, 90),
        ] {
            for palette in [&p256, &p16] {
                for x in 0..8 {
                    for y in 0..8 {
                        let a = ordered_dither_quantize(color, x, y, palette);
                        let b = ordered_dither_quantize(color, x, y, palette);
                        assert_eq!(a, b, "must be deterministic");
                    }
                }
            }
        }
    }

    #[test]
    fn dither_neighbor_independence() {
        // THE critical regression test (D5): the same target cell's true color,
        // with different surrounding neighbor colors in two otherwise-identical
        // frames, must yield the same dithered result. This is the property
        // diffing depends on — test it directly.
        let p256 = Palette::xterm256();
        let target_pos = (3usize, 2usize);
        for fg_attempt in 0..50u8 {
            let target = Rgb::new(fg_attempt, 100, 50);
            let mut a = ordered_dither_quantize(target, target_pos.0, target_pos.1, &p256);
            // Perturb the environment: a would-be "previous" neighbor's error
            // path must not influence our target cell's result.
            let _ = ordered_dither_quantize(Rgb::new(0, 0, 0), 0, 0, &p256);
            let _ = ordered_dither_quantize(Rgb::new(255, 255, 255), 4, 4, &p256);
            let _ = ordered_dither_quantize(Rgb::new(50, 200, 30), 2, 2, &p256);
            let b = ordered_dither_quantize(target, target_pos.0, target_pos.1, &p256);
            assert_eq!(a, b, "neighbor writes must not affect target cell's dither");
            let _ = a; // silence unused-mut in path where assignment is re-checked
            a = b;
        }
    }

    #[test]
    fn dither_index_stays_in_palette_bounds() {
        let p16 = Palette::basic16();
        for x in 0..16 {
            for y in 0..16 {
                let idx = ordered_dither_quantize(Rgb::new(200, 100, 50), x, y, &p16);
                assert!((idx as usize) < p16.entries.len());
            }
        }
    }
}
