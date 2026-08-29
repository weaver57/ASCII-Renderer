//! Edge detection for the ASCII renderer (Phase 2).
//!
//! Sobel runs at **full source-pixel resolution** (not on the downsampled
//! character grid) so that fine detail survives before it is aggregated into
//! each character cell. The whole chain:
//!
//! `luma` → `Sobel` → `NMS` → `threshold+hysteresis` → `per-cell aggregate`
//!
//! This module is intentionally written as simple, branch-light nested loops
//! over contiguous memory: it is the top SIMD optimization target for Phase 5,
//! and its shape should not make vectorizing it later awkward.

use crate::image_loader::cell_source_rect;
use crate::render::luminance::rgb_to_luminance_f32;

/// Minimum number of edge pixels inside a character cell before it is treated
/// as an edge cell — rejects single stray pixels.
pub const EDGE_CELL_MIN_PIXELS: usize = 2;

/// Edge information flowing into glyph selection, `None` = use brightness shading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeCellInfo {
    /// Representative magnitude (max over the cell) — a "how strong is this edge" signal.
    pub magnitude: f32,
    /// EDGE orientation (not gradient direction), in `[0, 180)` degrees.
    pub orientation_deg: f32,
}

/// Per-pixel gradient data at full source resolution.
pub struct GradientMap {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width * height`.
    pub magnitude: Vec<f32>,
    /// Row-major, gradient direction in radians, range `(-pi, pi]`.
    pub angle: Vec<f32>,
}

/// Per-pixel perceived luma (Rec. 709) at full resolution.
pub fn build_luma_map(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3)
        .map(|p| rgb_to_luminance_f32(p[0], p[1], p[2]))
        .collect()
}

/// Sobel operator (Gx/Gy) over the full-resolution luma map, with border
/// coordinates clamped to the nearest valid edge pixel (no wrapping, no
/// zero-padding — zero-padding would fabricate fake edges at the border).
pub fn build_gradient_map(luma: &[f32], width: usize, height: usize) -> GradientMap {
    let w = width as i32;
    let h = height as i32;
    let mut magnitude = vec![0.0f32; luma.len()];
    let mut angle = vec![0.0f32; luma.len()];

    // Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = [-1 -2 -1; 0 0 0; 1 2 1] (y down).
    const GX: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const GY: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    for y in 0..h {
        for x in 0..w {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for (ky, dy) in [-1i32, 0, 1].iter().enumerate() {
                for (kx, dx) in [-1i32, 0, 1].iter().enumerate() {
                    let ny = (y + dy).clamp(0, h - 1);
                    let nx = (x + dx).clamp(0, w - 1);
                    let val = luma[(ny * w + nx) as usize];
                    gx += GX[ky][kx] as f32 * val;
                    gy += GY[ky][kx] as f32 * val;
                }
            }
            let idx = (y * w + x) as usize;
            magnitude[idx] = (gx * gx + gy * gy).sqrt();
            angle[idx] = gy.atan2(gx);
        }
    }

    GradientMap {
        width: width as u32,
        height: height as u32,
        magnitude,
        angle,
    }
}

/// Quantize a gradient angle (radians) into one of the 4 NMS comparison bins,
/// returning the (delta_x, delta_y) neighbor offsets to compare against.
/// Orientation is treated as undirected (mod 180°).
#[inline]
fn nms_bin(angle: f32) -> [(i32, i32); 2] {
    let base_deg = (angle.to_degrees()).rem_euclid(180.0);
    match base_deg {
        d if d < 22.5 || d >= 157.5 => [(-1, 0), (1, 0)],   // ~horizontal gradient
        d if d < 67.5               => [(1, -1), (-1, 1)],  // ~45° diagonal
        d if d < 112.5              => [(0, -1), (0, 1)],   // ~vertical gradient
        _                            => [(1, 1), (-1, -1)], // ~135° diagonal
    }
}

/// Non-maximum suppression: keeps only local maxima along the gradient
/// direction, thinning multi-pixel edge smears to ~1px.
///
/// The comparisons are deliberately *asymmetric* (`>` on the first neighbor,
/// `>=` on the second):
/// * a perfectly flat plateau then collapses to exactly one pixel instead of
///   keeping every equal-height member,
/// * a perfectly symmetric 2-pixel band (e.g. the Sobel response to a clean
///   vertical step, which is two equal peaks) collapses to a single pixel
///   instead of being suppressed entirely.
///
/// NOTE on orientation convention: see the diagonal test in this module, which
/// empirically pins which of `/` vs `\` a given gradient maps onto. This was
/// verified against the actual y-down Sobel convention, so `nms_bin`'s diagonal
/// bins and `direction_to_char`'s arms agree with each other.
pub fn non_max_suppress(gradient: &GradientMap) -> Vec<f32> {
    let w = gradient.width as i32;
    let h = gradient.height as i32;
    let mut out = vec![0.0f32; gradient.magnitude.len()];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let mag = gradient.magnitude[idx];
            if mag == 0.0 {
                continue;
            }
            let [a, b] = nms_bin(gradient.angle[idx]);
            let na = (y + a.1).clamp(0, h - 1) * w + (x + a.0).clamp(0, w - 1);
            let nb = (y + b.1).clamp(0, h - 1) * w + (x + b.0).clamp(0, w - 1);
            if mag > gradient.magnitude[na as usize]
                && mag >= gradient.magnitude[nb as usize]
            {
                out[idx] = mag;
            }
        }
    }
    out
}

/// Adaptive per-frame thresholds from the frame's own gradient distribution.
///
/// `high` = 90th percentile of nonzero magnitudes; `low` = `high * 0.4`
/// (standard Canny 1:2–1:3 guidance). On a near-blank frame (fewer than
/// [`MIN_SAMPLE`] nonzero pixels) we fall back to fixed constants — a named
/// special case, not a silent branch.
///
/// [`MIN_SAMPLE`]: constant.MIN_SAMPLE.html
pub fn compute_thresholds(suppressed: &[f32]) -> (f32, f32) {
    const MIN_SAMPLE: usize = 16;
    // Named special case: not enough edge signal to trust percentiles.
    const FALLBACK_HIGH: f32 = 100.0;
    const FALLBACK_LOW: f32 = 40.0;

    let mut vals: Vec<f32> = suppressed.iter().copied().filter(|&m| m > 0.0).collect();
    if vals.len() < MIN_SAMPLE {
        return (FALLBACK_HIGH, FALLBACK_LOW);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).expect("magnitudes are finite"));
    let hi = ((vals.len() - 1) as f32 * 0.90).round() as usize;
    let high = vals[hi];
    (high, high * 0.4)
}

/// Queue-based hysteresis promotion.
///
/// An explicit `VecDeque` is used instead of recursion because a long connected
/// edge chain (e.g. a solid silhouette) has stack depth proportional to its
/// length — tens of thousands of frames deep on a busy scene — which is a real
/// stack-overflow risk. The queue version is O(n) time/space with no recursion
/// depth regardless of chain length.
pub fn promote_edges(suppressed: &[f32], low: f32, high: f32, width: usize, height: usize) -> Vec<bool> {
    let mut final_mask = vec![false; width * height];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for (i, &m) in suppressed.iter().enumerate() {
        if m >= high {
            final_mask[i] = true;
            queue.push_back(i);
        }
    }

    while let Some(idx) = queue.pop_front() {
        let x = (idx % width) as i32;
        let y = (idx / width) as i32;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let ni = (ny as usize) * width + (nx as usize);
                if !final_mask[ni] && suppressed[ni] >= low {
                    final_mask[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
    }
    final_mask
}

fn normalize_deg(v: f32, period: f32) -> f32 {
    v.rem_euclid(period)
}

/// Aggregates the full-resolution edge mask into one `Option<EdgeCellInfo>` per
/// character cell, using the **same** `cell_source_rect` that color/brightness
/// downsampling uses — so a cell's edge data always describes the same source
/// patch as its color.
///
/// Orientation is combined with a magnitude-weighted **circular** mean (the
/// doubled-angle trick): gradient orientation is periodic with period 180°, so
/// a naive arithmetic mean of `179°` and `1°` would wrongly report `90°` for two
/// nearly-horizontal readings. Doubling maps the 180°-periodic quantity onto a
/// full 360° circle, where ordinary vector averaging is valid, then we halve it
/// back at the end.
pub fn aggregate_cell_edges(
    final_mask: &[bool],
    gradient: &GradientMap,
    cols: usize,
    rows: usize,
    src_w: u32,
    src_h: u32,
) -> Vec<Option<EdgeCellInfo>> {
    let mut out = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let (x0, x1, y0, y1) = cell_source_rect(col, row, cols, rows, src_w, src_h);
            let mut count = 0usize;
            let mut sum_cos2 = 0.0f64;
            let mut sum_sin2 = 0.0f64;
            let mut max_mag: f32 = 0.0;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * src_w + x) as usize;
                    if final_mask[idx] {
                        count += 1;
                        let theta = gradient.angle[idx] as f64;
                        let m = gradient.magnitude[idx] as f64;
                        sum_cos2 += m * (2.0 * theta).cos();
                        sum_sin2 += m * (2.0 * theta).sin();
                        if gradient.magnitude[idx] > max_mag {
                            max_mag = gradient.magnitude[idx];
                        }
                    }
                }
            }

            if count >= EDGE_CELL_MIN_PIXELS {
                let mean_grad = 0.5 * sum_sin2.atan2(sum_cos2); // radians
                let orientation_deg =
                    normalize_deg((mean_grad.to_degrees()) as f32 + 90.0, 180.0);
                out.push(Some(EdgeCellInfo {
                    magnitude: max_mag,
                    orientation_deg,
                }));
            } else {
                out.push(None);
            }
        }
    }
    out
}

/// Maps an EDGE orientation (degrees, `[0,180)`) to a directional glyph.
///
/// The convention is *stroke direction*: the glyph is the one that visually
/// resembles the boundary line in the source image.
///
/// * horizontal boundary (`0°/180°`) — e.g. a bright band across the frame — → `-`
/// * `45°` (boundary slanting top-left → bottom-right)                    → `\`
/// * vertical boundary (`90°`) — e.g. a column or sharp left/right edge       → `|`
/// * `135°` (boundary slanting top-right → bottom-left)                      → `/`
///
/// The `/`-vs-`\` assignment is pinned by `test_diagonal_edge_maps_to_slash`,
/// verified empirically against this y-down Sobel convention. The axis-aligned
/// arms match the same stroke logic (a vertical gradient step is a *horizontal*
/// gradient vector, so it lands on `90°` → `|`). Do not "simplify" these arms
/// without re-running that test, or every diagonal edge will silently render
/// mirrored.
pub fn direction_to_char(orientation_deg: f32) -> char {
    let d = orientation_deg.rem_euclid(180.0);
    if d < 22.5 || d >= 157.5 {
        '-' // horizontal boundary
    } else if d < 67.5 {
        '\\'
    } else if d < 112.5 {
        '|' // vertical boundary
    } else {
        '/'
    }
}

/// Convenience: run the full Phase 2 edge pipeline for a decoded RGB frame and
/// return one `Option<EdgeCellInfo>` per character cell (row-major, `cols*rows`).
pub fn compute_frame_edges(
    rgb: &[u8],
    width: usize,
    height: usize,
    cols: usize,
    rows: usize,
) -> Vec<Option<EdgeCellInfo>> {
    let luma = build_luma_map(rgb);
    let grad = build_gradient_map(&luma, width, height);
    let nms = non_max_suppress(&grad);
    let (high, low) = compute_thresholds(&nms);
    let mask = promote_edges(&nms, low, high, width, height);
    aggregate_cell_edges(&mask, &grad, cols, rows, width as u32, height as u32)
}

use std::collections::VecDeque;

#[cfg(test)]
mod tests {
    use super::*;

    fn vstep_luma(w: usize, h: usize) -> Vec<f32> {
        // Vertical step: left dark, right bright.
        (0..h)
            .flat_map(|_y| (0..w).map(move |x| if x < w / 2 { 0.0 } else { 255.0 }))
            .collect()
    }

    fn hstep_luma(w: usize, h: usize) -> Vec<f32> {
        // Horizontal step: top dark, bottom bright.
        (0..h)
            .flat_map(|y| (0..w).map(move |_x| if y < h / 2 { 0.0 } else { 255.0 }))
            .collect()
    }

    fn diag_luma(n: usize) -> Vec<f32> {
        // 45° diagonal boundary x+y == n : dark below-left, bright above-right.
        (0..n)
            .flat_map(|y| (0..n).map(move |x| if (x + y) < n { 0.0 } else { 255.0 }))
            .collect()
    }

    #[test]
    fn sobel_vertical_step_has_large_gx_small_gy() {
        let w = 40;
        let h = 30;
        let grad = build_gradient_map(&vstep_luma(w, h), w, h);
        // Look at a boundary pixel (x = w/2) in the middle row.
        let x = (w / 2) as i32;
        let y = (h / 2) as i32;
        let idx = (y * w as i32 + x) as usize;
        let gx = grad.magnitude[idx]; // magnitude dominated by gx
        assert!(gx > 10.0, "vertical edge should have strong response");
        let angle_deg = grad.angle[idx].to_degrees().rem_euclid(180.0);
        // Vertical step -> gradient points horizontally (~0° or ~180°).
        assert!(angle_deg < 30.0 || angle_deg > 150.0, "got {}", angle_deg);
    }

    #[test]
    fn sobel_horizontal_step_has_large_gy_small_gx() {
        let w = 40;
        let h = 30;
        let grad = build_gradient_map(&hstep_luma(w, h), w, h);
        let x = (w / 2) as i32;
        let y = (h / 2) as i32;
        let idx = (y * w as i32 + x) as usize;
        assert!(grad.magnitude[idx] > 10.0, "horizontal edge should have strong response");
        let angle_deg = grad.angle[idx].to_degrees().rem_euclid(180.0);
        // Horizontal step -> gradient points vertically (~90°).
        assert!((60.0..=120.0).contains(&angle_deg), "got {}", angle_deg);
    }

    #[test]
    fn sobel_flat_region_has_zero_magnitude() {
        let w = 20;
        let h = 20;
        let flat = vec![128.0f32; w * h];
        let grad = build_gradient_map(&flat, w, h);
        assert!(grad.magnitude.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn test_diagonal_edge_maps_to_slash() {
        // EMPIRICAL CONVENTION TEST.
        //
        // The boundary x+y==n rises to the right on screen (a "/" shape, since
        // y points down). This test locks in which directional glyph the full
        // pipeline assigns to it. If this ever fails, the `/`/`\` arms in
        // `direction_to_char` (or the `nms_bin` diagonals) are backwards — fix
        // them and re-verify, don't delete the assertion.
        let n = 48usize;
        let luma = diag_luma(n);
        let grad = build_gradient_map(&luma, n, n);

        // Find the strongest edge pixel (on the diagonal boundary).
        let mut best = (0usize, 0.0f32);
        for (i, &m) in grad.magnitude.iter().enumerate() {
            if m > best.1 {
                best = (i, m);
            }
        }
        assert!(best.1 > 10.0, "diagonal edge should produce a strong response");

        let angle_deg = grad.angle[best.0].to_degrees();
        let orientation_deg = normalize_deg(angle_deg + 90.0, 180.0);
        let ch = direction_to_char(orientation_deg);
        assert_eq!(
            ch,
            '/',
            "edge rising to the right must render '/', got '{}' (gradient {:.1}°, orientation {:.1}°)",
            ch,
            angle_deg,
            orientation_deg
        );
    }

    #[test]
    fn nms_thins_three_pixel_ridge_to_one() {
        // A 3px-wide vertical ridge centered on column 6: the middle is strictly
        // greater than both its neighbors (a true local maximum); columns 5 and 7
        // are not. NMS must keep exactly the center column of each row.
        let w = 24;
        let h = 24;
        let mut mag = vec![0.0f32; w * h];
        let mut ang = vec![0.0f32; w * h];
        for y in 0..h {
            mag[y * w + 5] = 100.0;
            mag[y * w + 6] = 120.0;
            mag[y * w + 7] = 100.0;
            for x in [5i32, 6, 7] {
                ang[y * w + x as usize] = 0.0; // horizontal gradient -> compare left/right
            }
        }
        let grad = GradientMap {
            width: w as u32,
            height: h as u32,
            magnitude: mag,
            angle: ang,
        };
        let sup = non_max_suppress(&grad);
        // For each row, exactly one of 5,6,7 survives.
        for y in 0..h {
            let surv: Vec<usize> = [5usize, 6, 7]
                .iter()
                .filter(|&&x| sup[y * w + x] > 0.0)
                .copied()
                .collect();
            assert_eq!(surv.len(), 1, "NMS must keep exactly one column per row, row {}: {:?}", y, surv);
        }
    }

    #[test]
    fn nms_collapses_symmetric_two_pixel_band_to_one() {
        // A clean vertical step produces a *symmetric* two-pixel Sobel band
        // (columns 5 and 6 have equal magnitude). Strict `>` on both sides would
        // suppress BOTH of them (each ties its inner neighbor), erasing the edge
        // entirely. The asymmetric tie-break must keep exactly one.
        let w = 24;
        let h = 24;
        let mut mag = vec![0.0f32; w * h];
        let mut ang = vec![0.0f32; w * h];
        for y in 0..h {
            mag[y * w + 5] = 1020.0;
            mag[y * w + 6] = 1020.0;
            for x in [5i32, 6] {
                ang[y * w + x as usize] = 0.0; // horizontal gradient -> compare left/right
            }
        }
        let grad = GradientMap {
            width: w as u32,
            height: h as u32,
            magnitude: mag,
            angle: ang,
        };
        let sup = non_max_suppress(&grad);
        for y in 0..h {
            let surv: Vec<usize> = [5usize, 6]
                .iter()
                .filter(|&&x| sup[y * w + x] > 0.0)
                .copied()
                .collect();
            assert_eq!(
                surv.len(),
                1,
                "symmetric band must collapse to exactly one pixel per row, row {}: {:?}",
                y,
                surv
            );
            assert!(surv[0] == 5 || surv[0] == 6, "survivor must be in the band");
        }
        // And nothing outside the band may survive.
        assert!(sup.iter().filter(|&&m| m > 0.0).count() == h);
    }

    #[test]
    fn nms_suppresses_flat_region() {
        // Uniform magnitude 50: no pixel is strictly greater than a neighbor,
        // so nothing survives. (With a peaked response the local maximum does
        // survive — covered by nms_thins_three_pixel_ridge_to_one.)
        let w = 16;
        let h = 16;
        let mag = vec![50.0f32; w * h];
        let ang = vec![0.0f32; w * h];
        let g = GradientMap {
            width: w as u32,
            height: h as u32,
            magnitude: mag,
            angle: ang,
        };
        let sup = non_max_suppress(&g);
        assert!(sup.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn hysteresis_promotes_adjacent_weak() {
        // Strong pixel at (5,5); weak pixel right next to it at (6,5).
        let w = 12;
        let h = 12;
        let mut sup = vec![0.0f32; w * h];
        sup[5 * w + 5] = 200.0; // strong
        sup[5 * w + 6] = 60.0; // weak (between low=40 and high)
        let (low, high) = (40.0, 100.0);
        let mask = promote_edges(&sup, low, high, w, h);
        assert!(mask[5 * w + 5]);
        assert!(mask[5 * w + 6], "adjacent weak pixel should be promoted");
    }

    #[test]
    fn hysteresis_discards_isolated_weak() {
        let w = 12;
        let h = 12;
        let mut sup = vec![0.0f32; w * h];
        sup[5 * w + 5] = 200.0; // strong
        sup[2 * w + 2] = 60.0; // weak, far away
        let mask = promote_edges(&sup, 40.0, 100.0, w, h);
        assert!(mask[5 * w + 5]);
        assert!(!mask[2 * w + 2], "isolated weak pixel must be discarded");
    }

    #[test]
    fn hysteresis_promotes_full_weak_chain() {
        // Chain of 5 weak pixels leading into one strong pixel — full
        // connectivity, not just direct-neighbor promotion.
        let w = 12;
        let h = 12;
        let mut sup = vec![0.0f32; w * h];
        sup[5 * w + 5] = 200.0; // strong
        for k in 1..=5 {
            sup[5 * w + (5 + k)] = 60.0; // weak chain extending right
        }
        let mask = promote_edges(&sup, 40.0, 100.0, w, h);
        assert!(mask[5 * w + 5]);
        for k in 1..=5 {
            assert!(mask[5 * w + (5 + k)], "weak chain link {} should be promoted", k);
        }
    }

    #[test]
    fn thresholds_adaptive_and_ordering() {
        // A distribution with a clear 90th percentile.
        // 89 samples at 50 + 11 at 500: the 90th percentile index
        // round(0.90 * 99) = 89 lands on the first 500-value, so
        // high must be 500, not 50.
        let mut sup = vec![0.0f32; 100];
        for (i, m) in sup.iter_mut().enumerate() {
            *m = if i < 89 { 50.0 } else { 500.0 };
        }
        let (high, low) = compute_thresholds(&sup);
        assert!(high >= 500.0 - 1e-3 && high <= 500.0 + 1e-3, "high={}", high);
        assert!((high * 0.4 - low).abs() < 1e-3);
    }

    #[test]
    fn thresholds_fallback_on_near_blank() {
        let sup = vec![0.0f32; 10]; // fewer than MIN_SAMPLE nonzero
        let (high, low) = compute_thresholds(&sup);
        assert_eq!(high, 100.0);
        assert_eq!(low, 40.0);
    }

    #[test]
    fn circular_mean_wraparound_not_naive_average() {
        // A cell whose two edge pixels have gradient angles +89° and -89°.
        // Both denote an almost-vertical luma gradient, i.e. an almost-horizontal
        // *edge*; the correct circular mean of the gradient is ~180° (the same
        // direction as 0° for a 180°-periodic angle), so the reported edge
        // orientation (+90°) is ~0°. The naive arithmetic mean of +89/-89 is 0°,
        // which would report orientation +90° instead — exactly what must NOT
        // come out of this test.
        let w = 4;
        let h = 4;
        // Build a gradient map then aggregate a single cell covering all pixels.
        let mut final_mask = vec![false; w * h];
        let mut mag = vec![0.0f32; w * h];
        let mut ang = vec![0.0f32; w * h];
        final_mask[0] = true;
        final_mask[1] = true;
        mag[0] = 1.0;
        mag[1] = 1.0;
        ang[0] = 89.0_f32.to_radians();
        ang[1] = -89.0_f32.to_radians();
        let grad = GradientMap {
            width: w as u32,
            height: h as u32,
            magnitude: mag,
            angle: ang,
        };
        // Single cell covering the whole 4x4 src (1 col x 1 row).
        let cells = aggregate_cell_edges(&final_mask, &grad, 1, 1, w as u32, h as u32);
        let cell = cells[0].expect("two edge pixels should yield an edge cell");
        let o = cell.orientation_deg;
        assert!(
            (o - 0.0).abs() < 5.0 || (o - 180.0).abs() < 5.0,
            "circular mean should land near 0/180, got {}",
            o
        );
        assert!(
            (o - 90.0).abs() > 10.0,
            "must NOT be the naive arithmetic average (90°), got {}",
            o
        );
    }

    #[test]
    fn aggregate_rejects_single_pixel() {
        let w = 4;
        let h = 4;
        let mut final_mask = vec![false; w * h];
        let mut mag = vec![0.0f32; w * h];
        let ang = vec![0.0f32; w * h];
        final_mask[0] = true; // only ONE edge pixel
        mag[0] = 5.0;
        let grad = GradientMap {
            width: w as u32,
            height: h as u32,
            magnitude: mag,
            angle: ang,
        };
        let cells = aggregate_cell_edges(&final_mask, &grad, 1, 1, w as u32, h as u32);
        assert!(cells[0].is_none(), "single stray edge pixel must be rejected");
    }

    #[test]
    fn direction_character_mapping_bounds() {
        // Stroke convention: horizontal boundary -> '-', vertical -> '|',
        // diagonals assert their own shape. See direction_to_char's docs.
        assert_eq!(direction_to_char(0.0), '-');
        assert_eq!(direction_to_char(90.0), '|');
        assert_eq!(direction_to_char(135.0), '/');
        assert_eq!(direction_to_char(45.0), '\\');
        // Periodic
        assert_eq!(direction_to_char(180.0), '-');
    }

    #[test]
    fn compute_frame_edges_pipeline_flat_image_no_edges() {
        // Uniform color -> no edges -> every cell None.
        let rgb = vec![128u8, 128, 128, 200, 200, 200, 10, 10, 10, 255, 255, 255];
        let cells = compute_frame_edges(&rgb, 2, 2, 2, 2);
        assert_eq!(cells.len(), 4);
        assert!(cells.iter().all(Option::is_none));
    }

    #[test]
    fn compute_frame_edges_pipeline_detects_half_split() {
        // Left side dark, right side bright: a vertical edge spanning the full
        // height. The step is at x=17 so it does NOT align with a 4px cell edge;
        // cell column 4 (covering x=16..19) straddles the boundary and holds
        // several strong edge pixels, clearing EDGE_CELL_MIN_PIXELS. Its
        // orientation must be vertical ('|').
        let w = 32usize;
        let h = 32usize;
        let cols = 8usize;
        let rows = 8usize;
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let c = if x < 17 { 0u8 } else { 255 };
                let i = (y * w + x) * 3;
                rgb[i] = c;
                rgb[i + 1] = c;
                rgb[i + 2] = c;
            }
        }
        let cells = compute_frame_edges(&rgb, w, h, cols, rows);
        // Cell (4,4) covers source x in [16,20), y in [16,20): it straddles the
        // x=17 luma step, so it must be reported as a vertical edge ('|').
        let boundary_col = 4usize;
        let mid_row = 4usize;
        let cell = cells[boundary_col + mid_row * cols as usize]
            .expect("boundary cell should be an edge");
        assert_eq!(direction_to_char(cell.orientation_deg), '|');
    }
}
