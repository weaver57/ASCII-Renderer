//! Terminal color palettes: the fixed 256-color cube + grayscale ramp, and the
//! 16-color table, with "redmean" perceptually-weighted nearest-match (D6).
//!
//! The cube (indices 16–231) and grayscale ramp (232–255) are fixed by the
//! xterm 256-color standard. The 16 basic indices (0–15) are terminal-theme-
//! dependent in practice — the RGB values here are indicative xterm defaults
//! only, used solely for the true 16-color fallback mode where there is no
//! alternative.

use crate::render::grid::Rgb;

/// xterm's fixed 6×6×6 color cube levels (indices 16–231).
const CUBE_LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];

/// Nearest level index in `CUBE_LEVELS` for a single 8-bit channel.
#[inline]
fn nearest_cube_level(v: u8) -> usize {
    // CUBE_LEVELS is sorted ascending; find the closest by scanning (6 entries
    // — scanning is fine and avoids branchy binary-search overhead at 6 items).
    let mut best = 0usize;
    let mut best_dist = i32::MAX;
    for (i, &lv) in CUBE_LEVELS.iter().enumerate() {
        let d = (v as i32 - lv as i32).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

/// Finds the cube index (0–5 per channel) and the exact cube RGB for a color.
///
/// Nearest-level-per-channel, then the xterm formula
/// `index = 16 + 36*r_level + 6*g_level + b_level` (indices 16–231).
pub fn cube_index(color: Rgb) -> (usize, Rgb) {
    let rl = nearest_cube_level(color.r);
    let gl = nearest_cube_level(color.g);
    let bl = nearest_cube_level(color.b);
    let idx = 16 + 36 * rl + 6 * gl + bl;
    let exact = Rgb::new(CUBE_LEVELS[rl], CUBE_LEVELS[gl], CUBE_LEVELS[bl]);
    (idx, exact)
}

/// The 24-step grayscale ramp (indices 232–255); `gray(i) = 8 + 10*i`.
pub fn gray_entry(i: usize) -> (usize, Rgb) {
    let i = i.clamp(0, 23);
    let v = (8 + 10 * i) as u8;
    (232 + i, Rgb::new(v, v, v))
}

/// Nearest grayscale ramp entry for a target color's luma.
pub fn gray_index(color: Rgb) -> (usize, Rgb) {
    // Use Rec.709 luma as the mapping axis (matches the renderer's perception).
    let lum = ((color.r as u32 * 299 + color.g as u32 * 587 + color.b as u32 * 114) / 1000) as u8;
    let mut best = 0usize;
    let mut best_dist = i32::MAX;
    for i in 0..24 {
        let v = 8 + 10 * i;
        let d = (lum as i32 - v as i32).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    gray_entry(best)
}

/// "redmean" perceptually-weighted distance between two colors (D6).
///
/// Weights the R/G/B distance terms by how bright the involved reds are,
/// approximating human perception far better than flat Euclidean distance
/// without the cost of full CIELAB conversion.
#[inline]
pub fn redmean_distance(a: Rgb, b: Rgb) -> f32 {
    let rmean = (a.r as f32 + b.r as f32) / 2.0;
    let dr = a.r as f32 - b.r as f32;
    let dg = a.g as f32 - b.g as f32;
    let db = a.b as f32 - b.b as f32;
    ((2.0 + rmean / 256.0) * dr * dr
        + 4.0 * dg * dg
        + (2.0 + (255.0 - rmean) / 256.0) * db * db)
        .sqrt()
}

/// A fixed lookup table of palette entries, built once at startup.
#[derive(Debug, Clone)]
pub struct Palette {
    /// All 256 entries: 0–15 basic, 16–231 cube, 232–255 grayscale ramp.
    pub entries: Vec<Rgb>,
    /// Approximate average level step between adjacent cube levels, used by the
    /// dithering nudge (§4.8 of the plan). The cube's real spacing is
    /// non-uniform (95, 40, 40, 40, 40), so this is a documented approximation.
    pub approx_step: f32,
}

impl Palette {
    /// The full 256-color xterm palette.
    pub fn xterm256() -> Self {
        let mut entries = Vec::with_capacity(256);
        // 0–15: basic ANSI colors (indicative xterm defaults — see module docs).
        entries.extend([
            Rgb::new(0, 0, 0),       // 0  black
            Rgb::new(128, 0, 0),     // 1  red
            Rgb::new(0, 128, 0),     // 2  green
            Rgb::new(128, 128, 0),   // 3  yellow
            Rgb::new(0, 0, 128),     // 4  blue
            Rgb::new(128, 0, 128),   // 5  magenta
            Rgb::new(0, 128, 128),   // 6  cyan
            Rgb::new(192, 192, 192), // 7  white
            Rgb::new(128, 128, 128), // 8  bright black
            Rgb::new(255, 0, 0),     // 9  bright red
            Rgb::new(0, 255, 0),     // 10 bright green
            Rgb::new(255, 255, 0),   // 11 bright yellow
            Rgb::new(0, 0, 255),     // 12 bright blue
            Rgb::new(255, 0, 255),   // 13 bright magenta
            Rgb::new(0, 255, 255),   // 14 bright cyan
            Rgb::new(255, 255, 255), // 15 bright white
        ]);
        // 16–231: the 6×6×6 cube.
        for rl in 0..6 {
            for gl in 0..6 {
                for bl in 0..6 {
                    let (idx, rgb) = cube_index(Rgb::new(CUBE_LEVELS[rl], CUBE_LEVELS[gl], CUBE_LEVELS[bl]));
                    debug_assert_eq!(idx, entries.len());
                    entries.push(rgb);
                }
            }
        }
        // 232–255: the grayscale ramp.
        for i in 0..24 {
            let (idx, rgb) = gray_entry(i);
            debug_assert_eq!(idx, entries.len());
            entries.push(rgb);
        }
        debug_assert_eq!(entries.len(), 256);
        Self {
            entries,
            approx_step: 51.0,
        }
    }

    /// A fixed 16-entry basic palette, used only for the true 16-color
    /// fallback mode.
    pub fn basic16() -> Self {
        Self {
            entries: Palette::xterm256().entries[0..16].to_vec(),
            approx_step: 25.0,
        }
    }

    /// Nearest palette index for `color` using redmean distance.
    pub fn nearest_index(&self, color: Rgb) -> usize {
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for (i, &entry) in self.entries.iter().enumerate() {
            let d = redmean_distance(color, entry);
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        best
    }

    /// Returns the palette entry at `idx`.
    #[inline]
    pub fn entry(&self, idx: usize) -> Rgb {
        self.entries[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_black_is_index_16() {
        let (idx, rgb) = cube_index(Rgb::new(0, 0, 0));
        assert_eq!(idx, 16);
        assert_eq!(rgb, Rgb::new(0, 0, 0));
    }

    #[test]
    fn cube_white_is_index_231() {
        let (idx, rgb) = cube_index(Rgb::new(255, 255, 255));
        assert_eq!(idx, 231);
        assert_eq!(rgb, Rgb::new(255, 255, 255));
    }

    #[test]
    fn cube_levels_nearest() {
        // 50 → nearest cube level is 0 (dist 50) vs 95 (dist 45) → 95? No:
        // 95-50=45 < 50, so level 1 (95). Check via cube_index.
        let (idx, rgb) = cube_index(Rgb::new(50, 50, 50));
        // level index 1 → 95
        assert_eq!(rgb, Rgb::new(95, 95, 95));
        // 16 + 36*1 + 6*1 + 1 = 59
        assert_eq!(idx, 59);
    }

    #[test]
    fn gray_ramp_bounds() {
        let (idx0, g0) = gray_entry(0);
        assert_eq!((idx0, g0), (232, Rgb::new(8, 8, 8)));
        let (idx23, g23) = gray_entry(23);
        assert_eq!((idx23, g23), (255, Rgb::new(238, 238, 238)));
    }

    #[test]
    fn gray_index_picks_nearest_luma() {
        // Luma ~255 → index 23 (238)
        let (idx, _) = gray_index(Rgb::new(255, 255, 255));
        assert_eq!(idx, 255);
        // Luma ~0 → index 0 (8)
        let (idx, _) = gray_index(Rgb::new(0, 0, 0));
        assert_eq!(idx, 232);
    }

    #[test]
    fn palette_256_builds_correct_indices() {
        let p = Palette::xterm256();
        assert_eq!(p.entries.len(), 256);
        assert_eq!(p.entry(16), Rgb::new(0, 0, 0));
        assert_eq!(p.entry(231), Rgb::new(255, 255, 255));
        assert_eq!(p.entry(232), Rgb::new(8, 8, 8));
        assert_eq!(p.entry(255), Rgb::new(238, 238, 238));
    }

    #[test]
    fn redmean_symmetric() {
        let a = Rgb::new(10, 20, 30);
        let b = Rgb::new(200, 180, 160);
        assert!((redmean_distance(a, b) - redmean_distance(b, a)).abs() < 1e-4);
        assert_eq!(redmean_distance(a, a), 0.0);
    }

    #[test]
    fn nearest_matches_known_entry() {
        let p = Palette::xterm256();
        // Pure primaries exactly match the *basic* brighter entries (9–14),
        // not the cube: the basic set's exact RGB match wins under redmean
        // because its distance is 0.
        assert_eq!(p.nearest_index(Rgb::new(255, 0, 0)), 9);    // bright red
        assert_eq!(p.nearest_index(Rgb::new(0, 255, 0)), 10);   // bright green
        assert_eq!(p.nearest_index(Rgb::new(0, 0, 255)), 12);   // bright blue
        // Mid gray 128 exactly matches basic index 8 (not the gray ramp).
        assert_eq!(p.nearest_index(Rgb::new(128, 128, 128)), 8);
        // Cube-only colors (absent from the basic set) resolve to their exact
        // cube entry: 95 = CUBE_LEVELS[1] → 16 + 36*1 + 6*1 + 1 = 59.
        assert_eq!(p.nearest_index(Rgb::new(95, 95, 95)), 59);
        // 175 = CUBE_LEVELS[3] on red → 16 + 36*3 = 124.
        assert_eq!(p.nearest_index(Rgb::new(175, 0, 0)), 124);
    }

    #[test]
    fn basic16_has_16_entries() {
        let p = Palette::basic16();
        assert_eq!(p.entries.len(), 16);
    }
}
