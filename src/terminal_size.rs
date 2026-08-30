use std::sync::atomic::{AtomicBool, Ordering};

use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Console::{
    GetCurrentConsoleFontEx, GetStdHandle, CONSOLE_FONT_INFOEX, STD_OUTPUT_HANDLE,
};

/// Result of a terminal cell dimension measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellDimensions {
    pub width_px: u16,
    pub height_px: u16,
    pub aspect_ratio: f32,
}

static OVERRIDE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
static OVERRIDE_SET: AtomicBool = AtomicBool::new(false);

pub fn measure_terminal_cell_dimensions() -> Option<CellDimensions> {
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        if handle == INVALID_HANDLE_VALUE || handle.is_null() {
            return None;
        }
        let mut font_info: CONSOLE_FONT_INFOEX = std::mem::zeroed();
        font_info.cbSize = std::mem::size_of::<CONSOLE_FONT_INFOEX>() as u32;
        let result = GetCurrentConsoleFontEx(handle, 1, &mut font_info);
        if result == 0 {
            return None;
        }
        let raw_x = font_info.dwFontSize.X;
        let raw_y = font_info.dwFontSize.Y;
        if raw_x == 0 || raw_y == 0 {
            return None;
        }
        Some(CellDimensions {
            width_px: raw_x as u16,
            height_px: raw_y as u16,
            aspect_ratio: raw_x as f32 / raw_y as f32,
        })
    }
}

pub fn get_cell_dimensions() -> CellDimensions {
    measure_terminal_cell_dimensions().unwrap_or(CellDimensions {
        width_px: 8,
        height_px: 16,
        aspect_ratio: 0.5,
    })
}

pub fn set_char_aspect_override(aspect: f32) -> bool {
    if aspect > 0.0 && aspect.is_finite() && aspect >= 0.1 && aspect <= 1.0 {
        let _ = OVERRIDE.set(aspect);
        OVERRIDE_SET.store(true, Ordering::Release);
        true
    } else {
        false
    }
}

/// Estimates char_aspect from terminal size assuming a standard 16:9 monitor.
fn estimate_aspect_from_terminal_size(term_cols: u16, term_rows: u16) -> f32 {
    const SCREEN_ASPECT: f32 = 16.0 / 9.0;
    let estimated = SCREEN_ASPECT * term_rows as f32 / term_cols as f32;
    estimated.clamp(0.2, 0.8)
}

/// Returns the terminal's character cell aspect ratio (width/height).
pub fn get_char_aspect() -> f32 {
    if let Some(&override_val) = OVERRIDE.get() {
        return override_val;
    }

    static CACHED_ASPECT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();

    *CACHED_ASPECT.get_or_init(|| {
        let measured = measure_terminal_cell_dimensions();
        match measured {
            Some(d) if d.aspect_ratio >= 0.35 && d.aspect_ratio <= 0.65 => {
                d.aspect_ratio
            }
            Some(d) if d.aspect_ratio > 1.0 => {
                let inverted = d.height_px as f32 / d.width_px as f32;
                if inverted >= 0.35 && inverted <= 0.65 {
                    eprintln!(
                        "[ascii_renderer] note: measured aspect {:.3} ({}x{} px) \
                         looks inverted; using {:.3} instead.\n\
                         Use --char-aspect to override if output looks distorted.",
                        d.aspect_ratio, d.width_px, d.height_px, inverted
                    );
                    inverted
                } else {
                    eprintln!(
                        "[ascii_renderer] note: Windows Console API returned \
                         unreliable font metrics (aspect {:.3}, inverted {:.3}).\n\
                         Falling back to 0.5.  Use --char-aspect to override.",
                        d.aspect_ratio, inverted
                    );
                    0.5
                }
            }
            Some(d) => {
                eprintln!(
                    "[ascii_renderer] note: measured char aspect = {:.3} \
                     ({}x{} px). If output looks distorted, use --char-aspect to override.",
                    d.aspect_ratio, d.width_px, d.height_px
                );
                d.aspect_ratio
            }
            None => {
                0.5
            }
        }
    })
}

pub fn estimate_from_terminal(term_cols: u16, term_rows: u16) -> f32 {
    estimate_aspect_from_terminal_size(term_cols, term_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_dimensions() {
        let dims = get_cell_dimensions();
        assert!(dims.width_px > 0);
        assert!(dims.height_px > 0);
        assert!(dims.aspect_ratio > 0.0);
        assert_eq!(dims.aspect_ratio, 0.5);
    }

    #[test]
    fn test_aspect_override_invalid() {
        assert!(!set_char_aspect_override(0.0));
        assert!(!set_char_aspect_override(-1.0));
        assert!(!set_char_aspect_override(f32::NAN));
        assert!(!set_char_aspect_override(f32::INFINITY));
        assert!(!set_char_aspect_override(1.5));
    }

    #[test]
    fn test_aspect_override_valid() {
        assert!(set_char_aspect_override(0.6));
        assert!(set_char_aspect_override(0.45));
    }

    #[test]
    fn test_estimate_from_terminal_size_1080p() {
        let est = estimate_from_terminal_size(130, 34);
        assert!((est - 0.467).abs() < 0.01);
    }

    #[test]
    fn test_estimate_from_terminal_size_80col() {
        let est = estimate_from_terminal_size(80, 25);
        assert!((est - 0.556).abs() < 0.01);
    }
}
