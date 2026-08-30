use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Console::{
    GetCurrentConsoleFontEx, GetStdHandle, CONSOLE_FONT_INFOEX, STD_OUTPUT_HANDLE,
};

use crate::config;

/// Result of a terminal cell dimension measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellDimensions {
    /// Width of one character cell in pixels.
    pub width_px: u16,
    /// Height of one character cell in pixels.
    pub height_px: u16,
    /// Aspect ratio: width / height. For typical monospace fonts ~0.5 (cells are ~2x taller than wide).
    pub aspect_ratio: f32,
}

// ── Shared CLI override state ──────────────────────────────────────────────
// CLI `--char-aspect` override is set once at startup and never changes.
// One shared static: `set_char_aspect_override` writes and `get_char_aspect`
// reads the same slot. `OnceLock`'s set/get already provides the
// happens-before, so no extra synchronization is needed.
static CLI_OVERRIDE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();

/// Attempts to measure the current terminal's character cell dimensions
/// using the Windows Console API (`GetCurrentConsoleFontEx`).
///
/// Returns `None` if:
/// - Not running on Windows
/// - The API call fails (e.g., redirected output, no console attached)
/// - The returned font size is invalid (zero)
///
/// On many Windows terminals (especially Windows Terminal, VS Code
/// integrated terminal, etc.) `dwFontSize` returns unreliable values —
/// X/Y may be swapped, zeroed, or reported in unexpected units.  The
/// caller should sanity-check the result before trusting it.
pub fn measure_terminal_cell_dimensions() -> Option<CellDimensions> {
    // SAFETY: Windows API calls with valid handles and pointers
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

        let width = font_info.dwFontSize.X;
        let height = font_info.dwFontSize.Y;

        if width == 0 || height == 0 {
            return None;
        }

        let raw_aspect = width as f32 / height as f32;

        Some(CellDimensions {
            width_px: width as u16,
            height_px: height as u16,
            aspect_ratio: raw_aspect,
        })
    }
}

/// Gets the full cell dimensions (width, height, aspect) with fallback.
///
/// Useful when you need the absolute pixel dimensions, not just the ratio.
pub fn get_cell_dimensions() -> CellDimensions {
    measure_terminal_cell_dimensions().unwrap_or(CellDimensions {
        width_px: 8,
        height_px: 16,
        aspect_ratio: 0.5,
    })
}

/// Sets a manual CLI override for the character aspect ratio.
///
/// This is used by the CLI `--char-aspect` flag to bypass the automatic
/// measurement and force a specific aspect ratio. Returns true if the
/// override was applied (value was valid).
///
/// Valid range: [0.1, 1.0] — covers all plausible monospace fonts.
pub fn set_char_aspect_override(aspect: f32) -> bool {
    if aspect > 0.0 && aspect.is_finite() && aspect >= 0.1 && aspect <= 1.0 {
        let _ = CLI_OVERRIDE.set(aspect);
        true
    } else {
        false
    }
}

/// Returns the terminal's character cell aspect ratio (width/height).
///
/// **Priority order:**
/// 1. CLI `--char-aspect` override (set via `set_char_aspect_override`)
/// 2. Config file `~/.ascii_renderer.toml` (saved via `--calibrate`)
/// 3. Windows Console API measurement with sanity gate (measured at runtime)
/// 4. Heuristic fallback from terminal dimensions
/// 5. Hardcoded default of 0.5
///
/// **Sanity gate (step 3):** The Windows Console API (`GetCurrentConsoleFontEx`)
/// frequently returns wrong `dwFontSize` values — especially on Windows
/// Terminal, VS Code integrated terminal, and other modern terminals.
/// Common failure modes:
///   - X and Y are swapped → aspect > 1.0 (cells wider than tall — never
///     true for monospace fonts)
///   - X or Y is in unexpected units (e.g., font height instead of cell
///     pixel width) → aspect far outside the normal [0.3, 0.7] range
///
/// Any measured value outside [0.3, 0.7] is treated as unreliable and
/// the heuristic/default is used instead.  A real override via
/// `set_char_aspect_override` (the `--char-aspect` CLI flag) or config file
/// bypasses this sanity gate.
///
/// This function is cheap to call repeatedly; the Windows API call is fast
/// and results are cached.
pub fn get_char_aspect() -> f32 {
    // 1. CLI override takes highest priority
    if let Some(&override_val) = CLI_OVERRIDE.get() {
        return override_val;
    }

    // 2. Config file (loaded once, cached via OnceLock)
    static CONFIG_ASPECT: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    let config_val = CONFIG_ASPECT.get_or_init(|| config::Config::load().char_aspect);
    if let Some(val) = config_val {
        return *val;
    }

    // 3. Windows Console API measurement with sanity gate
    static MEASURED_ASPECT: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    let measured = MEASURED_ASPECT.get_or_init(|| {
        let measured = measure_terminal_cell_dimensions();
        match measured {
            Some(d) if d.aspect_ratio >= 0.3 && d.aspect_ratio <= 0.7 => {
                // Looks sane for a monospace terminal font.
                Some(d.aspect_ratio)
            }
            Some(d) => {
                // Measured but out of sane range — API is lying.
                eprintln!(
                    "[ascii_renderer] warning: measured char aspect = {:.3} \
                     ({}\u{00D7}{} px) is outside sane range [0.3, 0.7]. \
                     Falling back to heuristic/default.",
                    d.aspect_ratio, d.width_px, d.height_px
                );
                None
            }
            None => {
                // Could not measure at all (non-Windows, no console, etc.)
                None
            }
        }
    });
    measured.unwrap_or_else(|| {
        // 4. Heuristic fallback from terminal dimensions
        let heuristic = estimate_from_terminal(80, 25); // fallback default terminal size
        if heuristic >= 0.3 && heuristic <= 0.7 {
            heuristic
        } else {
            // 5. Hardcoded default
            0.5
        }
    })
}

/// Heuristic fallback: estimate char aspect from terminal size when
/// GetCurrentConsoleFontEx is unavailable or returns unreliable values.
///
/// Model: assume the terminal fills a 16:9 display. If the terminal is
/// `cols` wide and `rows` tall, the character cell that makes exactly
/// `cols`×`rows` fill that screen has aspect `(rows/cols)·(16/9)`. For a
/// common 80×25 terminal that is `(25/80)·16/9 ≈ 0.556`, i.e. cells ~1.8×
/// taller than wide — close to the ~0.5 of typical monospace fonts.
///
/// This is purely a best-effort guess and should not replace the real API.
pub fn estimate_aspect_from_terminal_size(term_cols: u16, term_rows: u16) -> f32 {
    if term_cols == 0 || term_rows == 0 {
        return 0.5;
    }
    let guess = (term_rows as f32 / term_cols as f32) * (16.0 / 9.0);
    guess.clamp(0.3, 0.7)
}

/// Public wrapper so main.rs can call the heuristic for debug output.
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
    fn test_cell_dimensions_struct() {
        let dims = CellDimensions {
            width_px: 10,
            height_px: 20,
            aspect_ratio: 0.5,
        };
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
        assert!(set_char_aspect_override(0.5));
        assert!(set_char_aspect_override(0.45));
    }

    #[test]
    fn test_estimate_from_terminal_size_1080p() {
        let est = estimate_aspect_from_terminal_size(130, 34);
        assert!((est - 0.467).abs() < 0.01);
    }

    #[test]
    fn test_estimate_from_terminal_size_80col() {
        let est = estimate_aspect_from_terminal_size(80, 25);
        assert!((est - 0.556).abs() < 0.01);
    }
}
