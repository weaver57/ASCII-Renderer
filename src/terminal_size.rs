use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::core::BOOL;
use windows_sys::Win32::System::Console::{
    GetCurrentConsoleFontEx, GetStdHandle, CONSOLE_FONT_INFOEX, STD_OUTPUT_HANDLE,
};

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

/// Attempts to measure the current terminal's character cell dimensions in pixels
/// using the Windows Console API (`GetCurrentConsoleFontEx`).
///
/// Returns `None` if:
/// - Not running on Windows
/// - The API call fails (e.g., redirected output, no console attached)
/// - The returned font size is invalid (zero)
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

        Some(CellDimensions {
            width_px: width as u16,
            height_px: height as u16,
            aspect_ratio: width as f32 / height as f32,
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

/// Sets a manual override for the character aspect ratio.
///
/// This is used by the CLI `--char-aspect` flag to bypass the automatic
/// measurement and force a specific aspect ratio. Returns true if the
/// override was applied (value was valid).
pub fn set_char_aspect_override(aspect: f32) -> bool {
    if aspect > 0.0 && aspect.is_finite() {
        static OVERRIDE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        // OnceLock can only be set once; if already set, ignore subsequent calls
        // (the first call wins, typically from CLI parsing before rendering)
        let _ = OVERRIDE.set(aspect);
        true
    } else {
        false
    }
}

/// Returns the terminal's character cell aspect ratio (width/height).
///
/// On Windows, attempts to query the actual console font metrics.
/// Falls back to 0.5 (standard monospace assumption: cells ~2x taller than wide)
/// when measurement is unavailable or fails.
///
/// If a CLI override was provided via `set_char_aspect_override`, that value
/// takes precedence over the measured or fallback value.
///
/// This function is cheap to call repeatedly; the Windows API call is fast.
pub fn get_char_aspect() -> f32 {
    // Check for CLI override first
    static OVERRIDE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    if let Some(&override_val) = OVERRIDE.get() {
        return override_val;
    }

    // Cache the measured/fallback result to avoid repeated syscalls
    static CACHED_ASPECT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();

    *CACHED_ASPECT.get_or_init(|| {
        measure_terminal_cell_dimensions()
            .map(|d| d.aspect_ratio)
            .unwrap_or(0.5)
    })
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
        // Default fallback should be 8x16 = 0.5
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
        // Invalid values (0, negative, NaN) should be rejected
        assert!(!set_char_aspect_override(0.0));
        assert!(!set_char_aspect_override(-1.0));
        assert!(!set_char_aspect_override(f32::NAN));
        assert!(!set_char_aspect_override(f32::INFINITY));
    }

    #[test]
    fn test_aspect_override_valid() {
        // Valid values should be accepted (note: first set wins in OnceLock)
        assert!(set_char_aspect_override(0.6));
        assert!(set_char_aspect_override(0.55)); // Will be ignored, first wins
    }
}