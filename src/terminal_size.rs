use std::sync::OnceLock;

/// Hardcoded default character aspect ratio (cell width / cell height).
///
/// Most monospace terminal fonts are approximately 0.5 (cells 2× taller than
/// wide), but the exact value varies per font and terminal.  Users can
/// override with `--char-aspect`.
const DEFAULT_CHAR_ASPECT: f32 = 0.6;

static OVERRIDE: OnceLock<f32> = OnceLock::new();

/// Sets a manual override for the character aspect ratio.
pub fn set_char_aspect_override(aspect: f32) -> bool {
    if aspect > 0.0 && aspect.is_finite() && aspect >= 0.1 && aspect <= 1.0 {
        let _ = OVERRIDE.set(aspect);
        true
    } else {
        false
    }
}

/// Returns the terminal's character cell aspect ratio (width/height).
///
/// Returns the CLI override if `--char-aspect` was provided, otherwise
/// the hardcoded default.  No runtime measurement — the Windows Console
/// API (`GetCurrentConsoleFontEx`) is notoriously unreliable across
/// terminals and is deliberately not used.
pub fn get_char_aspect() -> f32 {
    if let Some(&override_val) = OVERRIDE.get() {
        return override_val;
    }
    DEFAULT_CHAR_ASPECT
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
