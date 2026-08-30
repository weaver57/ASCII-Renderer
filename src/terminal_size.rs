use std::sync::OnceLock;

use crate::config;

/// Hardcoded default character aspect ratio (cell width / cell height).
///
/// 0.5 is the industry standard — most monospace fonts are approximately
/// 2× taller than wide.  Users can override with `--char-aspect` or
/// by saving a calibrated value to `~/.ascii_renderer.toml`.
const DEFAULT_CHAR_ASPECT: f32 = 0.5;

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
/// Priority order:
/// 1. CLI `--char-aspect` override (if provided)
/// 2. Config file `~/.ascii_renderer.toml` (if saved via `--calibrate`)
/// 3. Hardcoded default of 0.5
pub fn get_char_aspect() -> f32 {
    // 1. CLI override takes highest priority
    if let Some(&override_val) = OVERRIDE.get() {
        return override_val;
    }

    // 2. Config file (loaded once, cached via OnceLock)
    static CONFIG_ASPECT: OnceLock<Option<f32>> = OnceLock::new();
    let config_val = CONFIG_ASPECT.get_or_init(|| config::Config::load().char_aspect);
    if let Some(val) = config_val {
        return *val;
    }

    // 3. Hardcoded default
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
