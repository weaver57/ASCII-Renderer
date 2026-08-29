/// Fast integer-based perceived luminance (BT.709 coefficients).
/// Returns a luminance value in the range [0, 255].
#[inline(always)]
pub fn rgb_to_luminance(r: u8, g: u8, b: u8) -> u8 {
    // 54/256 ≈ 0.2109, 183/256 ≈ 0.7148, 19/256 ≈ 0.0742
    let lum = (54u32 * r as u32 + 183u32 * g as u32 + 19u32 * b as u32) >> 8;
    lum as u8
}

/// Precise ITU-R BT.709 perceived luminance: 0.2126*R + 0.7152*G + 0.0722*B
#[inline(always)]
#[allow(dead_code)]
pub fn rgb_to_luminance_f32(r: u8, g: u8, b: u8) -> f32 {
    0.2126 * (r as f32) + 0.7152 * (g as f32) + 0.0722 * (b as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminance_extremes() {
        assert_eq!(rgb_to_luminance(0, 0, 0), 0);
        assert_eq!(rgb_to_luminance(255, 255, 255), 255);
    }

    #[test]
    fn test_green_dominance() {
        let green_lum = rgb_to_luminance(0, 255, 0);
        let red_lum = rgb_to_luminance(255, 0, 0);
        let blue_lum = rgb_to_luminance(0, 0, 255);
        assert!(green_lum > red_lum);
        assert!(red_lum > blue_lum);
    }
}
