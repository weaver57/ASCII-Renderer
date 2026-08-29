use super::luminance::rgb_to_luminance;
use std::io::Write;

pub const SHORT_RAMP: &str = " .:-=+*#%@";
pub const DETAILED_RAMP: &str = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
pub const BLOCK_RAMP: &str = " ░▒▓█";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    TrueColor,
    Grayscale,
    Monochrome,
}

pub struct AsciiRenderer {
    ramp: Vec<char>,
    color_mode: ColorMode,
}

impl AsciiRenderer {
    pub fn new(ramp_str: &str, color_mode: ColorMode, invert: bool) -> Self {
        let mut chars: Vec<char> = ramp_str.chars().collect();
        if chars.is_empty() {
            chars = vec![' '];
        }
        if invert {
            chars.reverse();
        }
        Self {
            ramp: chars,
            color_mode,
        }
    }

    /// Maps a luminance value [0, 255] to a character in the current ramp.
    #[inline(always)]
    pub fn luminance_to_char(&self, lum: u8) -> char {
        let max_idx = self.ramp.len() - 1;
        let idx = (lum as usize * max_idx) / 255;
        self.ramp[idx]
    }

    /// Renders a raw RGB24 frame (`width * height * 3` bytes) into a formatted ANSI terminal byte buffer.
    ///
    /// Reuses the `output` buffer to prevent heap allocations every frame.
    pub fn render_frame(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        output: &mut Vec<u8>,
    ) {
        output.clear();

        if width == 0 || height == 0 || rgb_data.len() < width * height * 3 {
            return;
        }

        // Move cursor to top-left (0, 0)
        output.extend_from_slice(b"\x1b[H");

        let mut last_color: Option<(u8, u8, u8)> = None;
        let mut char_buf = [0u8; 4];

        for y in 0..height {
            let row_offset = y * width * 3;
            for x in 0..width {
                let pixel_offset = row_offset + x * 3;
                let r = rgb_data[pixel_offset];
                let g = rgb_data[pixel_offset + 1];
                let b = rgb_data[pixel_offset + 2];

                let lum = rgb_to_luminance(r, g, b);
                let ch = self.luminance_to_char(lum);

                match self.color_mode {
                    ColorMode::TrueColor => {
                        let color = (r, g, b);
                        if last_color != Some(color) {
                            write!(output, "\x1b[38;2;{};{};{}m", r, g, b).unwrap();
                            last_color = Some(color);
                        }
                    }
                    ColorMode::Grayscale => {
                        let color = (lum, lum, lum);
                        if last_color != Some(color) {
                            write!(output, "\x1b[38;2;{};{};{}m", lum, lum, lum).unwrap();
                            last_color = Some(color);
                        }
                    }
                    ColorMode::Monochrome => {}
                }

                let encoded = ch.encode_utf8(&mut char_buf);
                output.extend_from_slice(encoded.as_bytes());
            }
            if y + 1 < height {
                output.extend_from_slice(b"\r\n");
            }
        }

        // Reset color attributes at the end of the frame
        if self.color_mode != ColorMode::Monochrome {
            output.extend_from_slice(b"\x1b[0m");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_mapping_bounds() {
        let renderer = AsciiRenderer::new(" @", ColorMode::Monochrome, false);
        assert_eq!(renderer.luminance_to_char(0), ' ');
        assert_eq!(renderer.luminance_to_char(255), '@');
    }

    #[test]
    fn test_inverted_ramp() {
        let renderer = AsciiRenderer::new(" @", ColorMode::Monochrome, true);
        assert_eq!(renderer.luminance_to_char(0), '@');
        assert_eq!(renderer.luminance_to_char(255), ' ');
    }

    #[test]
    fn test_empty_ramp_safe() {
        let renderer = AsciiRenderer::new("", ColorMode::Monochrome, false);
        assert_eq!(renderer.luminance_to_char(0), ' ');
        assert_eq!(renderer.luminance_to_char(255), ' ');
    }

    #[test]
    fn test_unicode_multibyte_ramp() {
        let renderer = AsciiRenderer::new(BLOCK_RAMP, ColorMode::Monochrome, false);
        assert_eq!(renderer.luminance_to_char(0), ' ');
        assert_eq!(renderer.luminance_to_char(255), '█');

        let mut output = Vec::new();
        let rgb_data = vec![255, 255, 255, 0, 0, 0];
        renderer.render_frame(&rgb_data, 2, 1, &mut output);

        let output_str = String::from_utf8(output).expect("Output must be valid UTF-8");
        assert!(output_str.contains('█'));
        assert!(output_str.contains(' '));
    }

    #[test]
    fn test_stress_ramp_all_luminance_values() {
        let ramps = [SHORT_RAMP, DETAILED_RAMP, BLOCK_RAMP, "ABC", "⚡🔥🚀✨", "X"];
        for ramp in ramps {
            let renderer_normal = AsciiRenderer::new(ramp, ColorMode::Monochrome, false);
            let renderer_inverted = AsciiRenderer::new(ramp, ColorMode::Monochrome, true);
            for lum in 0..=255u8 {
                let ch_norm = renderer_normal.luminance_to_char(lum);
                let ch_inv = renderer_inverted.luminance_to_char(lum);
                assert!(ramp.contains(ch_norm));
                assert!(ramp.contains(ch_inv));
            }
        }
    }
}
