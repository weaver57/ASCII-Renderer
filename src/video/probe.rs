use std::path::Path;
use std::process::Command;

/// Probes a video file's native dimensions using ffmpeg's stderr output.
///
/// Runs `ffmpeg -i <path>` and parses the `Stream #0: Video: ... WxH ...`
/// line from the error output.  Falls back to `None` on failure (e.g.,
/// ffmpeg not installed, non-video file, parse error).
pub fn probe_video_dimensions<P: AsRef<Path>>(path: P) -> Option<(u32, u32)> {
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-i")
        .arg(path.as_ref())
        .output()
        .ok()?;

    // ffmpeg writes stream info to stderr
    let stderr = String::from_utf8_lossy(&output.stderr);

    for line in stderr.lines() {
        if !line.contains("Stream") || !line.contains("Video") {
            continue;
        }
        if let Some((w, h)) = parse_dimensions_from_stream_line(line) {
            return Some((w, h));
        }
    }
    None
}

/// Extracts the first `NNNxNNN` pattern from a stream info line.
fn parse_dimensions_from_stream_line(line: &str) -> Option<(u32, u32)> {
    for token in line.split(|c: char| c == ' ' || c == ',') {
        if let Some((w_str, h_str)) = token.split_once('x') {
            if let (Ok(w), Ok(h)) = (w_str.trim().parse::<u32>(), h_str.trim().parse::<u32>()) {
                if w > 0 && h > 0 {
                    return Some((w, h));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dimensions_from_stream_line() {
        let line = "    Stream #0:0: Video: h264 (High), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9]";
        assert_eq!(parse_dimensions_from_stream_line(line), Some((1920, 1080)));
    }

    #[test]
    fn test_parse_dimensions_different_order() {
        let line = "    Stream #0:0: Video: hevc, 1280x720, 30 fps";
        assert_eq!(parse_dimensions_from_stream_line(line), Some((1280, 720)));
    }

    #[test]
    fn test_parse_dimensions_no_match() {
        let line = "    Stream #0:0: Audio: aac (LC), 48000 Hz, stereo";
        assert_eq!(parse_dimensions_from_stream_line(line), None);
    }
}
