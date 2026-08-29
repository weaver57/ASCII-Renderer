use anyhow::{Context, Result};
use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};

pub struct FFmpegDecoder {
    child: Child,
    reader: BufReader<ChildStdout>,
    width: usize,
    height: usize,
    frame_bytes: usize,
}

impl FFmpegDecoder {
    pub fn new<P: AsRef<Path>>(
        input_path: P,
        width: usize,
        height: usize,
        fps: Option<f64>,
    ) -> Result<Self> {
        let path = input_path.as_ref();
        if !path.exists() {
            anyhow::bail!("Input file does not exist: {:?}", path);
        }
        if width == 0 || height == 0 {
            anyhow::bail!("Width and height must be greater than zero (got {}x{})", width, height);
        }

        let input_str = path.to_string_lossy().to_string();

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-nostdin")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-i")
            .arg(&input_str);

        if let Some(target_fps) = fps {
            if target_fps.is_finite() && target_fps > 0.0 {
                cmd.arg("-r").arg(format!("{:.2}", target_fps));
            }
        }

        cmd.arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-s")
            .arg(format!("{}x{}", width, height))
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn FFmpeg process for input: {}", input_str))?;

        let stdout = child
            .stdout
            .take()
            .context("Failed to open FFmpeg stdout pipe")?;

        let frame_bytes = width * height * 3;
        // Buffer at least 2 frames in reader to reduce syscalls
        let reader = BufReader::with_capacity(frame_bytes * 2, stdout);

        Ok(Self {
            child,
            reader,
            width,
            height,
            frame_bytes,
        })
    }

    #[allow(dead_code)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[allow(dead_code)]
    pub fn height(&self) -> usize {
        self.height
    }

    #[allow(dead_code)]
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// Reads the next raw RGB24 frame into `buffer`.
    /// Returns `Ok(true)` if a full frame was read, or `Ok(false)` on EOF.
    pub fn read_frame(&mut self, buffer: &mut [u8]) -> Result<bool> {
        assert_eq!(
            buffer.len(),
            self.frame_bytes,
            "Target buffer size must match frame_bytes"
        );

        match self.reader.read_exact(buffer) {
            Ok(()) => Ok(true),
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e).context("Error reading frame from FFmpeg stdout"),
        }
    }
}

impl Drop for FFmpegDecoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_nonexistent_file_errors() {
        let res = FFmpegDecoder::new(PathBuf::from("nonexistent_video_12345.mp4"), 64, 48, None);
        assert!(res.is_err());
    }

    #[test]
    fn test_zero_dimensions_error() {
        let path = PathBuf::from("Cargo.toml"); // exists
        let res = FFmpegDecoder::new(path, 0, 48, None);
        assert!(res.is_err());
    }
}
