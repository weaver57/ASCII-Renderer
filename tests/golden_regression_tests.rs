//! Golden-output regression suite for Phase 5.
//!
//! Drives both the single-threaded Phase 1–4 scalar pipeline and the Phase 5
//! threaded+SIMD pipeline through identical synthetic inputs and asserts that
//! the final ANSI output frames are byte-identical. This is the ultimate
//! zero-divergence guarantee (§5, "Golden output regression").

use ascii_renderer::pipeline::{run_pipeline_with_decoder_and_writer, PipelineConfig};
use ascii_renderer::render::{
    ascii::{AsciiRenderer, ColorMode},
    edge::compute_frame_edges_from_luma,
    grid::build_char_grid_into,
    temporal::TemporalEdgeSmoother,
};
use ascii_renderer::video::decoder::{DecodedFrame, VideoDecoder};
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Synthetic frame decoder that yields the exact same deterministic frames as
/// the single-threaded test path below.
struct MockDecoder {
    total_frames: usize,
    current_frame: usize,
    width: usize,
    height: usize,
    fps: f64,
}

impl MockDecoder {
    fn new(total_frames: usize, width: usize, height: usize, fps: f64) -> Self {
        Self {
            total_frames,
            current_frame: 0,
            width,
            height,
            fps,
        }
    }
}

impl VideoDecoder for MockDecoder {
    fn read_frame(&mut self) -> anyhow::Result<Option<DecodedFrame>> {
        if self.current_frame >= self.total_frames {
            return Ok(None);
        }

        let y_size = self.width * self.height;
        let uv_size = (self.width / 2) * (self.height / 2);
        let total_size = y_size + 2 * uv_size;
        let mut data = vec![128u8; total_size];

        // Deterministic pattern: moving diagonal stripes
        let shift = (self.current_frame % self.width) as u8;
        for y in 0..self.height {
            for x in 0..self.width {
                let val = ((x as u8).wrapping_add(shift) ^ (y as u8)).min(235).max(16);
                data[y * self.width + x] = val;
            }
        }

        // Neutral chroma
        for i in y_size..total_size {
            data[i] = 128;
        }

        let pts = self.current_frame as f64 / self.fps;
        self.current_frame += 1;

        Ok(Some(DecodedFrame {
            data,
            pts_seconds: Some(pts),
        }))
    }

    fn avg_fps(&self) -> f64 {
        self.fps
    }
}

/// In-memory writer that records bytes for comparison.
#[derive(Clone, Default)]
struct MockWriter {
    buffer: Arc<Mutex<Vec<u8>>>,
}

impl MockWriter {
    fn bytes(&self) -> Vec<u8> {
        self.buffer.lock().unwrap().clone()
    }
}

impl Write for MockWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Runs the single-threaded Phase 1–4 scalar pipeline on the same frame
/// sequence as the decoder would produce, and returns the concatenated ANSI
/// output. This mirrors `main.rs`'s `compute_frame_edges_from_luma` path.
///
/// Both the scalar and threaded pipelines operate at grid resolution: the
/// FFmpeg decoder is configured with `-s cols x rows` (§4.2), so it already
/// scales every frame to the output grid before it enters the pipeline. Edge
/// detection and YUV downsampling are therefore both 1:1 with respect to the
/// grid. The mock decoder reproduces that grid-resolution frame here.
fn run_scalar_pipeline(
    frame_count: usize,
    cols: usize,
    rows: usize,
    _fps: f64,
    color_mode: ColorMode,
    ramp_str: &str,
    invert: bool,
) -> Vec<u8> {
    let width = cols;
    let height = rows;
    // Match the pipeline's decode thread: color space is inferred from the
    // grid height (`detect_color_space(height, None)`), not hardcoded.
    let color_space = ascii_renderer::video::yuv::detect_color_space(rows as u32, None);
    let mut out = Vec::new();
    let mut smoother = TemporalEdgeSmoother::new(0.35);
    let renderer = AsciiRenderer::new(ramp_str, color_mode, invert);
    let mut double_grid = ascii_renderer::diff::DoubleGrid::new(cols, rows);
    let color_support = ascii_renderer::caps::detect_color_support();
    let palette = match color_support {
        ascii_renderer::caps::ColorSupport::Palette256 => ascii_renderer::palette::Palette::xterm256(),
        ascii_renderer::caps::ColorSupport::Basic16 => ascii_renderer::palette::Palette::basic16(),
        _ => ascii_renderer::palette::Palette::xterm256(),
    };
    let mut output_state = ascii_renderer::terminal::OutputState::new(color_support, false);
    let mono = color_mode == ColorMode::Monochrome;

    // Pre-allocate scratch buffers
    let mut cell_luma = vec![0.0f32; cols * rows];
    let mut cell_color = vec![(0u8, 0u8, 0u8); cols * rows];

    for frame_idx in 0..frame_count {
        // Build the exact same grid-resolution luma frame as the decoder
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let mut luma_map = vec![0.0f32; y_size];

        let shift = (frame_idx % width) as u8;
        for y in 0..height {
            for x in 0..width {
                let val = ((x as u8).wrapping_add(shift) ^ (y as u8)).min(235).max(16);
                luma_map[y * width + x] = val as f32;
            }
        }

        // Scalar edge pipeline (1:1 to grid)
        let cell_edges = compute_frame_edges_from_luma(&luma_map, width, height, cols, rows);
        let smoothed = smoother.update(cell_edges);

        // Downsample YUV planes (1:1 to grid; luma only matters for monochrome,
        // we need full RGB for color)
        let y_plane: Vec<u8> = luma_map.iter().map(|&v| v as u8).collect();
        let u_plane = vec![128u8; uv_size];
        let v_plane = vec![128u8; uv_size];

        ascii_renderer::video::yuv::downsample_yuv_planes(
            &y_plane,
            &u_plane,
            &v_plane,
            width,
            height,
            color_space,
            cols,
            rows,
            &mut cell_luma,
            &mut cell_color,
        );

        // Build char grid and diff
        build_char_grid_into(
            double_grid.write_buffer(),
            &cell_luma,
            &cell_color,
            &smoothed,
            &renderer,
            color_mode,
        );
        let runs = double_grid.dirty_runs();

        // Emit ANSI
        ascii_renderer::terminal::emit_cells(
            double_grid.write_buffer(),
            &runs,
            &mut output_state,
            &mut out,
            &palette,
            mono,
        );
        double_grid.present();
    }

    out
}

/// Runs the Phase 5 threaded+SIMD pipeline and returns the concatenated ANSI output.
fn run_threaded_pipeline(
    frame_count: usize,
    cols: usize,
    rows: usize,
    fps: f64,
    color_mode: ColorMode,
    ramp_str: &str,
    invert: bool,
) -> Vec<u8> {
    // Decoder produces grid-resolution frames (cols x rows), exactly as the
    // FFmpegDecoder does with `-s cols x rows`.
    let decoder = MockDecoder::new(frame_count, cols, rows, fps);
    let writer = MockWriter::default();
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.target_fps = Some(fps);
    config.color_mode = color_mode;
    config.ramp_str = ramp_str.to_string();
    config.invert = invert;
    config.yuv_pool_capacity = 3;
    config.output_pool_capacity = 3;
    config.channel_capacity = 2;
    config.drop_threshold = Duration::from_millis(100);

    run_pipeline_with_decoder_and_writer(decoder, config, Arc::clone(&shutdown), writer.clone())
        .expect("Pipeline must complete cleanly");

    writer.bytes()
}

/// Main golden regression: scalar output == threaded output for monochrome.
#[test]
fn golden_monochrome() {
    let frame_count = 10;
    let cols = 40;
    let rows = 20;
    let fps = 30.0;
    let color_mode = ColorMode::Monochrome;
    let ramp_str = ascii_renderer::render::SHORT_RAMP;
    let invert = false;

    let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
    let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

    assert_eq!(scalar, threaded, "Monochrome golden regression failed");
}

/// Golden regression for TrueColor mode.
#[test]
fn golden_truecolor() {
    let frame_count = 10;
    let cols = 40;
    let rows = 20;
    let fps = 30.0;
    let color_mode = ColorMode::TrueColor;
    let ramp_str = ascii_renderer::render::SHORT_RAMP;
    let invert = false;

    let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
    let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

    assert_eq!(scalar, threaded, "TrueColor golden regression failed");
}

/// Golden regression with invert flag.
#[test]
fn golden_invert() {
    let frame_count = 10;
    let cols = 40;
    let rows = 20;
    let fps = 30.0;
    let color_mode = ColorMode::Monochrome;
    let ramp_str = ascii_renderer::render::SHORT_RAMP;
    let invert = true;

    let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
    let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

    assert_eq!(scalar, threaded, "Invert golden regression failed");
}

/// Golden regression with different grid sizes (odd dimensions, non-multiple-of-8).
#[test]
fn golden_odd_dimensions() {
    let test_cases = [
        (33, 17),
        (17, 33),
        (7, 7),
        (100, 100),
        (1, 1),
        (64, 64),
        (33, 33),
    ];

    for (cols, rows) in test_cases {
        let frame_count = 5;
        let fps = 30.0;
        let color_mode = ColorMode::Monochrome;
        let ramp_str = ascii_renderer::render::SHORT_RAMP;
        let invert = false;

        let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
        let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

        assert_eq!(
            scalar, threaded,
            "Golden regression failed for {}x{} grid",
            cols, rows
        );
    }
}

/// Golden regression with many frames to catch cumulative state drift.
#[test]
fn golden_many_frames() {
    let frame_count = 50;
    let cols = 50;
    let rows = 25;
    let fps = 30.0;
    let color_mode = ColorMode::Monochrome;
    let ramp_str = ascii_renderer::render::SHORT_RAMP;
    let invert = false;

    let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
    let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

    assert_eq!(scalar, threaded, "Many-frames golden regression failed");
}

/// Golden regression with DETAILED_RAMP (different character set).
#[test]
fn golden_detailed_ramp() {
    let frame_count = 10;
    let cols = 40;
    let rows = 20;
    let fps = 30.0;
    let color_mode = ColorMode::Monochrome;
    let ramp_str = ascii_renderer::render::DETAILED_RAMP;
    let invert = false;

    let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
    let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

    assert_eq!(scalar, threaded, "DETAILED_RAMP golden regression failed");
}

/// Golden regression with BLOCK_RAMP.
#[test]
fn golden_block_ramp() {
    let frame_count = 10;
    let cols = 40;
    let rows = 20;
    let fps = 30.0;
    let color_mode = ColorMode::Monochrome;
    let ramp_str = ascii_renderer::render::BLOCK_RAMP;
    let invert = false;

    let scalar = run_scalar_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);
    let threaded = run_threaded_pipeline(frame_count, cols, rows, fps, color_mode, ramp_str, invert);

    assert_eq!(scalar, threaded, "BLOCK_RAMP golden regression failed");
}