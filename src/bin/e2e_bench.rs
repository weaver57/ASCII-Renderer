//! Phase 5.11 — End-to-end throughput benchmark (§6 of the architecture plan).
//!
//! This is Phase 5's headline exit criterion: a *measured, documented* speedup
//! number, obtained by actually running both pipelines, not an estimate.
//!
//! It drives the SAME representative test video (used since Phase 3/4's
//! benchmarks) through
//!
//!   (a) the fully single-threaded, scalar Phase 1–4 pipeline, and
//!   (b) the Phase 5 threaded + SIMD pipeline,
//!
//! measuring wall-clock frames/second for each, sustained over a configurable
//! number of frames (default 300 — well past the first few frames' one-time
//! setup costs), and reporting the speedup multiplier.
//!
//! To make the comparison fair, the video is decoded ONCE up front (outside the
//! timing window) into in-memory grid-resolution YUV frames, and both pipelines
//! consume the *identical* frame sequence. Decode is therefore excluded from
//! both sides, so the number measures processing throughput — which is exactly
//! what Phase 5 optimizes.
//!
//! The Phase 5 pipeline is run with `pacing = false` so the Render thread does
//! not sleep to the source frame rate and the Process thread does not drop
//! frames: otherwise the benchmark would measure wall-clock pacing (30 fps),
//! not compute throughput.
//!
//! Usage:
//!   cargo run --release --bin e2e_bench [--video path] [--grid WxH] [--frames N]

use std::io;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use ascii_renderer::caps::{self, ColorSupport};
use ascii_renderer::diff::DoubleGrid;
use ascii_renderer::palette::Palette;
use ascii_renderer::parallel::{
    build_gradient_map_simd, init_thread_pool, non_max_suppress_parallel,
};
use ascii_renderer::pipeline::{run_pipeline_with_decoder_and_writer, PipelineConfig};
use ascii_renderer::render::ascii::ColorMode;
use ascii_renderer::render::edge::{build_gradient_map, compute_frame_edges_from_luma, non_max_suppress};
use ascii_renderer::render::grid::build_char_grid_into;
use ascii_renderer::render::temporal::TemporalEdgeSmoother;
use ascii_renderer::render::AsciiRenderer;
use ascii_renderer::terminal::{self, OutputState};
use ascii_renderer::video::decoder::{DecodedFrame, FFmpegDecoder, OutputFormat, VideoDecoder};
use ascii_renderer::video::yuv::{
    build_luma_map_y, create_yuv_frame, downsample_yuv_planes, ColorRange, ColorSpace,
};

/// Command-line arguments for the end-to-end benchmark.
#[derive(Parser, Debug)]
#[command(name = "e2e_bench", about = "Phase 5 end-to-end throughput benchmark")]
struct Args {
    /// Path to the representative test video.
    #[arg(long, default_value = "video0_1-3-1.mp4")]
    video: String,

    /// Output grid dimensions as WxH.
    #[arg(long, default_value = "160x90")]
    grid: String,

    /// Number of frames to process on each side (sustained measurement).
    #[arg(long, default_value = "300")]
    frames: usize,
}

fn main() {
    let args = Args::parse();
    let (cols, rows) = parse_grid(&args.grid);
    let frames = args.frames.max(1);

    // ── 0. Decode the representative video ONCE, outside the timing window ──
    let frames_vec = decode_frames(&args.video, cols, rows, frames);
    println!(
        "=== Phase 5 End-to-End Throughput Benchmark ===");
    println!(
        "Video: {} | Grid: {}x{} | Frames/side: {}",
        args.video, cols, rows, frames_vec.len()
    );
    println!();

    // ── 1. (a) Single-threaded scalar Phase 1–4 pipeline ──
    let scalar_fps = bench_scalar(&frames_vec, cols, rows);
    println!("(a) Scalar Phase 1-4 pipeline : {:8.1} FPS", scalar_fps);

    // ── 2. (b) Phase 5 threaded + SIMD pipeline ──
    let threaded_fps = bench_threaded(&frames_vec, cols, rows);
    println!("(b) Threaded + SIMD pipeline  : {:8.1} FPS", threaded_fps);

    let speedup = threaded_fps / scalar_fps;
    println!();
    println!("Speedup: {:.2}x", speedup);
    println!("(end-to-end throughput, same video, sustained over {} frames)", frames_vec.len());
    println!();

    // ── 3. Component-level speedups (where the gain came from) ──
    bench_components();
}

fn parse_grid(s: &str) -> (usize, usize) {
    let (w, h) = s.split_once('x').expect("--grid must be WxH");
    (w.parse().expect("invalid width"), h.parse().expect("invalid height"))
}

/// Decodes up to `frames` grid-resolution YUV420P frames from the video into
/// memory. Timed separately so decode is excluded from both pipeline timings.
fn decode_frames(path: &str, cols: usize, rows: usize, frames: usize) -> Vec<DecodedFrame> {
    let mut decoder = FFmpegDecoder::new(
        path,
        cols,
        rows,
        Some(30.0),
        OutputFormat::Yuv420p,
    )
    .expect("failed to open video");
    let mut out = Vec::new();
    while let Some(frame) = decoder.read_frame().expect("decode error") {
        out.push(frame);
        if out.len() >= frames {
            break;
        }
    }
    if out.is_empty() {
        eprintln!("warning: decoded 0 frames from {path}; nothing to benchmark");
    }
    out
}

/// Runs the single-threaded scalar Phase 1–4 pipeline over the given decoded
/// frames (mirroring `main.rs`'s YUV render loop) and returns sustained FPS.
fn bench_scalar(frames: &[DecodedFrame], cols: usize, rows: usize) -> f64 {
    if frames.is_empty() {
        return 0.0;
    }
    let width = cols;
    let height = rows;

    let color_mode = ColorMode::Monochrome;
    let ramp_str = ascii_renderer::render::SHORT_RAMP;
    let color_space = ascii_renderer::video::yuv::detect_color_space(rows as u32, None);

    let color_support = caps::detect_color_support();
    let palette = match color_support {
        ColorSupport::Palette256 => Palette::xterm256(),
        ColorSupport::Basic16 => Palette::basic16(),
        _ => Palette::xterm256(),
    };
    let mut output_state = OutputState::new(color_support, false);
    let mut grid = DoubleGrid::new(cols, rows);
    let mut smoother = TemporalEdgeSmoother::new(0.35);
    let renderer = AsciiRenderer::new(ramp_str, color_mode, false);

    let mut luma_map = Vec::new();
    let mut out_bytes = Vec::new();
    let mut sink = io::sink();
    let mono = true;

    let start = Instant::now();
    for frame in frames {
        let yuv_frame = create_yuv_frame(
            &frame.data,
            width as u32,
            height as u32,
            width,
            width / 2,
            width / 2,
            ColorRange::Limited,
            color_space,
            frame.pts_seconds,
        );

        build_luma_map_y(&yuv_frame.y, &mut luma_map);
        let cell_edges = compute_frame_edges_from_luma(&luma_map, width, height, cols, rows);
        let (cell_luma, cell_color) =
            ascii_renderer::video::yuv::downsample_yuv(&yuv_frame, cols, rows);
        let smoothed = smoother.update(cell_edges);

        {
            let write = grid.write_buffer();
            build_char_grid_into(
                write,
                &cell_luma,
                &cell_color,
                &smoothed,
                &renderer,
                color_mode,
            );
        }
        let runs = grid.dirty_runs();
        out_bytes.clear();
        terminal::emit_cells(
            grid.write_buffer(),
            &runs,
            &mut output_state,
            &mut out_bytes,
            &palette,
            mono,
        );
        terminal::write_frame(&out_bytes, false, &mut sink).unwrap();
        grid.present();
    }
    let elapsed = start.elapsed();
    frames.len() as f64 / elapsed.as_secs_f64()
}

/// Replays an in-memory list of decoded frames through the `VideoDecoder`
/// trait so the threaded pipeline consumes the same frames as the scalar path.
struct ReplayDecoder {
    frames: Vec<DecodedFrame>,
    cursor: usize,
    fps: f64,
}

impl ReplayDecoder {
    fn new(frames: Vec<DecodedFrame>, fps: f64) -> Self {
        Self { frames, cursor: 0, fps }
    }
}

impl VideoDecoder for ReplayDecoder {
    fn read_frame(&mut self) -> anyhow::Result<Option<DecodedFrame>> {
        if self.cursor >= self.frames.len() {
            return Ok(None);
        }
        let frame = self.frames[self.cursor].clone();
        self.cursor += 1;
        Ok(Some(frame))
    }

    fn avg_fps(&self) -> f64 {
        self.fps
    }
}

/// Runs the Phase 5 threaded + SIMD pipeline (pacing disabled) over the given
/// decoded frames and returns sustained FPS.
fn bench_threaded(frames: &[DecodedFrame], cols: usize, rows: usize) -> f64 {
    if frames.is_empty() {
        return 0.0;
    }
    init_thread_pool();

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.color_mode = ColorMode::Monochrome;
    config.ramp_str = ascii_renderer::render::SHORT_RAMP.to_string();
    config.invert = false;
    config.target_fps = Some(30.0);
    config.pacing = false; // throughput mode: no pacing, no dropping
    config.yuv_pool_capacity = 4;
    config.output_pool_capacity = 4;
    config.channel_capacity = 2;

    let shutdown = Arc::new(AtomicBool::new(false));
    let sink = io::sink();
    let decoder = ReplayDecoder::new(frames.to_vec(), 30.0);

    let start = Instant::now();
    run_pipeline_with_decoder_and_writer(decoder, config, Arc::clone(&shutdown), sink)
        .expect("threaded pipeline failed");
    let elapsed = start.elapsed();
    frames.len() as f64 / elapsed.as_secs_f64()
}

/// Component-level before/after microbenchmarks, per the plan's "also worth
/// recording" note: Sobel-alone and box-filter-alone scalar vs rayon+SIMD.
fn bench_components() {
    use ascii_renderer::parallel::downsample_yuv_planes_parallel as par_down;

    init_thread_pool();

    // ── Sobel + NMS, scalar vs SIMD, on 1080p synthetic luma ──
    const W: usize = 1920;
    const H: usize = 1080;
    let mut luma = vec![0.0f32; W * H];
    for y in 0..H {
        for x in 0..W {
            // Vertical step + diagonal + checkerboard for rich gradient angles.
            let step: f32 = if x < W / 2 { 30.0 } else { 220.0 };
            let diag: f32 = if x > y { 255.0 } else { 0.0 };
            let ck: f32 = if ((x / 40) + (y / 40)) % 2 == 0 { 40.0 } else { 200.0 };
            luma[y * W + x] = (step + diag * 0.4 + ck * 0.3).min(255.0);
        }
    }

    const ITERS: usize = 5;
    let mut t_scalar_sobel = 0u128;
    let mut t_simd_sobel = 0u128;
    let mut t_scalar_nms = 0u128;
    let mut t_simd_nms = 0u128;
    for _ in 0..ITERS {
        let s = Instant::now();
        let g = build_gradient_map(&luma, W, H);
        t_scalar_sobel += s.elapsed().as_micros();
        let s = Instant::now();
        let _ = non_max_suppress(&g);
        t_scalar_nms += s.elapsed().as_micros();

        let s = Instant::now();
        let g = build_gradient_map_simd(&luma, W, H);
        t_simd_sobel += s.elapsed().as_micros();
        let s = Instant::now();
        let _ = non_max_suppress_parallel(&g);
        t_simd_nms += s.elapsed().as_micros();
    }

    // ── Box-filter downsample, scalar vs parallel, 1080p → 160x90 ──
    const G_COLS: usize = 160;
    const G_ROWS: usize = 90;
    let y = luma.iter().map(|&v| v as u8).collect::<Vec<_>>();
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    let mut cl_scalar = vec![0.0f32; G_COLS * G_ROWS];
    let mut cc_scalar = vec![(0u8, 0u8, 0u8); G_COLS * G_ROWS];
    let mut cl_par = cl_scalar.clone();
    let mut cc_par = cc_scalar.clone();

    let mut t_scalar_down = 0u128;
    let mut t_par_down = 0u128;
    for _ in 0..ITERS {
        let s = Instant::now();
        downsample_yuv_planes(&y, &u, &v, W, H, ColorSpace::Bt709, G_COLS, G_ROWS, &mut cl_scalar, &mut cc_scalar);
        t_scalar_down += s.elapsed().as_micros();

        let s = Instant::now();
        par_down(&y, &u, &v, W, H, ColorSpace::Bt709, G_COLS, G_ROWS, &mut cl_par, &mut cc_par);
        t_par_down += s.elapsed().as_micros();
    }

    let sobel_scalar = t_scalar_sobel as f64 / ITERS as f64 / 1000.0;
    let sobel_simd = t_simd_sobel as f64 / ITERS as f64 / 1000.0;
    let nms_scalar = t_scalar_nms as f64 / ITERS as f64 / 1000.0;
    let nms_simd = t_simd_nms as f64 / ITERS as f64 / 1000.0;
    let down_scalar = t_scalar_down as f64 / ITERS as f64 / 1000.0;
    let down_par = t_par_down as f64 / ITERS as f64 / 1000.0;

    println!("--- Component-level (1080p synthetic, {ITERS} iters, ms/frame) ---");
    println!(
        "Sobel      : scalar {:7.2} ms | SIMD {:7.2} ms | {:.2}x",
        sobel_scalar, sobel_simd, sobel_scalar / sobel_simd
    );
    println!(
        "NMS        : scalar {:7.2} ms | rayon {:7.2} ms | {:.2}x",
        nms_scalar, nms_simd, nms_scalar / nms_simd
    );
    println!(
        "Box-filter : scalar {:7.2} ms | rayon {:7.2} ms | {:.2}x",
        down_scalar, down_par, down_scalar / down_par
    );
}
