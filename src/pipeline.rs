//! Three-stage video processing pipeline (Decode -> Process -> Render).
//!
//! Implements Milestone 5.3 (§4.4–4.6) of the Phase 5 Concurrency & SIMD Architecture Plan:
//! - **Decode Thread**: Runs FFmpeg subprocess / frame source and computes `target_instant` via `PlaybackClock` (§4.5).
//! - **Process Thread**: Owns all Phase 1–4 compute (Sobel, NMS, Hysteresis, Downsample, Temporal Smoothing, CharGrid, DoubleGrid diffing, Color resolution). Drops stale frames before compute (D4).
//! - **Render Thread**: Applies pacing (sleep until `target_instant`) and writes frames to stdout/writer with DECRQM mode 2026 sync output.
//! - **Buffer Pooling**: Pre-allocated cross-thread recycling with `BufferPool<T>` and `PoolGuard<T>` (D5).

use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::caps::{self, ColorSupport};
use crate::diff::DoubleGrid;
use crate::palette::Palette;
use crate::pool::{BufferPool, PoolGuard};
use crate::parallel::{
    build_gradient_map_simd, downsample_yuv_planes_parallel, init_thread_pool,
    non_max_suppress_parallel,
};
use crate::render::ascii::ColorMode as RenderColorMode;
use crate::render::edge::{aggregate_cell_edges, compute_thresholds, promote_edges, EdgeCellInfo};
use crate::render::grid::build_char_grid_into;
use crate::render::temporal::TemporalEdgeSmoother;
use crate::render::AsciiRenderer;
use crate::terminal::{self, OutputState};
use crate::video::clock::PlaybackClock;
use crate::video::decoder::{FFmpegDecoder, OutputFormat, VideoDecoder};
use crate::video::yuv::{build_luma_map_y, detect_color_space, ColorRange, ColorSpace};

/// Pre-allocated buffer holding raw YUV planes and frame metadata across the Decode -> Process channel.
pub struct PooledYuvBuffers {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub range: ColorRange,
    pub color_space: ColorSpace,
    pub pts_seconds: f64,
    pub target_instant: Instant,
}

impl PooledYuvBuffers {
    /// Creates an empty buffer instance for pool initialization.
    pub fn empty() -> Self {
        Self {
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            width: 0,
            height: 0,
            range: ColorRange::Limited,
            color_space: ColorSpace::Bt709,
            pts_seconds: 0.0,
            target_instant: Instant::now(),
        }
    }
}

/// Pre-allocated buffer holding serialized ANSI bytes and display deadline across the Process -> Render channel.
pub struct PooledOutputBuffer {
    pub bytes: Vec<u8>,
    pub target_instant: Instant,
}

impl PooledOutputBuffer {
    /// Creates an empty output buffer instance for pool initialization.
    pub fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            target_instant: Instant::now(),
        }
    }
}

/// Messages sent from Decode thread to Process thread.
pub enum DecodedMessage {
    Frame(PooledYuvBuffers),
    Eof,
}

/// Messages sent from Process thread to Render thread.
pub enum RenderMessage {
    Ready(PooledOutputBuffer),
    Eof,
}

/// Configuration settings for the three-stage pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub cols: usize,
    pub rows: usize,
    pub target_fps: Option<f64>,
    pub color_mode: RenderColorMode,
    pub ramp_str: String,
    pub invert: bool,
    pub yuv_pool_capacity: usize,
    pub output_pool_capacity: usize,
    pub channel_capacity: usize,
    pub drop_threshold: Duration,
    /// When `true` (the default), the pipeline paces to the source frame rate:
    /// the Render thread sleeps until each frame's `target_instant` and the
    /// Process thread drops frames that fall hopelessly behind (D4). Set to
    /// `false` for throughput benchmarking so the pipeline runs every frame at
    /// maximum speed with no pacing and no dropping.
    pub pacing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            cols: 80,
            rows: 24,
            target_fps: Some(30.0),
            color_mode: RenderColorMode::TrueColor,
            ramp_str: crate::render::SHORT_RAMP.to_string(),
            invert: false,
            yuv_pool_capacity: 4,
            output_pool_capacity: 4,
            channel_capacity: 2,
            drop_threshold: Duration::from_millis(100),
            pacing: true,
        }
    }
}

/// Persistent thread-local compute state inside the Process thread.
///
/// Allocations and state here (such as `DoubleGrid` diffing history and `TemporalEdgeSmoother`)
/// are persistent across the entire playback session and never cross thread boundaries.
pub struct ProcessThreadState {
    pub luma_map: Vec<f32>,
    pub gradient_magnitude: Vec<f32>,
    pub gradient_angle: Vec<f32>,
    pub nms_magnitude: Vec<f32>,
    pub edge_mask: Vec<bool>,
    pub cell_luma: Vec<f32>,
    pub cell_color: Vec<(u8, u8, u8)>,
    pub cell_edges: Vec<Option<EdgeCellInfo>>,
    pub double_grid: DoubleGrid,
    pub smoother: TemporalEdgeSmoother,
    pub color_support: ColorSupport,
    pub palette: Palette,
    pub output_state: OutputState,
    pub renderer: AsciiRenderer,
    pub color_mode: RenderColorMode,
    pub cols: usize,
    pub rows: usize,
}

impl ProcessThreadState {
    /// Creates a new `ProcessThreadState` initialized with terminal capabilities and grid dimensions.
    pub fn new(
        cols: usize,
        rows: usize,
        color_mode: RenderColorMode,
        ramp_str: &str,
        invert: bool,
    ) -> Self {
        let color_support = caps::detect_color_support();
        let sync_output_supported = caps::query_sync_output_support();
        let palette = match color_support {
            ColorSupport::Palette256 => Palette::xterm256(),
            ColorSupport::Basic16 => Palette::basic16(),
            _ => Palette::xterm256(),
        };
        let output_state = OutputState::new(color_support, sync_output_supported);
        let double_grid = DoubleGrid::new(cols, rows);
        let smoother = TemporalEdgeSmoother::new(0.35);
        let renderer = AsciiRenderer::new(ramp_str, color_mode, invert);
        let cell_count = cols * rows;

        Self {
            luma_map: Vec::new(),
            gradient_magnitude: Vec::new(),
            gradient_angle: Vec::new(),
            nms_magnitude: Vec::new(),
            edge_mask: Vec::new(),
            cell_luma: vec![0.0; cell_count],
            cell_color: vec![(0, 0, 0); cell_count],
            cell_edges: vec![None; cell_count],
            double_grid,
            smoother,
            color_support,
            palette,
            output_state,
            renderer,
            color_mode,
            cols,
            rows,
        }
    }

    /// Resizes scratch buffers to accommodate the input source pixel dimensions.
    pub fn ensure_frame_capacity(&mut self, width: usize, height: usize) {
        let px_count = width * height;
        if self.luma_map.len() < px_count {
            self.luma_map.resize(px_count, 0.0);
        }
        if self.gradient_magnitude.len() < px_count {
            self.gradient_magnitude.resize(px_count, 0.0);
        }
        if self.gradient_angle.len() < px_count {
            self.gradient_angle.resize(px_count, 0.0);
        }
        if self.nms_magnitude.len() < px_count {
            self.nms_magnitude.resize(px_count, 0.0);
        }
        if self.edge_mask.len() < px_count {
            self.edge_mask.resize(px_count, false);
        }
    }
}

/// Executes all Phase 1–4 compute steps for a single frame inside the Process thread.
pub fn process_frame(
    yuv: &PooledYuvBuffers,
    state: &mut ProcessThreadState,
    out_bytes: &mut Vec<u8>,
) {
    let src_w = yuv.width as usize;
    let src_h = yuv.height as usize;
    state.ensure_frame_capacity(src_w, src_h);

    // 1. Build luma map directly from Y plane
    build_luma_map_y(&yuv.y, &mut state.luma_map);

    // 2. Full-resolution edge detection (Sobel + NMS + Hysteresis + Aggregation)
    //
    // Sobel and NMS are rayon row-band-sharded (shared-read / disjoint-write,
    // see parallel.rs). Sobel additionally uses SIMD (`wide::f32x8`). Hysteresis
    // promotion remains the single-threaded O(N) queue traversal from Phase 2 (D7).
    let gradient = build_gradient_map_simd(&state.luma_map, src_w, src_h);
    let nms_magnitude = non_max_suppress_parallel(&gradient);
    let (high, low) = compute_thresholds(&nms_magnitude);
    let mask = promote_edges(&nms_magnitude, low, high, src_w, src_h);
    state.cell_edges = aggregate_cell_edges(
        &mask,
        &gradient,
        state.cols,
        state.rows,
        src_w as u32,
        src_h as u32,
    );

    // 3. Temporal edge smoothing
    let smoothed = state.smoother.update(state.cell_edges.clone());

    // 4. Downsample YUV planes into per-cell luma & RGB colors (zero heap allocation)
    downsample_yuv_planes_parallel(
        &yuv.y,
        &yuv.u,
        &yuv.v,
        src_w,
        src_h,
        yuv.color_space,
        state.cols,
        state.rows,
        &mut state.cell_luma,
        &mut state.cell_color,
    );


    // 5. Materialize character grid into the write buffer
    build_char_grid_into(
        state.double_grid.write_buffer(),
        &state.cell_luma,
        &state.cell_color,
        &smoothed,
        &state.renderer,
        state.color_mode,
    );

    // 6. Compute dirty diff runs between current and previous frame
    let runs = state.double_grid.dirty_runs();

    // 7. Emit ANSI escape sequences for dirty runs into output buffer
    out_bytes.clear();
    let mono = state.color_mode == RenderColorMode::Monochrome;
    terminal::emit_cells(
        state.double_grid.write_buffer(),
        &runs,
        &mut state.output_state,
        out_bytes,
        &state.palette,
        mono,
    );

    // 8. Present grid (swap write buffer into read buffer for subsequent diffing)
    state.double_grid.present();
}

/// Runs the 3-stage video pipeline using any decoder implementing `VideoDecoder` and a generic output writer.
pub fn run_pipeline_with_decoder_and_writer<D: VideoDecoder + Send + 'static, W: Write + Send + 'static>(
    mut decoder: D,
    config: PipelineConfig,
    shutdown: Arc<AtomicBool>,
    mut writer: W,
) -> Result<()> {
    // Initialize Rayon thread pool once, with headroom for Decode/Render OS threads (§4.7)
    init_thread_pool();

    let yuv_pool = Arc::new(BufferPool::new(config.yuv_pool_capacity, PooledYuvBuffers::empty));

    let output_pool = Arc::new(BufferPool::new(
        config.output_pool_capacity,
        PooledOutputBuffer::empty,
    ));

    let (decode_to_process_tx, decode_to_process_rx): (
        Sender<DecodedMessage>,
        Receiver<DecodedMessage>,
    ) = bounded(config.channel_capacity);
    let (process_to_render_tx, process_to_render_rx): (
        Sender<RenderMessage>,
        Receiver<RenderMessage>,
    ) = bounded(config.channel_capacity);

    let decode_shutdown = Arc::clone(&shutdown);
    let decode_yuv_pool = Arc::clone(&yuv_pool);
    let decode_config = config.clone();

    // ── 1. Decode Thread ────────────────────────────────────────────────────────
    let decode_handle = std::thread::spawn(move || -> Result<()> {
        let mut clock: Option<PlaybackClock> = None;
        let mut frame_count: u64 = 0;

        loop {
            if decode_shutdown.load(Ordering::Relaxed) {
                break;
            }

            match decoder.read_frame()? {
                None => {
                    let _ = decode_to_process_tx.send(DecodedMessage::Eof);
                    break;
                }
                Some(decoded) => {
                    let pts = decoded.pts_seconds.unwrap_or_else(|| {
                        PlaybackClock::estimate_pts(
                            frame_count,
                            clock.as_ref().map_or(0.0, |c| c.avg_frame_rate() * 0.0),
                            decoder.avg_fps(),
                        )
                    });
                    frame_count += 1;

                    let target_instant = if decode_config.pacing {
                        let clock_ref = clock.get_or_insert_with(|| {
                            PlaybackClock::starting_now(pts, decoder.avg_fps())
                        });
                        clock_ref.target_instant(pts)
                    } else {
                        // Throughput mode: every frame is "due now", so the
                        // Process thread never drops it and the Render thread
                        // never sleeps. Measures raw compute throughput.
                        Instant::now()
                    };

                    let mut guard = decode_yuv_pool.acquire();
                    let width = decode_config.cols;
                    let height = decode_config.rows;
                    guard.width = width as u32;
                    guard.height = height as u32;
                    guard.range = ColorRange::Limited;
                    guard.color_space = detect_color_space(height as u32, None);
                    guard.pts_seconds = pts;
                    guard.target_instant = target_instant;

                    let y_size = width * height;
                    let uv_size = (width / 2) * (height / 2);

                    if decoded.data.len() >= y_size + 2 * uv_size {
                        guard.y.resize(y_size, 0);
                        guard.y.copy_from_slice(&decoded.data[0..y_size]);

                        guard.u.resize(uv_size, 0);
                        guard.u.copy_from_slice(&decoded.data[y_size..y_size + uv_size]);

                        guard.v.resize(uv_size, 0);
                        guard.v.copy_from_slice(&decoded.data[y_size + uv_size..y_size + 2 * uv_size]);
                    }

                    if decode_to_process_tx
                        .send(DecodedMessage::Frame(guard.into_inner()))
                        .is_err()
                    {
                        break; // Process thread disconnected / shut down
                    }
                }
            }
        }
        Ok(())
    });

    // ── 2. Process Thread ───────────────────────────────────────────────────────
    let process_shutdown = Arc::clone(&shutdown);
    let process_yuv_pool = Arc::clone(&yuv_pool);
    let process_output_pool = Arc::clone(&output_pool);
    let process_config = config.clone();

    let process_handle = std::thread::spawn(move || -> Result<()> {
        let mut state = ProcessThreadState::new(
            process_config.cols,
            process_config.rows,
            process_config.color_mode,
            &process_config.ramp_str,
            process_config.invert,
        );

        loop {
            if process_shutdown.load(Ordering::Relaxed) {
                break;
            }

            match decode_to_process_rx.recv() {
                Err(_) => break, // Decode thread closed / crashed
                Ok(DecodedMessage::Eof) => {
                    let _ = process_to_render_tx.send(RenderMessage::Eof);
                    break;
                }
                Ok(DecodedMessage::Frame(yuv_buffers)) => {
                    let yuv_guard = PoolGuard::wrap(yuv_buffers, process_yuv_pool.free_sender());

                    // Dual-point drop policy (D4, first check): drop late frames before expensive compute
                    if process_config.pacing {
                        let now = Instant::now();
                        if now > yuv_guard.target_instant + process_config.drop_threshold {
                            continue; // yuv_guard dropped here -> auto-returned to yuv_pool
                        }
                    }

                    let mut out_guard = process_output_pool.acquire();
                    process_frame(&yuv_guard, &mut state, &mut out_guard.bytes);
                    out_guard.target_instant = yuv_guard.target_instant;

                    if process_to_render_tx
                        .send(RenderMessage::Ready(out_guard.into_inner()))
                        .is_err()
                    {
                        break; // Render thread disconnected / shut down
                    }
                }
            }
        }
        Ok(())
    });

    // ── 3. Render Thread ────────────────────────────────────────────────────────
    let render_shutdown = Arc::clone(&shutdown);
    let render_output_pool = Arc::clone(&output_pool);
    let render_config = config.clone();
    let sync_output_supported = caps::query_sync_output_support();

    let render_handle = std::thread::spawn(move || -> Result<()> {
        loop {
            if render_shutdown.load(Ordering::Relaxed) {
                break;
            }

            match process_to_render_rx.recv() {
                Err(_) => break, // Process thread closed / crashed
                Ok(RenderMessage::Eof) => break,
                Ok(RenderMessage::Ready(output_buf)) => {
                    let out_guard =
                        PoolGuard::wrap(output_buf, render_output_pool.free_sender());

                    // Dual-point drop policy (D4, second check): sleep until target presentation time
                    if render_config.pacing {
                        let now = Instant::now();
                        if now < out_guard.target_instant {
                            let sleep_dur = out_guard.target_instant - now;
                            let step = Duration::from_millis(5);
                            let mut remaining = sleep_dur;
                            while remaining > Duration::ZERO {
                                if render_shutdown.load(Ordering::Relaxed) {
                                    break;
                                }
                                let s = remaining.min(step);
                                std::thread::sleep(s);
                                remaining = remaining.saturating_sub(s);
                            }
                        }
                    }

                    if !render_shutdown.load(Ordering::Relaxed) {
                        terminal::write_frame(
                            &out_guard.bytes,
                            sync_output_supported,
                            &mut writer,
                        )?;
                    }
                }
            }
        }
        Ok(())
    });

    // Join all three threads and propagate any panics / errors
    let res_decode = decode_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Decode thread panicked: {:?}", e))?;
    let res_process = process_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Process thread panicked: {:?}", e))?;
    let res_render = render_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Render thread panicked: {:?}", e))?;

    res_decode.context("Decode thread failed")?;
    res_process.context("Process thread failed")?;
    res_render.context("Render thread failed")?;

    Ok(())
}

/// Runs the 3-stage video pipeline using a generic output writer.
pub fn run_pipeline_with_writer<W: Write + Send + 'static>(
    video_path: &Path,
    config: PipelineConfig,
    shutdown: Arc<AtomicBool>,
    writer: W,
) -> Result<()> {
    let decoder = FFmpegDecoder::new(
        video_path,
        config.cols,
        config.rows,
        config.target_fps,
        OutputFormat::Yuv420p,
    )?;
    run_pipeline_with_decoder_and_writer(decoder, config, shutdown, writer)
}

/// Runs the 3-stage video pipeline targeting the terminal standard output.
pub fn run_pipeline(
    video_path: &Path,
    config: PipelineConfig,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let stdout = io::stdout();
    let writer = BufWriter::with_capacity(128 * 1024, stdout);
    run_pipeline_with_writer(video_path, config, shutdown, writer)
}
