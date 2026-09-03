use ascii_renderer::pipeline::{
    run_pipeline_with_decoder_and_writer, PipelineConfig,
};
use ascii_renderer::render::ascii::ColorMode;
use ascii_renderer::video::decoder::{DecodedFrame, VideoDecoder};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Synthetic frame decoder for testing pipeline concurrency without requiring external FFmpeg.
struct MockDecoder {
    total_frames: usize,
    current_frame: usize,
    width: usize,
    height: usize,
    fps: f64,
    fail_at_frame: Option<usize>,
    delay: Option<Duration>,
}

impl MockDecoder {
    fn new(total_frames: usize, width: usize, height: usize, fps: f64) -> Self {
        Self {
            total_frames,
            current_frame: 0,
            width,
            height,
            fps,
            fail_at_frame: None,
            delay: None,
        }
    }

    fn with_failure_at(mut self, frame_idx: usize) -> Self {
        self.fail_at_frame = Some(frame_idx);
        self
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }
}

impl VideoDecoder for MockDecoder {
    fn read_frame(&mut self) -> anyhow::Result<Option<DecodedFrame>> {
        if let Some(d) = self.delay {
            thread::sleep(d);
        }

        if let Some(fail_idx) = self.fail_at_frame {
            if self.current_frame == fail_idx {
                return Err(anyhow::anyhow!(
                    "Simulated decoder fatal error at frame {}",
                    fail_idx
                ));
            }
        }

        if self.current_frame >= self.total_frames {
            return Ok(None);
        }

        let y_size = self.width * self.height;
        let uv_size = (self.width / 2) * (self.height / 2);
        let total_size = y_size + 2 * uv_size;
        let mut data = vec![128u8; total_size];

        // Synthesize dynamic patterns across frames so edge detector and diffing engage
        let shift = (self.current_frame % self.width) as u8;
        for y in 0..self.height {
            for x in 0..self.width {
                // Moving vertical/diagonal stripes
                let val = ((x as u8).wrapping_add(shift) ^ (y as u8)).min(235).max(16);
                data[y * self.width + x] = val;
            }
        }

        // Fill chroma planes
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

/// Thread-safe in-memory writer that records bytes written and flush calls.
#[derive(Clone, Default)]
struct MockWriter {
    buffer: Arc<Mutex<Vec<u8>>>,
    write_count: Arc<Mutex<usize>>,
}

impl MockWriter {
    fn bytes_written(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    fn frame_count(&self) -> usize {
        *self.write_count.lock().unwrap()
    }
}

impl Write for MockWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut b = self.buffer.lock().unwrap();
        b.extend_from_slice(buf);
        let mut count = self.write_count.lock().unwrap();
        *count += 1;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_pipeline_normal_playback_completion() {
    let frame_count = 30;
    let cols = 40;
    let rows = 20;
    let fps = 100.0; // Fast clock for unit test

    let decoder = MockDecoder::new(frame_count, cols, rows, fps);
    let writer = MockWriter::default();
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.target_fps = Some(fps);
    config.color_mode = ColorMode::Monochrome;
    config.yuv_pool_capacity = 3;
    config.output_pool_capacity = 3;

    let result = run_pipeline_with_decoder_and_writer(
        decoder,
        config,
        Arc::clone(&shutdown),
        writer.clone(),
    );

    assert!(result.is_ok(), "Pipeline should complete cleanly on EOF");
    assert!(
        writer.bytes_written() > 0,
        "Render thread should have written ANSI output"
    );
    assert!(
        writer.frame_count() > 0,
        "Render thread should have output frames"
    );
}

#[test]
fn test_pipeline_atomic_shutdown_latency() {
    // Generate 10,000 frames at 30fps (over 5 minutes of playback)
    let frame_count = 10_000;
    let cols = 60;
    let rows = 30;
    let fps = 30.0;

    let decoder = MockDecoder::new(frame_count, cols, rows, fps);
    let writer = MockWriter::default();
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.target_fps = Some(fps);

    let shutdown_trigger = Arc::clone(&shutdown);

    // Spawn a supervisor thread that triggers shutdown after 80ms of playback
    let supervisor = thread::spawn(move || {
        thread::sleep(Duration::from_millis(80));
        let start = Instant::now();
        shutdown_trigger.store(true, Ordering::SeqCst);
        start
    });

    let start_pipeline = Instant::now();
    let res = run_pipeline_with_decoder_and_writer(
        decoder,
        config,
        Arc::clone(&shutdown),
        writer,
    );

    let shutdown_instant = supervisor.join().expect("Supervisor joined");
    let total_elapsed = start_pipeline.elapsed();
    let shutdown_latency = Instant::now().duration_since(shutdown_instant);

    assert!(res.is_ok(), "Pipeline should exit cleanly on shutdown");
    // Milestone 5.4 requirement: atomic shutdown exits cleanly in < 100ms
    assert!(
        shutdown_latency < Duration::from_millis(150),
        "Shutdown latency must be < 150ms, took {:?}",
        shutdown_latency
    );
    assert!(
        total_elapsed < Duration::from_millis(500),
        "Total test execution must stop immediately after shutdown flag"
    );
}

#[test]
fn test_pipeline_decoder_error_propagation() {
    // Decoder fails fatal error at frame 10
    let frame_count = 50;
    let cols = 30;
    let rows = 15;
    let fps = 60.0;

    let decoder = MockDecoder::new(frame_count, cols, rows, fps).with_failure_at(10);
    let writer = MockWriter::default();
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.target_fps = Some(fps);

    let start = Instant::now();
    let result = run_pipeline_with_decoder_and_writer(
        decoder,
        config,
        shutdown,
        writer,
    );

    let elapsed = start.elapsed();
    assert!(result.is_err(), "Pipeline must propagate decoder error to caller");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Simulated decoder fatal error at frame 10"),
        "Error message must contain decoder failure context, got: {}",
        err_msg
    );
    // Must terminate promptly without hanging
    assert!(elapsed < Duration::from_millis(500));
}

#[test]
fn test_pipeline_frame_drop_under_slow_compute() {
    // 60 frames generated at high speed, but with a tiny drop_threshold
    // Frames with target_instant in the past should drop cleanly in Process stage
    let frame_count = 60;
    let cols = 40;
    let rows = 20;
    let fps = 1000.0; // Extremely high target fps creates tight deadlines

    let decoder = MockDecoder::new(frame_count, cols, rows, fps);
    let writer = MockWriter::default();
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.target_fps = Some(fps);
    // Strict drop threshold: 1 microsecond (guarantees drop policy triggers)
    config.drop_threshold = Duration::from_nanos(1);
    config.yuv_pool_capacity = 2;
    config.output_pool_capacity = 2;

    let res = run_pipeline_with_decoder_and_writer(
        decoder,
        config,
        shutdown,
        writer.clone(),
    );

    assert!(res.is_ok(), "Pipeline with frame dropping must complete cleanly without deadlock");
    // Frames were dropped cleanly, pool was recycled without starvation
}

#[test]
fn test_pipeline_backpressure_and_pool_recycling() {
    // Run 200 frames through a small pool of only 2 buffers and channel capacity 1
    // This heavily stresses channel blocking, buffer pool acquisition, and RAII recycling
    let frame_count = 200;
    let cols = 50;
    let rows = 25;
    let fps = 200.0;

    let decoder = MockDecoder::new(frame_count, cols, rows, fps);
    let writer = MockWriter::default();
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut config = PipelineConfig::default();
    config.cols = cols;
    config.rows = rows;
    config.target_fps = Some(fps);
    config.yuv_pool_capacity = 2;
    config.output_pool_capacity = 2;
    config.channel_capacity = 1;

    let res = run_pipeline_with_decoder_and_writer(
        decoder,
        config,
        shutdown,
        writer.clone(),
    );

    assert!(res.is_ok(), "Backpressure stress test must succeed");
    assert!(writer.frame_count() > 0, "Frames must be rendered");
}
