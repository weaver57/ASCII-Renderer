# Phase 2 Performance Baselines (Milestone 2.9)

Recorded on: 2026-08-30
Hardware: (unspecified — user machine)
Build: `cargo build --release`
Rust: stable (see `Cargo.lock`)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Source resolution | 1920 × 1080 (1080p) |
| Output grid | 80 × 45 character cells |
| Iterations per test | 10 (after 3 warm-up) |

## Results

### Simple Vertical Edge (1920×1080, single step edge at center)

| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| Sobel (luma + gradient) | **112.86** | `build_luma_map` + `build_gradient_map` |
| NMS + Hysteresis (full pipeline) | **118.67** | `compute_frame_edges` |
| **Total edge pipeline** | **231.54** | Theoretical max **4.3 FPS** |

### Complex Checkerboard Pattern (1920×1080, 40px blocks)

| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| NMS + Hysteresis (full pipeline) | **137.63** | More edges to process |

## Interpretation

- The edge pipeline dominates frame time at ~230 ms/frame for 1080p on this hardware.
- Real-world video playback will need SIMD/parallelization (Phase 5) or resolution scaling to hit 30 FPS.
- The checkerboard pattern is ~16% slower due to more edge pixels surviving NMS/hysteresis.

## How to Reproduce

```sh
cd ASCII
cargo build --release
cargo run --release --bin benchmark
```

## Next Steps (Phase 3+)

- Phase 3: Wire `TemporalEdgeSmoother` into video loop (amortize edge cost across frames)
- Phase 4: Character-cell SIMD render path
- Phase 5: SIMD Sobel (AVX2/NEON), parallel NMS