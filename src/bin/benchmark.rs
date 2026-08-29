// Phase 2 — Manual Benchmarks (Milestone 2.9)
//
// Measures key pipeline stages on a 1080p-like workload.
// Usage: cargo run --release --bin benchmark

use ascii_renderer::render::edge::{build_luma_map, build_gradient_map, compute_frame_edges};
use std::time::Instant;

fn main() {
    // 1080p source dimensions
    const SRC_W: usize = 1920;
    const SRC_H: usize = 1080;
    // Output grid (80x45 character cells for 16:9)
    const COLS: usize = 80;
    const ROWS: usize = 45;

    // Generate synthetic 1080p RGB data with edges
    let mut rgb = vec![0u8; SRC_W * SRC_H * 3];
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            let i = (y * SRC_W + x) * 3;
            // Vertical edge at x=960 (center)
            let c: u8 = if x < 960 { 0 } else { 255 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }

    println!("=== ASCII Renderer Manual Benchmarks (Milestone 2.9) ===");
    println!("Source: {}x{}, Grid: {}x{}", SRC_W, SRC_H, COLS, ROWS);
    println!();

    // Warm-up
    for _ in 0..3 {
        let _ = build_luma_map(&rgb);
        let _ = build_gradient_map(&build_luma_map(&rgb), SRC_W, SRC_H);
        let _ = compute_frame_edges(&rgb, SRC_W, SRC_H, COLS, ROWS);
    }

    // Benchmark: Sobel (build_luma_map + build_gradient_map)
    let iterations = 10;
    let mut total_sobel = 0u128;
    for _ in 0..iterations {
        let start = Instant::now();
        let luma = build_luma_map(&rgb);
        let _ = build_gradient_map(&luma, SRC_W, SRC_H);
        total_sobel += start.elapsed().as_micros();
    }
    let avg_sobel_ms = total_sobel as f64 / iterations as f64 / 1000.0;
    println!("Sobel (luma + gradient): {:.2} ms avg over {} runs", avg_sobel_ms, iterations);

    // Benchmark: NMS + Hysteresis (full compute_frame_edges pipeline)
    let mut total_nms_hyst = 0u128;
    for _ in 0..iterations {
        let start = Instant::now();
        let _edges = compute_frame_edges(&rgb, SRC_W, SRC_H, COLS, ROWS);
        total_nms_hyst += start.elapsed().as_micros();
    }
    let avg_nms_hyst_ms = total_nms_hyst as f64 / iterations as f64 / 1000.0;
    println!("NMS + Hysteresis (full pipeline): {:.2} ms avg over {} runs", avg_nms_hyst_ms, iterations);

    // Total pipeline
    let total_ms = avg_sobel_ms + avg_nms_hyst_ms;
    println!();
    println!("Total edge pipeline: {:.2} ms ({:.1} FPS theoretical)", total_ms, 1000.0 / total_ms);

    // Also test on a more complex scene (checkerboard-like)
    println!();
    println!("--- Complex scene (checkerboard pattern) ---");
    let mut complex_rgb = vec![0u8; SRC_W * SRC_H * 3];
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            let i = (y * SRC_W + x) * 3;
            let block_x = x / 40;
            let block_y = y / 40;
            let c: u8 = if (block_x + block_y) % 2 == 0 { 0 } else { 255 };
            complex_rgb[i] = c;
            complex_rgb[i + 1] = c;
            complex_rgb[i + 2] = c;
        }
    }

    let mut total_complex = 0u128;
    for _ in 0..iterations {
        let start = Instant::now();
        let _edges = compute_frame_edges(&complex_rgb, SRC_W, SRC_H, COLS, ROWS);
        total_complex += start.elapsed().as_micros();
    }
    let avg_complex_ms = total_complex as f64 / iterations as f64 / 1000.0;
    println!("NMS + Hysteresis (complex): {:.2} ms avg over {} runs", avg_complex_ms, iterations);
}