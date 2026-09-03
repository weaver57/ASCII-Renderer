//! Differential parity tests for Rayon-accelerated data parallelism.
//!
//! Asserts byte-for-byte and float-exact equivalence between the single-threaded
//! `downsample_yuv_planes` and the multi-threaded `downsample_yuv_planes_parallel`.

use ascii_renderer::parallel::downsample_yuv_planes_parallel;
use ascii_renderer::video::yuv::{downsample_yuv_planes, ColorSpace};

/// Generates synthetic YUV420P planes for testing.
///
/// Creates deterministic, structured data that allows verifying block averaging correctness.
fn generate_test_yuv(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    for row in 0..height {
        for col in 0..width {
            // A dynamic gradient that helps verify grid mapping
            y[row * width + col] = ((row * 31 + col * 17 + 42) % 256) as u8;
        }
    }

    let cw = (width / 2).max(1);
    let ch = (height / 2).max(1);
    let mut u = vec![128u8; cw * ch];
    let mut v = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = ((row * 7 + col * 13 + 64) % 256) as u8;
            v[row * cw + col] = ((row * 11 + col * 23 + 128) % 256) as u8;
        }
    }

    (y, u, v)
}

/// Parity across a wide variety of resolutions, including those that
/// create fractional block boundaries and edge-case remainders.
#[test]
fn test_parallel_downsample_matches_scalar_across_resolutions() {
    let test_cases: Vec<(usize, usize, usize, usize, ColorSpace)> = vec![
        // (src_w, src_h, cols, rows, ColorSpace)
        (640, 480, 80, 24, ColorSpace::Bt709),
        (1920, 1080, 120, 40, ColorSpace::Bt709),
        (320, 240, 40, 15, ColorSpace::Bt601),
        (1280, 720, 100, 30, ColorSpace::Bt601),
        (100, 100, 33, 33, ColorSpace::Bt709),
        // Odd dimensions that don't divide evenly
        (1919, 1081, 101, 41, ColorSpace::Bt709),
        (853, 480, 80, 30, ColorSpace::Bt601),
        // Edge cases: very small or narrow grids
        (32, 32, 2, 2, ColorSpace::Bt601),
        (16, 16, 16, 16, ColorSpace::Bt709),
        (1000, 1000, 7, 7, ColorSpace::Bt709),
        (1, 1, 1, 1, ColorSpace::Bt601),
    ];

    for (src_w, src_h, cols, rows, cs) in test_cases {
        let (y, u, v) = generate_test_yuv(src_w, src_h);
        let mut scalar_luma = vec![0.0f32; cols * rows];
        let mut scalar_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes(&y, &u, &v, src_w, src_h, cs, cols, rows, &mut scalar_luma, &mut scalar_color);

        let mut parallel_luma = vec![0.0f32; cols * rows];
        let mut parallel_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes_parallel(&y, &u, &v, src_w, src_h, cs, cols, rows, &mut parallel_luma, &mut parallel_color);

        assert_eq!(scalar_luma.len(), parallel_luma.len());
        for i in 0..scalar_luma.len() {
            // Floating point arithmetic associativity (a+b+c vs (a+b)+c) can
            // cause minute rounding differences, though our exact local arithmetic
            // prevents it here since the loop structures are identical.
            assert!(
                (scalar_luma[i] - parallel_luma[i]).abs() < f32::EPSILON,
                "Luma mismatch at index {} for resolution {src_w}x{src_h} -> {cols}x{rows}: scalar={} parallel={}",
                i, scalar_luma[i], parallel_luma[i]
            );
        }

        assert_eq!(
            scalar_color, parallel_color,
            "Color mismatch for resolution {src_w}x{src_h} -> {cols}x{rows}"
        );
    }
}

/// Stress test to ensure no race conditions occur under concurrent workloads
/// by repeatedly running the parallel function over large inputs.
#[test]
fn test_parallel_downsample_stress_race_conditions() {
    let src_w = 1280;
    let src_h = 720;
    let cols = 160;
    let rows = 45;
    let (y, u, v) = generate_test_yuv(src_w, src_h);

    // Run 100 parallel evaluations of the same data across Rayon threads
    for _ in 0..100 {
        let mut parallel_luma = vec![0.0f32; cols * rows];
        let mut parallel_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes_parallel(&y, &u, &v, src_w, src_h, ColorSpace::Bt709, cols, rows, &mut parallel_luma, &mut parallel_color);

        let mut scalar_luma = vec![0.0f32; cols * rows];
        let mut scalar_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes(&y, &u, &v, src_w, src_h, ColorSpace::Bt709, cols, rows, &mut scalar_luma, &mut scalar_color);

        assert_eq!(scalar_luma, parallel_luma);
        assert_eq!(scalar_color, parallel_color);
    }
}
