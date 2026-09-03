//! Differential parity tests for SIMD-accelerated kernels.
//!
//! Asserts that SIMD implementations produce outputs identical to their scalar
//! Phase 1–2 reference implementations within a small float tolerance.

use ascii_renderer::parallel::{
    downsample_yuv_planes_parallel, downsample_yuv_planes_simd,
};
use ascii_renderer::video::yuv::{downsample_yuv_planes, ColorSpace};

/// Generates synthetic YUV420P planes for testing.
fn generate_test_yuv(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    for row in 0..height {
        for col in 0..width {
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

/// Parity test: SIMD box-filter vs scalar box-filter.
#[test]
fn test_simd_downsample_matches_scalar() {
    let test_cases: Vec<(usize, usize, usize, usize, ColorSpace)> = vec![
        (640, 480, 80, 24, ColorSpace::Bt709),
        (1920, 1080, 120, 40, ColorSpace::Bt709),
        (320, 240, 40, 15, ColorSpace::Bt601),
        (1280, 720, 100, 30, ColorSpace::Bt601),
        (100, 100, 33, 33, ColorSpace::Bt709),
        (1919, 1081, 101, 41, ColorSpace::Bt709),
        (853, 480, 80, 30, ColorSpace::Bt601),
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

        let mut simd_luma = vec![0.0f32; cols * rows];
        let mut simd_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes_simd(&y, &u, &v, src_w, src_h, cs, cols, rows, &mut simd_luma, &mut simd_color);

        assert_eq!(scalar_luma.len(), simd_luma.len());
        for i in 0..scalar_luma.len() {
            assert!(
                (scalar_luma[i] - simd_luma[i]).abs() < f32::EPSILON,
                "SIMD luma mismatch at index {} for {src_w}x{src_h} -> {cols}x{rows}: scalar={} simd={}",
                i, scalar_luma[i], simd_luma[i]
            );
        }

        assert_eq!(
            scalar_color, simd_color,
            "SIMD color mismatch for {src_w}x{src_h} -> {cols}x{rows}"
        );
    }
}

/// Parity test: SIMD box-filter vs parallel (non-SIMD) box-filter.
#[test]
fn test_simd_downsample_matches_parallel() {
    let test_cases = vec![
        (640, 480, 80, 24, ColorSpace::Bt709),
        (1920, 1080, 120, 40, ColorSpace::Bt709),
        (320, 240, 40, 15, ColorSpace::Bt601),
        (1919, 1081, 101, 41, ColorSpace::Bt709),
    ];

    for (src_w, src_h, cols, rows, cs) in test_cases {
        let (y, u, v) = generate_test_yuv(src_w, src_h);

        let mut parallel_luma = vec![0.0f32; cols * rows];
        let mut parallel_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes_parallel(&y, &u, &v, src_w, src_h, cs, cols, rows, &mut parallel_luma, &mut parallel_color);

        let mut simd_luma = vec![0.0f32; cols * rows];
        let mut simd_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes_simd(&y, &u, &v, src_w, src_h, cs, cols, rows, &mut simd_luma, &mut simd_color);

        for i in 0..parallel_luma.len() {
            assert!(
                (parallel_luma[i] - simd_luma[i]).abs() < f32::EPSILON,
                "SIMD vs parallel luma mismatch at index {} for {src_w}x{src_h} -> {cols}x{rows}",
                i
            );
        }
        assert_eq!(
            parallel_color, simd_color,
            "SIMD vs parallel color mismatch for {src_w}x{src_h} -> {cols}x{rows}"
        );
    }
}

/// Stress test for race conditions.
#[test]
fn test_simd_downsample_stress() {
    let src_w = 1280;
    let src_h = 720;
    let cols = 160;
    let rows = 45;
    let (y, u, v) = generate_test_yuv(src_w, src_h);

    for _ in 0..100 {
        let mut simd_luma = vec![0.0f32; cols * rows];
        let mut simd_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes_simd(&y, &u, &v, src_w, src_h, ColorSpace::Bt709, cols, rows, &mut simd_luma, &mut simd_color);

        let mut scalar_luma = vec![0.0f32; cols * rows];
        let mut scalar_color = vec![(0u8, 0u8, 0u8); cols * rows];
        downsample_yuv_planes(&y, &u, &v, src_w, src_h, ColorSpace::Bt709, cols, rows, &mut scalar_luma, &mut scalar_color);

        assert_eq!(scalar_luma, simd_luma);
        assert_eq!(scalar_color, simd_color);
    }
}