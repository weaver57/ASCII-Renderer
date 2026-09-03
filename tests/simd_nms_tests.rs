//! Differential parity tests for the SIMD-accelerated Non-Maximum Suppression.
//!
//! Asserts that `non_max_suppress_simd` produces outputs identical to the
//! scalar Phase 2 reference (`non_max_suppress`) and the non-SIMD parallel
//! kernel (`non_max_suppress_parallel`) across a wide range of dimensions.
//!
//! Boundary rows/columns (top, bottom, left, right) exercise the scalar
//! fallback path; widths not divisible by 8 exercise the SIMD tail.

use ascii_renderer::parallel::{
    build_gradient_map_parallel, build_gradient_map_simd, non_max_suppress_parallel,
    non_max_suppress_simd,
};
use ascii_renderer::render::edge::{build_gradient_map, non_max_suppress};

/// Constructs a deterministic synthetic luma map.
fn luma_from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let mut luma = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            luma.push(f(x, y));
        }
    }
    luma
}

/// A mix of edges to produce rich gradient angle coverage.
fn mixed_luma(w: usize, h: usize) -> Vec<f32> {
    luma_from_fn(w, h, |x, y| {
        let vertical: f32 = if x < w / 2 { 30.0 } else { 220.0 };
        let horizontal: f32 = if y < h / 2 { 40.0 } else { 200.0 };
        let diag: f32 = if x > y { 255.0 } else { 0.0 };
        (vertical + horizontal + diag * 0.5).min(255.0f32)
    })
}

/// Asserts two NMS outputs are identical (bit-exact — NMS is a discrete keep/zero
/// decision, so any divergence is a bug, not float noise).
fn assert_nms_equal(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    for i in 0..a.len() {
        assert_eq!(
            a[i], b[i],
            "{label}: mismatch at {i}: scalar={} simd={}",
            a[i], b[i]
        );
    }
}

/// SIMD NMS must match the scalar reference across many shapes, including
/// boundary rows/columns and non-multiple-of-8 widths.
#[test]
fn simd_nms_matches_scalar() {
    let cases = [
        (640, 480),
        (320, 240),
        (1920, 1080),
        (1919, 1081),
        (100, 100),
        (33, 17),
        (17, 33),
        (8, 8),
        (16, 16),
        (1, 1),
        (2, 2),
        (3, 3),
        (9, 9),
        (10, 10),
        (15, 15),
        (200, 150),
        (7, 100),
        (100, 7),
    ];
    for (w, h) in cases {
        let luma = mixed_luma(w, h);
        let scalar_grad = build_gradient_map(&luma, w, h);
        let simd_grad = build_gradient_map_simd(&luma, w, h);
        let scalar_nms = non_max_suppress(&scalar_grad);
        let simd_nms = non_max_suppress_simd(&simd_grad);
        assert_nms_equal(&scalar_nms, &simd_nms, &format!("scalar vs simd {w}x{h}"));
    }
}

/// SIMD NMS must match the non-SIMD parallel kernel.
#[test]
fn simd_nms_matches_parallel() {
    let cases = [
        (1920, 1080),
        (1919, 1081),
        (640, 480),
        (320, 240),
        (100, 100),
        (33, 17),
        (17, 33),
        (1, 1),
    ];
    for (w, h) in cases {
        let luma = mixed_luma(w, h);
        let par_grad = build_gradient_map_parallel(&luma, w, h);
        let simd_grad = build_gradient_map_simd(&luma, w, h);
        let par_nms = non_max_suppress_parallel(&par_grad);
        let simd_nms = non_max_suppress_simd(&simd_grad);
        assert_nms_equal(&par_nms, &simd_nms, &format!("parallel vs simd {w}x{h}"));
    }
}

/// Single clean vertical step: the two-peak suppression behavior must match.
#[test]
fn simd_nms_vertical_step() {
    let w = 320;
    let h = 240;
    let luma = luma_from_fn(w, h, |x, _y| if x < 160 { 30.0 } else { 220.0 });
    let scalar_grad = build_gradient_map(&luma, w, h);
    let simd_grad = build_gradient_map_simd(&luma, w, h);
    assert_nms_equal(
        &non_max_suppress(&scalar_grad),
        &non_max_suppress_simd(&simd_grad),
        "vertical step",
    );
}

/// Stress: many repetitions to catch nondeterminism / race conditions in the
/// Rayon sharding of the SIMD NMS kernel.
#[test]
fn simd_nms_stress_determinism() {
    let w = 1280;
    let h = 720;
    let luma = mixed_luma(w, h);
    let scalar_grad = build_gradient_map(&luma, w, h);
    let simd_grad = build_gradient_map_simd(&luma, w, h);
    let reference = non_max_suppress(&scalar_grad);

    for _ in 0..50 {
        let simd_nms = non_max_suppress_simd(&simd_grad);
        assert_nms_equal(&reference, &simd_nms, "stress");
    }
}

/// Very wide-but-short and tall-but-narrow shapes, stressing band boundaries.
#[test]
fn simd_nms_extreme_aspect() {
    let cases = [(4096, 3), (3, 4096), (8192, 16), (16, 8192), (17, 1), (1, 17)];
    for (w, h) in cases {
        let luma = mixed_luma(w, h);
        let scalar_grad = build_gradient_map(&luma, w, h);
        let simd_grad = build_gradient_map_simd(&luma, w, h);
        assert_nms_equal(
            &non_max_suppress(&scalar_grad),
            &non_max_suppress_simd(&simd_grad),
            &format!("extreme {w}x{h}"),
        );
    }
}
