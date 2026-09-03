//! Differential parity tests for the SIMD-accelerated Sobel gradient kernel.
//!
//! Asserts that `build_gradient_map_simd` produces outputs identical to the
//! scalar Phase 2 reference (`build_gradient_map`) and the non-SIMD parallel
//! kernel (`build_gradient_map_parallel`) across a wide range of dimensions,
//! including widths that are not multiples of the 8-lane SIMD width and the
//! top/bottom/left/right boundary rows & columns (which exercise the clamping
//! path in `load_sobel_3x10`).

use ascii_renderer::parallel::{build_gradient_map_parallel, build_gradient_map_simd};
use ascii_renderer::render::edge::{build_gradient_map, GradientMap};

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

/// Asserts two gradient maps are identical within float epsilon.
fn assert_gradients_close(a: &GradientMap, b: &GradientMap, label: &str) {
    assert_eq!(a.width, b.width, "{label}: width");
    assert_eq!(a.height, b.height, "{label}: height");
    assert_eq!(a.magnitude.len(), b.magnitude.len(), "{label}: mag len");
    assert_eq!(a.angle.len(), b.angle.len(), "{label}: angle len");
    for i in 0..a.magnitude.len() {
        assert!(
            (a.magnitude[i] - b.magnitude[i]).abs() < 1e-4,
            "{label}: magnitude mismatch at {i}: scalar={} simd={}",
            a.magnitude[i],
            b.magnitude[i]
        );
    }
    for i in 0..a.angle.len() {
        // Compare circularly (angles are periodic with period 2π).
        let mut diff = (a.angle[i] - b.angle[i]).abs() % std::f32::consts::TAU;
        if diff > std::f32::consts::PI {
            diff = std::f32::consts::TAU - diff;
        }
        assert!(
            diff < 1e-4,
            "{label}: angle mismatch at {i}: scalar={} simd={}",
            a.angle[i],
            b.angle[i]
        );
    }
}

/// Vertical step edge — exercises boundary columns and the SIMD/scalar boundary.
#[test]
fn simd_sobel_matches_scalar_vertical_step() {
    let w = 320;
    let h = 240;
    let luma = luma_from_fn(w, h, |x, _y| if x < 160 { 30.0 } else { 220.0 });

    let scalar = build_gradient_map(&luma, w, h);
    let simd = build_gradient_map_simd(&luma, w, h);
    assert_gradients_close(&scalar, &simd, "vertical step");
}

/// Horizontal step edge — exercises boundary rows.
#[test]
fn simd_sobel_matches_scalar_horizontal_step() {
    let w = 240;
    let h = 320;
    let luma = luma_from_fn(w, h, |_x, y| if y < 160 { 40.0 } else { 200.0 });

    let scalar = build_gradient_map(&luma, w, h);
    let simd = build_gradient_map_simd(&luma, w, h);
    assert_gradients_close(&scalar, &simd, "horizontal step");
}

/// Diagonal edge — exercises all angle bins.
#[test]
fn simd_sobel_matches_scalar_diagonal() {
    let w = 256;
    let h = 256;
    let luma = luma_from_fn(w, h, |x, y| if x > y { 255.0 } else { 0.0 });

    let scalar = build_gradient_map(&luma, w, h);
    let simd = build_gradient_map_simd(&luma, w, h);
    assert_gradients_close(&scalar, &simd, "diagonal");
}

/// Checkerboard — exercises all angle bins and high-frequency response.
#[test]
fn simd_sobel_matches_scalar_checkerboard() {
    let w = 300;
    let h = 180;
    let luma = luma_from_fn(w, h, |x, y| {
        if (x / 4 + y / 4) % 2 == 0 {
            255.0
        } else {
            0.0
        }
    });

    let scalar = build_gradient_map(&luma, w, h);
    let simd = build_gradient_map_simd(&luma, w, h);
    assert_gradients_close(&scalar, &simd, "checkerboard");
}

/// Odd dimensions not multiples of the 8-lane SIMD width, to exercise the scalar
/// tail, plus extreme boundary sizes.
#[test]
fn simd_sobel_matches_scalar_odd_dimensions() {
    let cases = [
        (1919, 1081),
        (100, 3),
        (3, 100),
        (7, 7),
        (16, 16),
        (1, 1),
        (33, 17),
        (1081, 1919),
        (2, 2),
        (9, 9),
        (10, 10),
        (15, 15),
        (17, 33),
        (8, 8),
        (201, 101),
        (640, 480),
    ];
    for (w, h) in cases {
        let luma = luma_from_fn(w, h, |x, y| ((x * 31 + y * 17) % 256) as f32);
        let scalar = build_gradient_map(&luma, w, h);
        let simd = build_gradient_map_simd(&luma, w, h);
        assert_gradients_close(&scalar, &simd, &format!("odd {w}x{h}"));
    }
}

/// The SIMD path must also match the non-SIMD Rayon parallel kernel.
#[test]
fn simd_sobel_matches_parallel() {
    let cases = [
        (1920, 1080),
        (1919, 1081),
        (640, 480),
        (320, 240),
        (100, 100),
        (33, 17),
        (1, 1),
    ];
    for (w, h) in cases {
        let luma = luma_from_fn(w, h, |x, y| {
            let vertical = if x < w / 2 { 30.0 } else { 220.0 };
            let diag = if x > y { 255.0 } else { 0.0 };
            vertical + diag * 0.5
        });
        let parallel = build_gradient_map_parallel(&luma, w, h);
        let simd = build_gradient_map_simd(&luma, w, h);
        assert_gradients_close(&parallel, &simd, &format!("parallel vs simd {w}x{h}"));
    }
}

/// Stress: many repetitions to catch nondeterminism / race conditions in the
/// Rayon sharding of the SIMD kernel.
#[test]
fn simd_sobel_stress_determinism() {
    let w = 1280;
    let h = 720;
    let luma = luma_from_fn(w, h, |x, y| ((x * 31 + y * 17) % 256) as f32);
    let reference = build_gradient_map(&luma, w, h);

    for _ in 0..50 {
        let simd = build_gradient_map_simd(&luma, w, h);
        assert_gradients_close(&reference, &simd, "stress");
    }
}
