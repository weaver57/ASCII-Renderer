//! Differential parity tests for the Rayon-sharded Sobel and NMS kernels.
//!
//! Asserts that `build_gradient_map_parallel` / `non_max_suppress_parallel`
//! produce outputs identical to the scalar Phase 2 reference implementations
//! (`build_gradient_map` / `non_max_suppress`).

use ascii_renderer::parallel::{build_gradient_map_parallel, non_max_suppress_parallel};
use ascii_renderer::render::edge::{build_gradient_map, non_max_suppress, GradientMap};

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
            "{label}: magnitude mismatch at {i}: scalar={} parallel={}",
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
            "{label}: angle mismatch at {i}: scalar={} parallel={}",
            a.angle[i],
            b.angle[i]
        );
    }
}

/// Vertical step edge — clean Sobel response, exercises boundary columns.
#[test]
fn sobel_parallel_matches_scalar_vertical_step() {
    let w = 320;
    let h = 240;
    let luma = luma_from_fn(w, h, |x, _y| if x < 160 { 30.0 } else { 220.0 });

    let scalar = build_gradient_map(&luma, w, h);
    let parallel = build_gradient_map_parallel(&luma, w, h);
    assert_gradients_close(&scalar, &parallel, "vertical step");
}

/// Horizontal step edge.
#[test]
fn sobel_parallel_matches_scalar_horizontal_step() {
    let w = 240;
    let h = 320;
    let luma = luma_from_fn(w, h, |_x, y| if y < 160 { 40.0 } else { 200.0 });

    let scalar = build_gradient_map(&luma, w, h);
    let parallel = build_gradient_map_parallel(&luma, w, h);
    assert_gradients_close(&scalar, &parallel, "horizontal step");
}

/// Diagonal edge — exercises the 45°/135° angle bins.
#[test]
fn sobel_parallel_matches_scalar_diagonal() {
    let w = 256;
    let h = 256;
    let luma = luma_from_fn(w, h, |x, y| if x > y { 255.0 } else { 0.0 });

    let scalar = build_gradient_map(&luma, w, h);
    let parallel = build_gradient_map_parallel(&luma, w, h);
    assert_gradients_close(&scalar, &parallel, "diagonal");
}

/// Smooth gradient (no hard edges).
#[test]
fn sobel_parallel_matches_scalar_gradient() {
    let w = 300;
    let h = 200;
    let luma = luma_from_fn(w, h, |x, y| ((x + y) % 128) as f32);

    let scalar = build_gradient_map(&luma, w, h);
    let parallel = build_gradient_map_parallel(&luma, w, h);
    assert_gradients_close(&scalar, &parallel, "gradient");
}

/// Checkerboard — exercises all angle bins and high-frequency response.
#[test]
fn sobel_parallel_matches_scalar_checkerboard() {
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
    let parallel = build_gradient_map_parallel(&luma, w, h);
    assert_gradients_close(&scalar, &parallel, "checkerboard");
}

/// Odd dimensions that are not multiples of the band size, to exercise the
/// last partial band.
#[test]
fn sobel_parallel_matches_scalar_odd_dimensions() {
    let cases = [
        (1919, 1081),
        (100, 3),
        (3, 100),
        (7, 7),
        (16, 16),
        (1, 1),
        (33, 17),
        (1081, 1919),
    ];
    for (w, h) in cases {
        let luma = luma_from_fn(w, h, |x, y| ((x * 31 + y * 17) % 256) as f32);
        let scalar = build_gradient_map(&luma, w, h);
        let parallel = build_gradient_map_parallel(&luma, w, h);
        assert_gradients_close(&scalar, &parallel, &format!("odd {w}x{h}"));
    }
}

/// NMS parity over the same synthetic inputs.
#[test]
fn nms_parallel_matches_scalar() {
    let w = 320;
    let h = 240;
    let luma = luma_from_fn(w, h, |x, y| {
        // A mix of edges: vertical step + diagonal.
        let vertical = if x < 100 { 30.0 } else { 220.0 };
        let diag = if x > y { 255.0 } else { 0.0 };
        vertical + diag * 0.5
    });
    let scalar_grad = build_gradient_map(&luma, w, h);
    let parallel_grad = build_gradient_map_parallel(&luma, w, h);

    let scalar_nms = non_max_suppress(&scalar_grad);
    let parallel_nms = non_max_suppress_parallel(&parallel_grad);

    assert_eq!(scalar_nms.len(), parallel_nms.len());
    for i in 0..scalar_nms.len() {
        assert!(
            (scalar_nms[i] - parallel_nms[i]).abs() < 1e-4,
            "NMS mismatch at {i}: scalar={} parallel={}",
            scalar_nms[i],
            parallel_nms[i]
        );
    }
}

/// NMS parity across many small/odd dimensions.
#[test]
fn nms_parallel_matches_scalar_many_sizes() {
    let cases = [(256, 256), (100, 100), (33, 17), (17, 33), (64, 48), (8, 8), (200, 150)];
    for (w, h) in cases {
        let luma = luma_from_fn(w, h, |x, y| ((x * 7 + y * 13) % 256) as f32);
        let scalar_grad = build_gradient_map(&luma, w, h);
        let parallel_grad = build_gradient_map_parallel(&luma, w, h);
        let scalar_nms = non_max_suppress(&scalar_grad);
        let parallel_nms = non_max_suppress_parallel(&parallel_grad);
        assert_eq!(
            scalar_nms,
            parallel_nms,
            "NMS must match exactly at {w}x{h}"
        );
    }
}
