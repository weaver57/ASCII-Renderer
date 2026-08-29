use crate::render::edge::{EdgeCellInfo, circular_lerp_deg};
use std::collections::VecDeque;

/// Smooths per-cell edge state across consecutive frames using an exponential
/// moving average on magnitude and circular interpolation on orientation.
///
/// Built and unit-tested in Phase 2; not yet wired into the video loop
/// (Phase 3 will call `update()` on each decoded frame).
pub struct TemporalEdgeSmoother {
    prev: Option<Vec<Option<EdgeCellInfo>>>,
    alpha: f32,
}

impl TemporalEdgeSmoother {
    /// Creates a new smoother with the given smoothing factor.
    ///
    /// `alpha` controls responsiveness: higher = snappier (less smoothing),
    /// lower = smoother/more laggy. Typical range 0.2–0.5.
    pub fn new(alpha: f32) -> Self {
        Self {
            prev: None,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Updates the smoother with the current frame's per-cell edge data and
    /// returns the temporally-smoothed version.
    ///
    /// Blending rules:
    /// * `Some → Some`: lerp magnitude, circular_lerp orientation.
    /// * `Some → None`: edge just appeared — take current at full strength
    ///   (no artificial fade-in).
    /// * `None → Some`: edge just disappeared — return `None` immediately
    ///   (no smoothing away a real change).
    /// * `None → None`: stays `None`.
    ///
    /// These rules stabilize *noisy flicker in orientation/magnitude of a
    /// persisting edge*, while leaving genuine scene changes (appear/disappear)
    /// sharp and responsive.
    pub fn update(&mut self, current: Vec<Option<EdgeCellInfo>>) -> Vec<Option<EdgeCellInfo>> {
        let smoothed = match &self.prev {
            None => current.clone(),
            Some(prev) => {
                let n = prev.len();
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(blend_cell(&current[i], &prev[i], self.alpha));
                }
                out
            }
        };
        self.prev = Some(smoothed.clone());
        smoothed
    }
}

/// Blends two cell states according to the §4.8 rules.
fn blend_cell(
    cur: &Option<EdgeCellInfo>,
    prev: &Option<EdgeCellInfo>,
    alpha: f32,
) -> Option<EdgeCellInfo> {
    match (cur, prev) {
        (Some(c), Some(p)) => Some(EdgeCellInfo {
            magnitude: p.magnitude * (1.0 - alpha) + c.magnitude * alpha,
            orientation_deg: circular_lerp_deg(p.orientation_deg, c.orientation_deg, alpha, 180.0),
        }),
        (Some(c), None) => Some(*c),      // appeared: no history to blend with
        (None, Some(_)) => None,          // disappeared: don't smooth away a real change
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::edge::EdgeCellInfo;

    #[test]
    fn temporal_smoother_magnitude_blends_linearly() {
        let mut smoother = TemporalEdgeSmoother::new(0.5);
        let cur = vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 0.0,
        })];
        let first = smoother.update(cur.clone());
        let second = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 200.0,
            orientation_deg: 0.0,
        })]);
        // First frame = cur. Second = 0.5 * 100 + 0.5 * 200 = 150.
        assert!((second[0].unwrap().magnitude - 150.0).abs() < 1e-5);
    }

    #[test]
    fn temporal_smoother_orientation_circular_lerp_short_way() {
        let mut smoother = TemporalEdgeSmoother::new(0.5);
        // 179° and 2° are both nearly-horizontal (short way is 179→180→2 = 3° span),
        // not the long way (179→0→2 = 181° span).
        let cur = vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 179.0,
        })];
        let first = smoother.update(cur);
        let second = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 2.0,
        })]);
        // Circular lerp at t=0.5 should land near the wraparound (~0.5° or ~179.5°),
        // NOT near 90° (naive linear lerp).
        let o = second[0].unwrap().orientation_deg;
        assert!(
            (o - 0.5).abs() < 5.0 || (o - 179.5).abs() < 5.0,
            "circular lerp should take the short way: got {}",
            o
        );
        assert!(
            (o - 90.0).abs() > 10.0,
            "must NOT be naive linear lerp (90°): got {}",
            o
        );
    }

    #[test]
    fn temporal_smoother_appeared_edge_no_fade_in() {
        let mut smoother = TemporalEdgeSmoother::new(0.5);
        // Frame 1: no edge
        let first = smoother.update(vec![None]);
        assert!(first[0].is_none());
        // Frame 2: edge appears
        let second = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 45.0,
        })]);
        // Must take the new value at full strength, no blending with None.
        assert_eq!(second[0].unwrap().magnitude, 100.0);
        assert_eq!(second[0].unwrap().orientation_deg, 45.0);
    }

    #[test]
    fn temporal_smoother_disappeared_edge_no_fade_out() {
        let mut smoother = TemporalEdgeSmoother::new(0.5);
        // Frame 1: edge present
        let first = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 45.0,
        })]);
        assert_eq!(first[0].unwrap().magnitude, 100.0);
        // Frame 2: edge gone
        let second = smoother.update(vec![None]);
        // Must return None immediately — don't smooth away a real disappearance.
        assert!(second[0].is_none());
    }

    #[test]
    fn temporal_smoother_alpha_zero_is_identity() {
        let mut smoother = TemporalEdgeSmoother::new(0.0);
        let cur = vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 30.0,
        })];
        let first = smoother.update(cur);
        let second = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 200.0,
            orientation_deg: 90.0,
        })]);
        // alpha=0 → keep previous exactly (within float epsilon).
        assert!((second[0].unwrap().magnitude - 100.0).abs() < 1e-5);
        assert!((second[0].unwrap().orientation_deg - 30.0).abs() < 1e-5);
    }

    #[test]
    fn temporal_smoother_alpha_one_is_current() {
        let mut smoother = TemporalEdgeSmoother::new(1.0);
        let first = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 100.0,
            orientation_deg: 30.0,
        })]);
        let second = smoother.update(vec![Some(EdgeCellInfo {
            magnitude: 200.0,
            orientation_deg: 90.0,
        })]);
        // alpha=1 → take current exactly.
        assert_eq!(second[0].unwrap().magnitude, 200.0);
        assert_eq!(second[0].unwrap().orientation_deg, 90.0);
    }

    #[test]
    fn circular_lerp_deg_midpoint_short_way() {
        // 179° and 2°: the short arc is 179→180(≡0)→2 = 3° total; midpoint ~0.5°.
        let mid = circular_lerp_deg(179.0, 2.0, 0.5, 180.0);
        assert!(
            (mid - 0.5).abs() < 1.0 || (mid - 179.5).abs() < 1.0,
            "short-way midpoint of 179/2 is ~0.5° or ~179.5°, got {}",
            mid
        );
        assert!((mid - 90.5).abs() > 10.0, "must not be naive linear midpoint");
    }

    #[test]
    fn circular_lerp_deg_t_zero_returns_a() {
        assert!((circular_lerp_deg(42.0, 120.0, 0.0, 180.0) - 42.0).abs() < 1e-5);
    }

    #[test]
    fn circular_lerp_deg_t_one_returns_b() {
        assert!((circular_lerp_deg(42.0, 120.0, 1.0, 180.0) - 120.0).abs() < 1e-5);
    }
}