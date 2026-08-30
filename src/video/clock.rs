/// PTS-based playback clock with drop-frame policy.
///
/// Tracks the relationship between wall-clock time and presentation timestamps
/// so each frame is rendered at the correct moment. Frames that fall too far
/// behind schedule are dropped entirely to recover lag.

use std::time::{Duration, Instant};

/// How many frame-durations a frame can be late before we skip it entirely.
const DROP_THRESHOLD_FRAMES: f64 = 2.0;

/// Default fallback frame rate when the container has no avg_frame_rate.
const DEFAULT_FPS: f64 = 24.0;

pub struct PlaybackClock {
    start_instant: Instant,
    start_pts_seconds: f64,
    avg_frame_rate: f64,
}

/// Decision for what to do with the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAction {
    /// Sleep until the target time, then render.
    SleepThenRender,
    /// Frame is slightly late but within tolerance — render immediately.
    RenderNow,
    /// Frame is way too late — skip rendering entirely to catch up.
    Drop,
}

impl PlaybackClock {
    /// Create a new clock anchored to the given PTS and wall-clock instant.
    pub fn starting_now(first_pts_seconds: f64, avg_frame_rate: f64) -> Self {
        Self {
            start_instant: Instant::now(),
            start_pts_seconds: first_pts_seconds,
            avg_frame_rate: if avg_frame_rate > 0.0 {
                avg_frame_rate
            } else {
                DEFAULT_FPS
            },
        }
    }

    /// Returns the wall-clock instant at which `pts_seconds` should be displayed.
    pub fn target_instant(&self, pts_seconds: f64) -> Instant {
        let offset = pts_seconds - self.start_pts_seconds;
        if offset >= 0.0 {
            self.start_instant + Duration::from_secs_f64(offset)
        } else {
            self.start_instant - Duration::from_secs_f64(-offset)
        }
    }

    /// Frame duration based on the average frame rate.
    pub fn frame_duration(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.avg_frame_rate)
    }

    /// Returns the average frame rate.
    pub fn avg_frame_rate(&self) -> f64 {
        self.avg_frame_rate
    }

    /// Decide what to do with a frame at the given PTS, based on the current
    /// wall-clock time.
    pub fn decide(&self, pts_seconds: f64) -> (FrameAction, Option<Duration>) {
        let target = self.target_instant(pts_seconds);
        let now = Instant::now();
        let drop_threshold =
            Duration::from_secs_f64(DROP_THRESHOLD_FRAMES / self.avg_frame_rate);

        if now > target + drop_threshold {
            (FrameAction::Drop, None)
        } else if now < target {
            (FrameAction::SleepThenRender, Some(target - now))
        } else {
            (FrameAction::RenderNow, None)
        }
    }

    /// Estimate PTS for a frame that lacks one, based on frame count.
    pub fn estimate_pts(frame_count: u64, first_pts: f64, avg_fps: f64) -> f64 {
        first_pts + frame_count as f64 / avg_fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_target_instant() {
        let clock = PlaybackClock::starting_now(1.0, 30.0);
        let target = clock.target_instant(2.0);
        let now = Instant::now();
        let diff = target.duration_since(now);
        assert!(
            diff.as_millis() < 2000,
            "target should be ~1s from now: {:?}",
            diff
        );
    }

    #[test]
    fn test_frame_duration() {
        let clock = PlaybackClock::starting_now(0.0, 30.0);
        let dur = clock.frame_duration();
        let ms = dur.as_millis();
        assert!(ms >= 30 && ms <= 40, "frame duration: {}ms", ms);
    }

    #[test]
    fn test_estimate_pts() {
        let pts = PlaybackClock::estimate_pts(60, 0.0, 30.0);
        assert!((pts - 2.0).abs() < 0.001, "60 frames at 30fps = 2s: {}", pts);
    }

    #[test]
    fn test_estimate_pts_with_offset() {
        let pts = PlaybackClock::estimate_pts(30, 5.0, 30.0);
        assert!((pts - 6.0).abs() < 0.001, "30 frames at 30fps from 5s = 6s: {}", pts);
    }

    #[test]
    fn test_avg_frame_rate_zero_fallback() {
        let clock = PlaybackClock::starting_now(0.0, 0.0);
        assert_eq!(clock.avg_frame_rate(), DEFAULT_FPS);
    }

    #[test]
    fn test_avg_frame_rate_valid() {
        let clock = PlaybackClock::starting_now(0.0, 60.0);
        assert_eq!(clock.avg_frame_rate(), 60.0);
    }

    #[test]
    fn test_decide_drop_when_very_late() {
        // A frame whose PTS was 10 seconds ago should be dropped
        let clock = PlaybackClock::starting_now(0.0, 30.0);
        let (action, _) = clock.decide(-10.0);
        assert_eq!(action, FrameAction::Drop);
    }

    #[test]
    fn test_decide_render_when_on_time() {
        // A frame whose PTS is "now" should render
        let clock = PlaybackClock::starting_now(0.0, 30.0);
        // PTS = start_pts, so target is now
        let (action, _) = clock.decide(0.0);
        // Should be either SleepThenRender (if slightly future) or RenderNow
        assert!(action != FrameAction::Drop, "on-time frame should not be dropped");
    }
}
