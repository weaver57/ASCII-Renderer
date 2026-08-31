use std::io::{self, Stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, size, Clear, ClearType,
        EnterAlternateScreen, LeaveAlternateScreen,
    },
};

use crate::caps::ColorSupport;
use crate::dither::ordered_dither_quantize;
use crate::diff::DirtyRun;
use crate::palette::Palette;
use crate::render::grid::{CharGrid, Rgb};

static PANIC_HOOK_SET: AtomicBool = AtomicBool::new(false);

/// RAII guard to ensure the terminal is restored to its original state even on panic or early exit.
pub struct TerminalGuard {
    stdout: Stdout,
    active: Arc<AtomicBool>,
}

impl TerminalGuard {
    pub fn init() -> io::Result<Self> {
        let active = Arc::new(AtomicBool::new(true));

        if !PANIC_HOOK_SET.swap(true, Ordering::SeqCst) {
            let default_panic_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |panic_info| {
                let mut stdout = io::stdout();
                let _ = execute!(stdout, Show, LeaveAlternateScreen);
                let _ = disable_raw_mode();
                let _ = stdout.flush();
                default_panic_hook(panic_info);
            }));
        }

        let mut stdout = io::stdout();
        enable_raw_mode()?;
        execute!(
            stdout,
            EnterAlternateScreen,
            Hide,
            Clear(ClearType::All),
            MoveTo(0, 0)
        )?;
        stdout.flush()?;

        Ok(Self { stdout, active })
    }

    /// Returns the current terminal dimensions as (columns, rows).
    pub fn get_size() -> io::Result<(u16, u16)> {
        size()
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
        // The ONE SGR reset for the whole program run (D1): clears the
        // persistent color state back to the terminal's default, so the
        // alternate-screen teardown doesn't leave the user's actual terminal
        // with a stale foreground color.
        let _ = write!(self.stdout, "\x1b[0m");
        let _ = execute!(
            self.stdout,
            Show,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
        let _ = self.stdout.flush();
    }
}

// ── Phase 4: resolved color + emission ──────────────────────────────────────

/// A color as it will be emitted to the terminal. Tracked persistently
/// (D1) so that identical successive resolved values suppress the
/// redundant escape code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedColor {
    /// Full 24-bit truecolor: `\x1b[38;2;R;G;Bm`
    Rgb(u8, u8, u8),
    /// Indexed 256- or 16-color: `\x1b[38;5;Nm`
    Indexed(u8),
    /// No color (Monochrome user mode, NoColor terminal support, or
    /// 16-color fallback rendered via the palette — see §4.7).
    None,
}

/// Global output state, persistent across the entire playback loop (D1).
pub struct OutputState {
    pub color_support: ColorSupport,
    pub sync_output_supported: bool,
    /// The color most recently emitted. Starts `None`, and is *never* reset
    /// between frames, rows, or dirty runs — only mutated as cells are
    /// emitted.
    pub last_resolved: Option<ResolvedColor>,
}

impl OutputState {
    pub fn new(color_support: ColorSupport, sync_output_supported: bool) -> Self {
        Self {
            color_support,
            sync_output_supported,
            last_resolved: None,
        }
    }
}

/// Resolves a cell's true content RGB to a `ResolvedColor` based on the
/// terminal's capabilities and the cell's grid position (used for dithering).
///
/// Monochrome user mode is handled *outside* this function by the caller:
/// it can call this with `ResolvedColor::None` directly, or this function
/// falls through to the normal `ColorSupport` resolution. We keep both paths
/// explicit so the test can verify them independently.
#[inline]
pub fn resolve_color(color: Rgb, support: ColorSupport, x: usize, y: usize, palette: &Palette) -> ResolvedColor {
    match support {
        ColorSupport::TrueColor => ResolvedColor::Rgb(color.r, color.g, color.b),
        ColorSupport::Palette256 => {
            let idx = ordered_dither_quantize(color, x, y, palette);
            ResolvedColor::Indexed(idx)
        }
        ColorSupport::Basic16 => {
            let idx = ordered_dither_quantize(color, x, y, palette);
            ResolvedColor::Indexed(idx)
        }
        ColorSupport::NoColor => ResolvedColor::None,
    }
}

/// Appends the ANSI escape code for `resolved` to `out` if it differs from
/// `last`, updating `last`. This is the single point of emission for all
/// SGR color sequences; it implements the color-run compression from D1.
#[inline]
fn emit_color_escape(out: &mut Vec<u8>, last: &mut Option<ResolvedColor>, resolved: ResolvedColor) {
    if *last == Some(resolved) {
        return; // color unchanged — suppress escape (D1)
    }
    match resolved {
        ResolvedColor::Rgb(r, g, b) => {
            let _ = write!(out, "\x1b[38;2;{};{};{}m", r, g, b);
        }
        ResolvedColor::Indexed(idx) => {
            let _ = write!(out, "\x1b[38;5;{}m", idx);
        }
        ResolvedColor::None => {}
    }
    *last = Some(resolved);
}

/// Emit only the dirty runs to `out`, using cursor jumps for positioning
/// (§4.4). This is the *diff-based* emission path that replaces Phase 3's
/// full-frame redraw. `last_resolved` persists across the whole call
/// sequence (D1).
pub fn emit_cells(
    grid: &CharGrid,
    runs: &[DirtyRun],
    state: &mut OutputState,
    out: &mut Vec<u8>,
    palette: &Palette,
    mono: bool, // true when user color mode is Monochrome
) {
    let mut char_buf = [0u8; 4];

    for run in runs {
        // Cursor jump to run start — ANSI uses 1-indexed positions.
        let _ = write!(out, "\x1b[{};{}H", run.row + 1, run.col_start + 1);

        for col in run.col_start..run.col_end {
            let cell = grid.cell(run.row, col);
            if mono || state.color_support == ColorSupport::NoColor {
                // Monochrome user mode or NoColor support: no color escapes at all.
                // (cursor jumps are still emitted — the terminal is color-capable;
                // only SGR color is suppressed in Monochrome mode.)
            } else {
                let resolved = resolve_color(cell.color, state.color_support, col, run.row, palette);
                emit_color_escape(out, &mut state.last_resolved, resolved);
            }
            let encoded = cell.glyph.encode_utf8(&mut char_buf);
            out.extend_from_slice(encoded.as_bytes());
        }
    }
}

/// Wrap the already-built output bytes in synchronized-update markers and
/// flush to stdout (§4.10).
pub fn write_frame(out: &[u8], sync_supported: bool, writer: &mut impl Write) -> io::Result<()> {
    if sync_supported {
        writer.write_all(b"\x1b[?2026h")?;
    }
    writer.write_all(out)?;
    if sync_supported {
        writer.write_all(b"\x1b[?2026l")?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{compute_dirty_runs, DirtyRun};
    use crate::palette::Palette;
    use crate::render::grid::{CharCell, CharGrid, Rgb, SENTINEL_CELL};

    fn cell(glyph: char, r: u8, g: u8, b: u8) -> CharCell {
        CharCell {
            glyph,
            color: Rgb::new(r, g, b),
        }
    }

    // ── Milestone 4.10: golden byte-sequence regression ───────────────────
    //
    // Pins the EXACT escape bytes emitted for a synthetic 2-frame sequence,
    // guarding diffing + color-run compression + color resolution working
    // together end-to-end (the test most likely to catch a real integration
    // bug that narrower unit tests would each individually miss).

    #[test]
    fn golden_truecolor_incremental_frame_bytes() {
        let mut displayed = CharGrid::new(4, 1, SENTINEL_CELL);
        let mut current = CharGrid::new(4, 1, SENTINEL_CELL);
        for c in 0..4 {
            displayed.set(c, cell('A', 10, 10, 10));
        }
        // Frame 2: only column 2 differs.
        for c in 0..4 {
            current.set(c, if c == 2 { cell('B', 20, 20, 20) } else { cell('A', 10, 10, 10) });
        }
        let runs = compute_dirty_runs(&current, &displayed);
        assert_eq!(runs.len(), 1, "only column 2 differs");
        assert_eq!(runs[0], DirtyRun { row: 0, col_start: 2, col_end: 3 });

        let palette = Palette::xterm256();
        let mut state = OutputState::new(ColorSupport::TrueColor, false);
        let mut out = Vec::new();
        emit_cells(&current, &runs, &mut state, &mut out, &palette, false);
        assert_eq!(
            out,
            b"\x1b[1;3H\x1b[38;2;20;20;20mB".to_vec(),
            "jump to (row 1, col 3), set the new color, draw 'B'"
        );
    }

    #[test]
    fn d1_color_state_persists_across_runs() {
        // Two far-apart dirty runs sharing one color: the second run must not
        // re-emit the escape (persistent color state — D1).
        let mut displayed = CharGrid::new(8, 1, SENTINEL_CELL);
        let mut current = CharGrid::new(8, 1, SENTINEL_CELL);
        for c in 0..8 {
            displayed.set(c, cell('A', 10, 10, 10));
        }
        // Dirty at col 0 and col 7; gap 6 > MERGE_GAP_THRESHOLD(4) → two runs.
        for c in 0..8 {
            current.set(c, if c == 0 || c == 7 { cell('B', 10, 10, 10) } else { cell('A', 10, 10, 10) });
        }
        let runs = compute_dirty_runs(&current, &displayed);
        assert_eq!(runs.len(), 2);

        let palette = Palette::xterm256();
        let mut state = OutputState::new(ColorSupport::TrueColor, false);
        let mut out = Vec::new();
        emit_cells(&current, &runs, &mut state, &mut out, &palette, false);
        assert_eq!(
            out,
            b"\x1b[1;1H\x1b[38;2;10;10;10mB\x1b[1;8HB".to_vec(),
            "same color across both runs → single escape, second suppressed"
        );
    }

    #[test]
    fn d1_color_state_persists_between_emit_calls() {
        // The same OutputState fed to a second emit_cells call (the next
        // frame) suppresses the redundant escape when the color is unchanged.
        let palette = Palette::xterm256();
        let mut state = OutputState::new(ColorSupport::TrueColor, false);

        let mut displayed = CharGrid::new(4, 1, SENTINEL_CELL);
        let mut current = CharGrid::new(4, 1, SENTINEL_CELL);
        for c in 0..4 {
            displayed.set(c, cell('A', 1, 1, 1));
        }
        for c in 0..4 {
            current.set(c, cell('B', 30, 30, 30));
        }
        // Frame N: everything changes → full frame, escape for (30,30,30).
        let runs = compute_dirty_runs(&current, &displayed);
        let mut out = Vec::new();
        emit_cells(&current, &runs, &mut state, &mut out, &palette, false);
        assert!(out.starts_with(b"\x1b[1;1H\x1b[38;2;30;30;30m"));

        // Frame N+1: same color, but 'B'→'C'. Cursor jumps again, but the
        // unchanged color (30,30,30) must suppress the escape.
        let mut frame_n_plus_1 = CharGrid::new(4, 1, SENTINEL_CELL);
        for c in 0..4 {
            frame_n_plus_1.set(c, cell('C', 30, 30, 30));
        }
        let runs2 = compute_dirty_runs(&frame_n_plus_1, &current);
        let mut out2 = Vec::new();
        emit_cells(&frame_n_plus_1, &runs2, &mut state, &mut out2, &palette, false);
        // Full frame redrawn (jump + all 'C' glyphs) but the unchanged color
        // (30,30,30) suppresses the escape entirely — D1 across frame calls.
        assert_eq!(
            out2,
            b"\x1b[1;1HCCCC".to_vec(),
            "no redundant escape for an unchanged color across frames"
        );
    }

    #[test]
    fn write_frame_wraps_sync_markers_when_supported() {
        let payload = b"abc";
        let mut sink = Vec::new();
        write_frame(payload, true, &mut sink).unwrap();
        assert_eq!(sink, b"\x1b[?2026habc\x1b[?2026l");
        let mut sink2 = Vec::new();
        write_frame(payload, false, &mut sink2).unwrap();
        assert_eq!(sink2, b"abc");
    }
}
