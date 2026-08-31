//! Frame-to-frame diffing: dirty-run detection with gap-merging (§4.3), and
//! the double-buffered `DoubleGrid` (§4.5) that never allocates per-frame.
//!
//! `CharCell` equality is plain field equality (`glyph == glyph && color ==
//! color`) — cheap and glyph-set-agnostic, so Phase 7's half-block/Braille
//! modes work with this unchanged.

use crate::render::grid::CharGrid;

/// A maximal contiguous span of dirty cells within a single row: columns
/// `[col_start, col_end)` in `row` differ between the current and displayed
/// frames. (Inclusive start, exclusive end — the Rust idiom.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyRun {
    pub row: usize,
    pub col_start: usize,
    pub col_end: usize,
}

/// After two dirty runs within `MERGE_GAP_THRESHOLD` cells are merged into one
/// run (which draws over the clean gap cells between them), the gap's cost
/// drops to a handful of glyph bytes at most — far cheaper than the ~9-byte
/// cursor-jump escape that would otherwise separate them.
pub const MERGE_GAP_THRESHOLD: usize = 4;

/// Computes dirty runs between two grids, then merges adjacent runs whose gap
/// is within `MERGE_GAP_THRESHOLD` (§4.3, D3). Row-ordered (top to bottom,
/// left to right within each row).
pub fn compute_dirty_runs(current: &CharGrid, displayed: &CharGrid) -> Vec<DirtyRun> {
    debug_assert_eq!(current.cols(), displayed.cols());
    debug_assert_eq!(current.rows(), displayed.rows());
    let cols = current.cols();
    let rows = current.rows();
    let mut runs: Vec<DirtyRun> = Vec::new();

    for row in 0..rows {
        let mut merged: Vec<DirtyRun> = Vec::new();
        let mut raw_start: Option<usize> = None;

        for col in 0..=cols {
            let dirty = col < cols && current.cell(row, col) != displayed.cell(row, col);
            match (dirty, &mut raw_start) {
                (true, _) => {
                    if raw_start.is_none() {
                        raw_start = Some(col);
                    }
                }
                (false, Some(start)) => {
                    let raw_run = DirtyRun {
                        row,
                        col_start: *start,
                        col_end: col,
                    };
                    raw_start = None;
                    merge_run(&mut merged, raw_run);
                }
                (false, None) => {}
            }
        }
        runs.extend(merged);
    }
    runs
}

/// Merge `run` into `merged` — absorb it into the last run if the gap is
/// within `MERGE_GAP_THRESHOLD`, otherwise push a new run.
fn merge_run(merged: &mut Vec<DirtyRun>, run: DirtyRun) {
    match merged.last_mut() {
        Some(last)
            if run.col_start > last.col_end
                && (run.col_start - last.col_end) <= MERGE_GAP_THRESHOLD =>
        {
            last.col_end = run.col_end;
        }
        _ => {
            merged.push(run);
        }
    }
}

/// Double-buffered `CharGrid`: write into one, diff against the other, then
/// `present()` to swap — no allocation or clone in the per-frame hot path.
#[derive(Debug)]
pub struct DoubleGrid {
    buffers: [CharGrid; 2],
    displayed: usize,
}

impl DoubleGrid {
    /// Create both grids filled with `SENTINEL_CELL` so that the very first
    /// real frame naturally diffs as a full redraw with no special-case branch.
    pub fn new(cols: usize, rows: usize) -> Self {
        use crate::render::grid::SENTINEL_CELL;
        Self {
            buffers: [
                CharGrid::new(cols, rows, SENTINEL_CELL),
                CharGrid::new(cols, rows, SENTINEL_CELL),
            ],
            displayed: 0,
        }
    }

    /// The buffer that should be written to this frame (frame N−2's old data).
    #[inline]
    pub fn write_buffer(&mut self) -> &mut CharGrid {
        &mut self.buffers[1 - self.displayed]
    }

    /// The buffer that was last displayed (frame N−1's data, needed for diffing).
    #[inline]
    pub fn displayed_buffer(&self) -> &CharGrid {
        &self.buffers[self.displayed]
    }

    /// Flip the display index — make the just-written buffer the new
    /// displayed buffer for the next frame's diff.
    #[inline]
    pub fn present(&mut self) {
        self.displayed = 1 - self.displayed;
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.buffers[0].cols()
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.buffers[0].rows()
    }

    /// Diffs the two internal buffers — the freshly-written buffer against the
    /// last-displayed one — returning the dirty runs. This exists as a method
    /// (rather than having callers pass both accessors to the free function)
    /// because external code cannot hold the mutable `write_buffer` borrow and
    /// the shared `displayed_buffer` borrow alive simultaneously; here the two
    /// `&self.buffers[..]` borrows are disjoint indices and never conflict.
    #[inline]
    pub fn dirty_runs(&self) -> Vec<DirtyRun> {
        let (write_idx, displayed_idx) = (1 - self.displayed, self.displayed);
        compute_dirty_runs(&self.buffers[write_idx], &self.buffers[displayed_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::grid::{CharCell, Rgb, SENTINEL_CELL};

    fn make_cell(glyph: char, r: u8, g: u8, b: u8) -> CharCell {
        CharCell {
            glyph,
            color: Rgb::new(r, g, b),
        }
    }

    fn fill_grid(grid: &mut CharGrid, glyph: char, r: u8, g: u8, b: u8) {
        let cell = make_cell(glyph, r, g, b);
        for idx in 0..grid.len() {
            grid.set(idx, cell);
        }
    }

    // ── Dirty run tests ────────────────────────────────────────────────────

    #[test]
    fn identical_grids_produce_no_runs() {
        let mut g = CharGrid::new(4, 3, SENTINEL_CELL);
        fill_grid(&mut g, 'A', 100, 100, 100);
        let runs = compute_dirty_runs(&g, &g);
        assert!(runs.is_empty(), "identical grids → no runs");
    }

    #[test]
    fn single_dirty_cell_produces_one_run() {
        let mut a = CharGrid::new(4, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(4, 1, SENTINEL_CELL);
        fill_grid(&mut a, 'A', 10, 10, 10);
        fill_grid(&mut b, 'A', 10, 10, 10);
        b.set(2, make_cell('B', 20, 20, 20));
        let runs = compute_dirty_runs(&a, &b);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], DirtyRun { row: 0, col_start: 2, col_end: 3 });
    }

    #[test]
    fn adjacent_runs_merge_within_threshold() {
        let mut a = CharGrid::new(10, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(10, 1, SENTINEL_CELL);
        fill_grid(&mut a, 'A', 10, 10, 10);
        fill_grid(&mut b, 'A', 10, 10, 10);
        b.set(1, make_cell('B', 20, 20, 20));
        b.set(4, make_cell('B', 20, 20, 20));
        // gap = 4 - 2 = 2 ≤ MERGE_GAP_THRESHOLD(4) → merge
        let runs = compute_dirty_runs(&a, &b);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], DirtyRun { row: 0, col_start: 1, col_end: 5 });
    }

    #[test]
    fn gap_beyond_threshold_stays_separate() {
        let mut a = CharGrid::new(12, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(12, 1, SENTINEL_CELL);
        fill_grid(&mut a, 'A', 10, 10, 10);
        fill_grid(&mut b, 'A', 10, 10, 10);
        b.set(1, make_cell('B', 20, 20, 20));
        b.set(7, make_cell('B', 20, 20, 20));
        // gap = 7 - 2 = 5 > MERGE_GAP_THRESHOLD(4) → separate
        let runs = compute_dirty_runs(&a, &b);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].col_start, 1);
        assert_eq!(runs[1].col_start, 7);
    }

    #[test]
    fn exactly_at_threshold_merges() {
        let mut a = CharGrid::new(10, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(10, 1, SENTINEL_CELL);
        fill_grid(&mut a, 'A', 10, 10, 10);
        fill_grid(&mut b, 'A', 10, 10, 10);
        b.set(1, make_cell('B', 20, 20, 20));
        b.set(6, make_cell('B', 20, 20, 20));
        // gap = 6 - 2 = 4 == MERGE_GAP_THRESHOLD(4) → merge
        let runs = compute_dirty_runs(&a, &b);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], DirtyRun { row: 0, col_start: 1, col_end: 7 });
    }

    #[test]
    fn multirun_across_rows() {
        let mut a = CharGrid::new(4, 2, SENTINEL_CELL);
        let mut b = CharGrid::new(4, 2, SENTINEL_CELL);
        fill_grid(&mut a, 'A', 10, 10, 10);
        fill_grid(&mut b, 'A', 10, 10, 10);
        b.set(1, make_cell('B', 20, 20, 20)); // row 0, col 1
        b.set(5, make_cell('B', 20, 20, 20)); // row 1, col 1
        let runs = compute_dirty_runs(&a, &b);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].row, 0);
        assert_eq!(runs[1].row, 1);
    }

    #[test]
    fn full_row_dirty_single_run() {
        let mut a = CharGrid::new(8, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(8, 1, SENTINEL_CELL);
        fill_grid(&mut a, 'A', 10, 10, 10);
        for i in 0..8 {
            b.set(i, make_cell('B', 20, 20, 20));
        }
        let runs = compute_dirty_runs(&a, &b);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], DirtyRun { row: 0, col_start: 0, col_end: 8 });
    }

    // ── DoubleGrid tests ──────────────────────────────────────────────────

    #[test]
    fn first_real_frame_reports_all_dirty() {
        let mut dg = DoubleGrid::new(4, 3);
        // Clone the displayed (sentinel) buffer into an owned local *before*
        // taking the mutable write borrow, so the two borrows never coexist.
        let displayed = dg.displayed_buffer().clone();
        {
            let write = dg.write_buffer();
            fill_grid(write, 'X', 100, 100, 100);
        }
        let runs = compute_dirty_runs(dg.write_buffer(), &displayed);
        // Every cell differs from the sentinel, so each of the 3 rows produces
        // one full-width run — a "full redraw", one run per row.
        assert_eq!(
            runs.len(),
            3,
            "first real frame against sentinel → one run per row"
        );
        for r in &runs {
            assert_eq!((r.col_start, r.col_end), (0, 4), "full-width dirty run");
        }
        assert_eq!(
            runs.iter().map(|r| r.col_end - r.col_start).sum::<usize>(),
            4 * 3,
            "full grid is dirty"
        );
    }

    #[test]
    fn present_swaps_buffers() {
        let mut dg = DoubleGrid::new(2, 2);
        fill_grid(dg.write_buffer(), 'A', 10, 10, 10);
        dg.present();
        assert_eq!(dg.displayed_buffer().cell(0, 0).glyph, 'A');
    }

    #[test]
    fn round_trip_write_present_diff_zero_dirty() {
        let mut dg = DoubleGrid::new(3, 3);
        fill_grid(dg.write_buffer(), 'A', 10, 10, 10);
        dg.present();
        // Snapshot the now-displayed 'A' grid before re-borrowing write_buffer.
        let displayed = dg.displayed_buffer().clone();
        fill_grid(dg.write_buffer(), 'A', 10, 10, 10);
        let runs = compute_dirty_runs(dg.write_buffer(), &displayed);
        assert!(runs.is_empty());
    }

    #[test]
    fn dirty_runs_method_compares_internal_buffers() {
        // Exercises the `dirty_runs` method the video loop calls each frame.
        let mut dg = DoubleGrid::new(4, 2);
        {
            let w = dg.write_buffer();
            fill_grid(w, 'A', 10, 10, 10);
        }
        // First frame against the sentinel → one full-width run per row.
        assert_eq!(dg.dirty_runs().len(), 2);
        dg.present();
        {
            let w = dg.write_buffer();
            fill_grid(w, 'A', 10, 10, 10);
        }
        // Same content as displayed → nothing dirty.
        assert!(dg.dirty_runs().is_empty());
        // Flip one cell (row 0, col 3) → a single dirty run on row 0.
        {
            let w = dg.write_buffer();
            w.set(3, make_cell('B', 20, 20, 20));
        }
        let runs = dg.dirty_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].row, 0);
        assert_eq!(runs[0].col_start, 3);
        assert_eq!(runs[0].col_end, 4);
    }
}
