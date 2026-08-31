use ascii_renderer::image_loader::{compute_image_grid_dimensions, load_and_resize_image};
use ascii_renderer::render::{
    AsciiRenderer, ColorMode, BLOCK_RAMP, DETAILED_RAMP, SHORT_RAMP,
};
use ascii_renderer::video::{FFmpegDecoder, OutputFormat};
use std::time::Instant;

// Phase 4 imports for stress tests
use ascii_renderer::caps::ColorSupport;
use ascii_renderer::diff::{compute_dirty_runs, DirtyRun, DoubleGrid};
use ascii_renderer::palette::Palette;
use ascii_renderer::render::grid::{CharCell, CharGrid, Rgb, SENTINEL_CELL};
use ascii_renderer::terminal::{OutputState, emit_cells, write_frame};

#[test]
fn test_stress_ramp_permutations() {
    let test_ramps = [
        "",
        " ",
        "@",
        " .:-=+*#%@",
        BLOCK_RAMP,
        DETAILED_RAMP,
        "░▒▓█",
        "😀😃😄😁😆",
        "アイウエオ",
        "1234567890!@#$%^&*()",
    ];

    for ramp in test_ramps {
        for invert in [false, true] {
            for mode in [ColorMode::TrueColor, ColorMode::Grayscale, ColorMode::Monochrome] {
                let renderer = AsciiRenderer::new(ramp, mode, invert);
                for lum in 0..=255 {
                    let ch = renderer.luminance_to_char(lum);
                    assert!(!ch.is_control());
                }
            }
        }
    }
}

#[test]
fn test_stress_extreme_dimensions_and_buffers() {
    let renderer = AsciiRenderer::new(SHORT_RAMP, ColorMode::TrueColor, false);
    let mut out = Vec::new();

    // 0x0 dimension
    renderer.render_frame(&[], 0, 0, &mut out);
    assert!(out.is_empty());

    // 1x1 dimension
    let pixel = [128, 64, 32];
    renderer.render_frame(&pixel, 1, 1, &mut out);
    assert!(!out.is_empty());

    // 1x500 dimension
    let col_pixels = vec![200u8; 1500];
    renderer.render_frame(&col_pixels, 1, 500, &mut out);
    assert!(!out.is_empty());

    // 500x1 dimension
    let row_pixels = vec![100u8; 1500];
    renderer.render_frame(&row_pixels, 500, 1, &mut out);
    assert!(!out.is_empty());

    // Buffer smaller than width * height * 3 (truncated data) -> must not panic
    let truncated = vec![50u8; 10];
    out.clear();
    renderer.render_frame(&truncated, 100, 100, &mut out);
    assert!(out.is_empty());
}

#[test]
fn test_stress_color_switching_throughput() {
    let width = 160;
    let height = 60;
    let frame_size = width * height * 3;

    // Generate gradient pattern
    let mut frame = vec![0u8; frame_size];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            frame[idx] = ((x * 255) / width) as u8;
            frame[idx + 1] = ((y * 255) / height) as u8;
            frame[idx + 2] = (((x + y) * 255) / (width + height)) as u8;
        }
    }

    let renderer = AsciiRenderer::new(SHORT_RAMP, ColorMode::TrueColor, false);
    let mut out = Vec::with_capacity(width * height * 32);

    let start = Instant::now();
    let iterations = 2000;
    for _ in 0..iterations {
        renderer.render_frame(&frame, width, height, &mut out);
    }
    let elapsed = start.elapsed();
    let fps = (iterations as f64) / elapsed.as_secs_f64();

    // `cargo test` runs a debug build, which is 5-20x slower than release.
    // Assertion: release must hold 100 FPS; debug only needs a generous floor
    // that still catches pathological regressions (e.g. accidental O(n²) output).
    let (threshold, label) = if cfg!(debug_assertions) {
        (20.0, "debug")
    } else {
        (100.0, "release")
    };
    println!(
        "Rendered {} frames of {}x{} in {:?} ({:.2} FPS, {} build)",
        iterations, width, height, elapsed, fps, label
    );
    assert!(
        fps > threshold,
        "Renderer must achieve at least {} FPS in a {} build (got {:.2})",
        threshold,
        label,
        fps
    );
}

#[test]
fn test_stress_all_color_modes_utf8_validity() {
    let width = 64;
    let height = 32;
    let mut frame = vec![0u8; width * height * 3];

    // Fill with semi-random RGB values
    for (i, byte) in frame.iter_mut().enumerate() {
        *byte = ((i * 37 + 13) % 256) as u8;
    }

    let ramps = [SHORT_RAMP, DETAILED_RAMP, BLOCK_RAMP];
    let modes = [ColorMode::TrueColor, ColorMode::Grayscale, ColorMode::Monochrome];

    let mut out = Vec::new();
    for ramp in ramps {
        for mode in modes {
            for invert in [false, true] {
                let renderer = AsciiRenderer::new(ramp, mode, invert);
                renderer.render_frame(&frame, width, height, &mut out);

                let text = String::from_utf8(out.clone());
                assert!(text.is_ok(), "Rendered output must always be valid UTF-8");
            }
        }
    }
}

#[test]
fn test_stress_decoder_invalid_files() {
    let invalid_res = FFmpegDecoder::new("this_file_does_not_exist.mp4", 64, 32, None, OutputFormat::Yuv420p);
    assert!(invalid_res.is_err());

    let invalid_dims = FFmpegDecoder::new("Cargo.toml", 0, 0, None, OutputFormat::Yuv420p);
    assert!(invalid_dims.is_err());
}

#[test]
fn test_stress_image_loader_invalid_files() {
    let invalid_res = load_and_resize_image("nonexistent_img.png", 64, 32);
    assert!(invalid_res.is_err());
}

#[test]
fn test_stress_grid_dimensions_fuzz_no_panic() {
    // Sweep pathological inputs (zero dims, huge dims, tiny terminals) and
    // assert the grid function never panics and always returns in-bounds,
    // positive dimensions.
    let img_dims: [(u32, u32); 6] = [
        (0, 0),
        (1, 1),
        (100, 100),
        (u32::MAX, u32::MAX),
        (u32::MAX, 1),
        (1, u32::MAX),
    ];
    for (w, h) in img_dims {
        for tc in [1u16, 5, 80, 300, u16::MAX] {
            for tr in [1u16, 5, 40, 200, u16::MAX] {
                for (cw, ch) in [
                    (None, None),
                    (Some(0), Some(0)),
                    (Some(usize::MAX), Some(usize::MAX)),
                    (Some(1), Some(1)),
                ] {
                    let (cols, rows) =
                        compute_image_grid_dimensions(w, h, cw, ch, tc, tr, 0.5);
                    assert!(cols >= 1 && rows >= 1, "must stay positive");
                    assert!(cols <= tc as usize, "cols must fit terminal");
                    assert!(rows <= tr.saturating_sub(1).max(1) as usize, "rows must fit terminal");
                }
            }
        }
    }
}

#[test]
fn test_real_fixture_end_to_end_many_terminal_sizes() {
    // The checked-in 100x100 square fixture, driven through the full
    // Phase 1 pipeline at a sweep of terminal sizes: aspect-correct grid
    // math -> resize -> render. Every iteration must stay in bounds, produce
    // the exact expected byte count, and emit valid UTF-8.
    for term_cols in [40u16, 80u16, 120u16, 200u16] {
        for term_rows in [20u16, 40u16, 60u16] {
            let max_rows = term_rows.saturating_sub(1).max(1);
            let (cols, rows) =
                compute_image_grid_dimensions(100, 100, None, None, term_cols, term_rows, 0.5);
            assert!(cols >= 1 && rows >= 1, "dims must be positive");
            assert!(cols <= term_cols as usize, "cols exceed terminal");
            assert!(rows <= max_rows as usize, "rows exceed terminal");

            let frame = load_and_resize_image("test_circle.png", cols as u32, rows as u32)
                .expect("checked-in fixture should always load");
            assert_eq!(frame.rgb_data.len(), cols * rows * 3, "resize must match grid exactly");

            for invert in [false, true] {
                let renderer = AsciiRenderer::new(SHORT_RAMP, ColorMode::TrueColor, invert);
                let mut out = Vec::new();
                renderer.render_frame(&frame.rgb_data, cols, rows, &mut out);
                assert!(
                    String::from_utf8(out).is_ok(),
                    "rendered frame must be valid UTF-8 at {}x{}",
                    cols,
                    rows
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Phase 4 — Comprehensive Stress Tests
//
// Covers the diff, palette, dither, emission, and full-pipeline subsystems
// across extreme dimensions, adversarial inputs, and multi-frame sequences.
// ══════════════════════════════════════════════════════════════════════════════

fn cell(glyph: char, r: u8, g: u8, b: u8) -> CharCell {
    CharCell {
        glyph,
        color: Rgb::new(r, g, b),
    }
}

fn fill_grid_stress(grid: &mut CharGrid, glyph: char, r: u8, g: u8, b: u8) {
    let c = cell(glyph, r, g, b);
    for i in 0..grid.len() {
        grid.set(i, c);
    }
}

// ── Diff / DoubleGrid stress ──────────────────────────────────────────────

#[test]
fn test_stress_diff_1x1_grid_boundary() {
    let mut a = CharGrid::new(1, 1, SENTINEL_CELL);
    let mut b = CharGrid::new(1, 1, SENTINEL_CELL);
    a.set(0, cell('A', 1, 1, 1));
    b.set(0, cell('B', 2, 2, 2));
    let runs = compute_dirty_runs(&b, &a);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0], DirtyRun { row: 0, col_start: 0, col_end: 1 });
}

#[test]
fn test_stress_diff_every_other_column_merges() {
    // 100 columns, every other column dirty: raw runs gap=1 ≤ 4 → all merge.
    let cols = 100;
    let mut a = CharGrid::new(cols, 1, SENTINEL_CELL);
    let mut b = CharGrid::new(cols, 1, SENTINEL_CELL);
    fill_grid_stress(&mut a, 'A', 10, 10, 10);
    fill_grid_stress(&mut b, 'A', 10, 10, 10);
    for i in (0..cols).step_by(2) {
        b.set(i, cell('B', 20, 20, 20));
    }
    let runs = compute_dirty_runs(&b, &a);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].col_start, 0);
    // Dirty cells sit at 0,2,...,98. The merged run is half-open [0, 99): the
    // trailing clean cell at 99 is never part of any raw run, so the merged
    // run's end is the last dirty cell + 1, not the grid width.
    assert_eq!(runs[0].col_end, cols - 1);
    assert_eq!(
        runs[0].col_end - runs[0].col_start,
        (cols / 2) + (cols / 2 - 1),
        "every dirty cell plus its 99 merged gap cells are redrawn"
    );
}

#[test]
fn test_stress_diff_sparse_dirty_merge_boundary() {
    let cols = 30;
    // Dirty cells spaced 4 apart (gap=3 ≤ 4) → all merge.
    {
        let mut a = CharGrid::new(cols, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(cols, 1, SENTINEL_CELL);
        fill_grid_stress(&mut a, 'A', 10, 10, 10);
        fill_grid_stress(&mut b, 'A', 10, 10, 10);
        for i in (0..cols).step_by(4) {
            b.set(i, cell('B', 20, 20, 20));
        }
        let runs = compute_dirty_runs(&b, &a);
        assert_eq!(runs.len(), 1, "spaced 4 apart → gap=3 ≤ threshold → merge");
        // Dirty at 0,4,8,...,28 → merged run [0, 29): last dirty cell at 28,
        // the trailing clean cell 29 never enters a run.
        assert_eq!(runs[0].col_end, cols - 1);
    }
    // Dirty cells spaced 6 apart (gap=5 > 4) → separate.
    {
        let mut a = CharGrid::new(cols, 1, SENTINEL_CELL);
        let mut b = CharGrid::new(cols, 1, SENTINEL_CELL);
        fill_grid_stress(&mut a, 'A', 10, 10, 10);
        fill_grid_stress(&mut b, 'A', 10, 10, 10);
        for i in (0..cols).step_by(6) {
            b.set(i, cell('B', 20, 20, 20));
        }
        let runs = compute_dirty_runs(&b, &a);
        assert!(runs.len() > 1, "spaced 6 apart → gap=5 > threshold → separate");
    }
}

#[test]
fn test_stress_doublegrid_1000_frame_oscillation() {
    let mut dg = DoubleGrid::new(10, 10);
    for frame in 0..1000u32 {
        {
            let w = dg.write_buffer();
            for i in 0..w.len() {
                w.set(i, if frame % 2 == 0 { cell('A', 10, 10, 10) } else { cell('B', 20, 20, 20) });
            }
        }
        let runs = dg.dirty_runs();
        if frame > 0 {
            // Full-grid flip: every cell differs from the previous frame → one
            // full-width half-open run PER ROW (runs never span rows, per the
            // D3 contract). Grid is 10 rows × 10 cols → 10 runs.
            assert_eq!(runs.len(), 10, "frame {}: full-grid flip → one run per row", frame);
            for r in &runs {
                assert_eq!((r.col_start, r.col_end), (0, 10), "full-width per-row run");
            }
            assert_eq!(
                runs.iter().map(|r| r.col_end - r.col_start).sum::<usize>(),
                10 * 10,
                "entire grid dirty — total coverage == rows*cols"
            );
        }
        dg.present();
    }
}

// ── Palette stress ────────────────────────────────────────────────────────

#[test]
fn test_stress_palette_full_spectrum_valid_index() {
    let p256 = Palette::xterm256();
    let p16 = Palette::basic16();
    for r in (0..=255).step_by(17) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(17) {
                let color = Rgb::new(r, g, b);
                let idx256 = p256.nearest_index(color);
                let idx16 = p16.nearest_index(color);
                assert!(idx256 < 256, "xterm256 OOB at ({},{},{})", r, g, b);
                assert!(idx16 < 16, "basic16 OOB at ({},{},{})", r, g, b);
            }
        }
    }
}

#[test]
fn test_stress_palette_extreme_colors() {
    let p = Palette::xterm256();
    assert_eq!(p.nearest_index(Rgb::new(0, 0, 0)), 0);
    let white = p.nearest_index(Rgb::new(255, 255, 255));
    assert!(white <= 15 || (white >= 232 && white <= 255),
        "white should match a bright entry, got {}", white);
    assert!(p.nearest_index(Rgb::new(127, 127, 127)) < 256);
}

// ── Dither stress ─────────────────────────────────────────────────────────

#[test]
fn test_stress_dither_large_grid_consistency() {
    use ascii_renderer::dither::ordered_dither_quantize;
    let p = Palette::xterm256();
    let cols = 200;
    let rows = 100;
    let mut results = vec![0u8; cols * rows];
    for y in 0..rows {
        for x in 0..cols {
            let r = ((x as f32 / cols as f32) * 255.0) as u8;
            let g = ((y as f32 / rows as f32) * 255.0) as u8;
            let idx = ordered_dither_quantize(Rgb::new(r, g, 128), x, y, &p);
            results[y * cols + x] = idx;
            assert!((idx as usize) < p.entries.len());
        }
    }
    // Determinism: re-run identical inputs → same output.
    for y in 0..rows {
        for x in 0..cols {
            let r = ((x as f32 / cols as f32) * 255.0) as u8;
            let g = ((y as f32 / rows as f32) * 255.0) as u8;
            let idx = ordered_dither_quantize(Rgb::new(r, g, 128), x, y, &p);
            assert_eq!(idx, results[y * cols + x], "non-deterministic at ({},{})", x, y);
        }
    }
}

// ── Emission stress ───────────────────────────────────────────────────────

/// Count exact occurrences of an escape sequence in `out` byte buffer.
fn count_escape(out: &[u8], esc: &[u8]) -> usize {
    out.windows(esc.len()).filter(|w| *w == esc).count()
}

#[test]
fn test_stress_emission_alternating_dirty_cells() {
    // Every-other-column dirty pattern: dirty cells (100,100,100) interleaved
    // with clean gap cells (50,50,50). All raw runs merge into ONE half-open
    // run [0, cols-1) because gap = 1 ≤ MERGE_GAP_THRESHOLD.
    //
    // D1 compression: each emitted cell alternates color from its predecessor,
    // so the color escapes cannot be suppressed — exactly one escape per emitted
    // cell is the correct behavior, proving D1 never over-suppresses a genuine
    // color change.
    let cols = 200;
    let mut displayed = CharGrid::new(cols, 1, SENTINEL_CELL);
    let mut current = CharGrid::new(cols, 1, SENTINEL_CELL);
    fill_grid_stress(&mut displayed, 'A', 50, 50, 50);
    fill_grid_stress(&mut current, 'A', 50, 50, 50);
    for i in (0..cols).step_by(2) {
        current.set(i, cell('B', 100, 100, 100));
    }
    let runs = compute_dirty_runs(&current, &displayed);
    assert_eq!(runs.len(), 1, "merge across the 1-cell gaps → single run");
    assert_eq!(runs[0].col_start, 0);
    // Dirty cells 0,2,...,198 → merged half-open run covers [0, cols-1).
    assert_eq!(runs[0].col_end, cols - 1);

    let palette = Palette::xterm256();
    let mut state = OutputState::new(ColorSupport::TrueColor, false);
    let mut out = Vec::new();
    emit_cells(&current, &runs, &mut state, &mut out, &palette, false);

    let esc_100 = count_escape(&out, b"\x1b[38;2;100;100;100m");
    let esc_50  = count_escape(&out, b"\x1b[38;2;50;50;50m");
    // cols/2 = 100 even cells (dirty), cols/2-1 = 99 odd cells (gap).
    // Total 199 = cols-1 emitted cells, one escape each.
    assert_eq!(esc_100, cols / 2, "one escape per dirty (even) column");
    assert_eq!(esc_50,  cols / 2 - 1, "one escape per clean (odd) gap column");
    assert_eq!(
        esc_100 + esc_50, cols - 1,
        "exactly one escape per emitted cell — no suppression of genuine color change"
    );
    assert!(out.windows(4).any(|w| w == b"\x1b[1;"), "cursor jumps present");
}

#[test]
fn test_stress_emission_multicolor_state_transitions() {
    let mut displayed = CharGrid::new(50, 1, SENTINEL_CELL);
    let mut current = CharGrid::new(50, 1, SENTINEL_CELL);
    fill_grid_stress(&mut displayed, 'X', 0, 0, 0);
    let colors: [(u8,u8,u8); 4] = [(10,20,30), (40,50,60), (70,80,90), (100,110,120)];
    for i in 0..50 { current.set(i, cell('X', 0, 0, 0)); }
    for (j, &(r, g, b)) in colors.iter().enumerate() {
        current.set(j * 10, cell('Y', r, g, b));
    }
    let runs = compute_dirty_runs(&current, &displayed);
    assert_eq!(runs.len(), 4, "four separate dirty cells → four runs");
    let palette = Palette::xterm256();
    let mut state = OutputState::new(ColorSupport::TrueColor, false);
    let mut out = Vec::new();
    emit_cells(&current, &runs, &mut state, &mut out, &palette, false);
    let mut escape_count = 0;
    let mut i = 0;
    while i < out.len() {
        if i + 4 < out.len() && &out[i..i+5] == b"\x1b[38;" {
            escape_count += 1;
            while i < out.len() && out[i] != b'm' { i += 1; }
        }
        i += 1;
    }
    assert_eq!(escape_count, 4, "four distinct colors → four escapes");
}

#[test]
fn test_stress_emission_monochrome_no_sgr() {
    // Monochrome user mode: cursor jumps + glyphs only — ZERO SGR color escapes,
    // regardless of the cell colors or the terminal's color support.
    let cols = 80;
    let mut displayed = CharGrid::new(cols, 1, SENTINEL_CELL);
    let mut current = CharGrid::new(cols, 1, SENTINEL_CELL);
    for c in 0..cols {
        displayed.set(c, cell('A', 10, 10, 10));
    }
    for c in 0..cols {
        let r = (c as u8).wrapping_mul(10);
        current.set(c, cell('E', r, 255 - r, r.wrapping_mul(3)));
    }
    let runs = compute_dirty_runs(&current, &displayed);
    assert_eq!(runs.len(), 1);
    let palette = Palette::xterm256();
    let mut state = OutputState::new(ColorSupport::TrueColor, false);
    let mut out = Vec::new();
    emit_cells(&current, &runs, &mut state, &mut out, &palette, true /* mono */);

    assert_eq!(count_escape(&out, b"\x1b[38"), 0, "mono emits no SGR color");
    assert!(out.windows(4).any(|w| w == b"\x1b[1;"), "cursor jump still present");
    // Single run starting at (row 1, col 1) → one 6-byte jump + all glyphs.
    assert_eq!(out.len(), b"\x1b[1;1H".len() + cols, "one jump + glyphs, no SGR");
}

#[test]
fn test_stress_emission_indexed_palette_compression() {
    // Reduced-palette emission path (Basic16): resolved colors are palette
    // INDICES. With a 16-entry palette the dither nudge is only ±12.5, so pure
    // primaries (values exactly on an entry, far from any neighbor) resolve to
    // the SAME index at every column — a true test of D1 compression on the
    // Indexed path. (A dithered 256-color gradient genuinely oscillates its
    // index column to column — that is dithering's job — so it's not a
    // compression target; see the note appended to this test.)
    let cols = 200;
    let mut displayed = CharGrid::new(cols, 1, SENTINEL_CELL);
    let mut current = CharGrid::new(cols, 1, SENTINEL_CELL);
    for c in 0..cols {
        displayed.set(c, cell('A', 0, 0, 0));
    }
    // Left half bright blue (255,0,0), right half bright red (0,0,255).
    for c in 0..cols {
        current.set(c, if c < 100 {
            cell('A', 255, 0, 0)
        } else {
            cell('A', 0, 0, 255)
        });
    }
    let runs = compute_dirty_runs(&current, &displayed);
    assert_eq!(runs.len(), 1, "full row differs → one merged run");

    let palette = Palette::basic16();
    let mut state = OutputState::new(ColorSupport::Basic16, false);
    let mut out = Vec::new();
    emit_cells(&current, &runs, &mut state, &mut out, &palette, false);

    // Parse every \x1b[38;5;N escape: in-bounds, valid syntax.
    let mut i = 0;
    let mut escapes = 0;
    while i + 7 <= out.len() {
        if &out[i..i + 7] == b"\x1b[38;5;" {
            escapes += 1;
            let mut j = i + 7;
            let mut v: usize = 0;
            while j < out.len() && out[j].is_ascii_digit() {
                v = v * 10 + (out[j] - b'0') as usize;
                j += 1;
            }
            assert!(v < 256, "palette index {} out of range", v);
            assert!(j < out.len() && out[j] == b'm', "malformed escape at {}", i);
            i = j;
        }
        i += 1;
    }
    // Each half is a single constant resolved index → exactly 2 escapes
    // (one per color region), even across 100 columns each — D1 collapses the
    // 199-cell run to 3 color transitions total.
    assert_eq!(escapes, 2, "constant indexed regions → one escape per region");
}

// ── Moving-sprite cascading diff stress ───────────────────────────────────

#[test]
fn test_stress_diff_moving_region_stays_localized() {
    // A 3×3 "sprite" drifts across a 20×20 uniform field. Each frame only the
    // sprite's old + new footprints change; everything else must stay clean,
    // frame after frame. This is the cascading-effect regression probe: a
    // hidden stale-dirty bug would make the dirty coverage sprawl.
    let (rows, cols) = (20usize, 20usize);
    let bg = cell('A', 50, 50, 50);
    let sprite = cell('X', 200, 30, 30);
    let mut dg = DoubleGrid::new(cols, rows);

    for step in 0..60usize {
        {
            let w = dg.write_buffer();
            for idx in 0..w.len() {
                w.set(idx, bg);
            }
            let sx = step % (cols - 3);
            let sy = [(step % (rows - 3)), (rows - 3 - (step % (rows - 3)))][step / 20 % 2];
            for dy in 0..3 {
                for dx in 0..3 {
                    w.set((sy + dy) * cols + (sx + dx), sprite);
                }
            }
        }
        if step == 0 {
            dg.present();
            continue; // first frame vs sentinel is a full redraw
        }
        let runs = dg.dirty_runs();
        let dirty: usize = runs.iter().map(|r| r.col_end - r.col_start).sum();
        assert!(dirty > 0, "step {}: sprite moved → something dirty", step);
        // Changed cells are exactly old∪new footprint (≤ 2 sprite areas = 18);
        // merged gaps add at most a couple of cells per row.
        assert!(
            dirty <= 2 * 3 * 3 + 8,
            "step {}: dirty coverage {} must stay localized to the moved sprite",
            step, dirty
        );
        dg.present();
    }
}

// ── Multi-frame pipeline stress ──────────────────────────────────────────

#[test]
fn test_stress_multiframe_incremental_pipeline() {
    let cols = 10;
    let mut dg = DoubleGrid::new(cols, 1);
    let palette = Palette::xterm256();
    let mut state = OutputState::new(ColorSupport::TrueColor, false);

    for frame in 0..10 {
        {
            let w = dg.write_buffer();
            for c in 0..cols {
                let glyph = if c < frame { 'A' } else { 'B' };
                let color = if c < frame { (10, 10, 10) } else { (20, 20, 20) };
                w.set(c, cell(glyph, color.0, color.1, color.2));
            }
        }
        let runs = dg.dirty_runs();
        if frame == 0 {
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0].col_end, cols);
        } else {
            assert_eq!(runs.len(), 1, "frame {}: one new dirty cell", frame);
            assert_eq!(runs[0].col_end - runs[0].col_start, 1);
        }
        let mut out = Vec::new();
        emit_cells(dg.write_buffer(), &runs, &mut state, &mut out, &palette, false);
        let mut sink = Vec::new();
        write_frame(&out, false, &mut sink).unwrap();
        assert_eq!(sink, out, "no sync markers when unsupported");
        dg.present();
    }
    // After 10th frame, write identical content → zero runs.
    {
        let w = dg.write_buffer();
        for c in 0..cols { w.set(c, cell('A', 10, 10, 10)); }
    }
    // Displayed was the frame-9 content which ended with 'B' everywhere.
    // write is 'A' everywhere → everything changed → 1 run.
    // So let's copy what was displayed: frame 9 has cols 0..9 = 'A', col 9 = 'B'.
    // Actually frame 9: c < 9 → 'A'(10), c == 9 → 'B'(20). So display = [A*9,B].
    // If we now write [A*9, A] → only col 9 changes (B→A) → 1 run.
    {
        let w = dg.write_buffer();
        for c in 0..cols { w.set(c, cell('A', 10, 10, 10)); }
    }
    let runs = dg.dirty_runs();
    assert_eq!(runs.len(), 1, "only last col differs → one run");
    assert_eq!(runs[0].col_start, 9);
    assert_eq!(runs[0].col_end, 10);
}

// ── caps parse stress ─────────────────────────────────────────────────────

#[test]
fn test_stress_caps_parse_adversarial_byte_sequences() {
    use ascii_renderer::caps::parse_sync_reply;
    // Huge status digits → fail closed.
    assert_eq!(parse_sync_reply(b"\x1b[?2026;12345$y"), Some(false));
    // Empty status → fail closed.
    assert_eq!(parse_sync_reply(b"\x1b[?2026;$y"), Some(false));
    // Incomplete → pending.
    assert_eq!(parse_sync_reply(b"\x1b[?2026;"), None);
    // 1024 bytes junk then valid reply → found.
    let mut acc = vec![0x42u8; 1024];
    acc.extend_from_slice(b"\x1b[?2026;3$y");
    assert_eq!(parse_sync_reply(&acc), Some(true));
}

// ── write_frame stress ────────────────────────────────────────────────────

#[test]
fn test_stress_write_frame_empty_and_large_payloads() {
    let mut sink = Vec::new();
    write_frame(b"", false, &mut sink).unwrap();
    assert!(sink.is_empty());
    let payload = vec![b'x'; 1_000_000];
    let mut sink = Vec::new();
    write_frame(&payload, true, &mut sink).unwrap();
    assert_eq!(&sink[..8], b"\x1b[?2026h");
    assert_eq!(&sink[sink.len()-8..], b"\x1b[?2026l");
    assert_eq!(&sink[8..sink.len()-8], payload.as_slice());
}
