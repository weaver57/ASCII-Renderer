//! Rayon-accelerated data parallelism for image downsampling and edge processing.
//!
//! Implements Milestone 5.5 and 5.6 (§4.7) of the Phase 5 Architecture Plan:
//! - Row-sharded box-filter downsampling of YUV planes.
//! - Row-band-sharded Sobel gradient computation and Non-Maximum Suppression (NMS).
//! - Zero `unsafe` and zero halo-copying: all workers read from shared, immutable input slices
//!   while writing disjoint, non-overlapping slices partitioned by `par_chunks_mut`.

use rayon::prelude::*;
use wide::f32x8;

use crate::render::edge::{nms_bin, GradientMap};
use crate::video::yuv::{yuv_to_rgb, ColorSpace};
use wide::{CmpGe, CmpGt};

/// SIMD lane count used throughout Phase 5 (§4.8).
const LANES: usize = 8;

/// Number of source rows per Rayon band for Sobel/NMS sharding.
///
/// Chosen so `band_count` is a small multiple of the worker count; Rayon's
/// work-stealing scheduler then load-balances the bands across threads.
const BAND_ROWS: usize = 8;

/// Initializes the global Rayon thread pool with a thread budget leaving headroom for
/// the Decode and Render OS threads (§4.7 of Phase 5 plan).
///
/// Returns `Ok(())` if newly initialized, or ignores if already initialized.
pub fn init_thread_pool() {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let rayon_threads = n.saturating_sub(2).max(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .build_global();
}

/// Downsamples raw Y, U, V planes into per-cell luma and RGB colors in parallel across character rows.
///
/// Shards by output character-grid rows using `par_chunks_mut(cols)`. Each worker computes its
/// assigned character-grid rows' cells by reading their non-overlapping source rectangles
/// from the shared, immutable Y/U/V plane slices.
///
/// **Safety & Concurrency**:
/// - Zero `unsafe`.
/// - Zero halo-padding or ghost cells needed because box-filter cells do not sample outside their
///   designated source bounding box.
/// - Output writes are non-overlapping by construction of `par_chunks_mut`.
pub fn downsample_yuv_planes_parallel(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    src_w: usize,
    src_h: usize,
    color_space: ColorSpace,
    cols: usize,
    rows: usize,
    cell_luma: &mut [f32],
    cell_color: &mut [(u8, u8, u8)],
) {
    if cols == 0 || rows == 0 || src_w == 0 || src_h == 0 {
        return;
    }
    debug_assert!(cell_luma.len() >= cols * rows);
    debug_assert!(cell_color.len() >= cols * rows);

    let luma_target = &mut cell_luma[..cols * rows];
    let color_target = &mut cell_color[..cols * rows];

    luma_target
        .par_chunks_mut(cols)
        .zip(color_target.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(row, (luma_row, color_row))| {
            for col in 0..cols {
                // Luma-space rectangle
                let x0 = col * src_w / cols;
                let x1 = ((col + 1) * src_w / cols).max(x0 + 1);
                let y0 = row * src_h / rows;
                let y1 = ((row + 1) * src_h / rows).max(y0 + 1);

                // Chroma-space rectangle (half resolution, with ceil on upper bound)
                let cx0 = x0 / 2;
                let cx1 = (x1 + 1) / 2; // div_ceil(x1, 2)
                let cy0 = y0 / 2;
                let cy1 = (y1 + 1) / 2;

                // Average Y over luma rect
                let mut y_acc = 0.0f32;
                let mut y_count = 0u32;
                for py in y0..y1 {
                    for px in x0..x1 {
                        y_acc += y_plane[py * src_w + px] as f32;
                        y_count += 1;
                    }
                }
                let avg_y = if y_count > 0 {
                    y_acc / y_count as f32
                } else {
                    0.0
                };

                // Average U, V over chroma rect
                let chroma_src_w = (src_w / 2).max(1);
                let mut u_acc = 0.0f32;
                let mut v_acc = 0.0f32;
                let mut c_count = 0u32;
                for py in cy0..cy1.min(src_h / 2) {
                    for px in cx0..cx1.min(src_w / 2) {
                        u_acc += u_plane[py * chroma_src_w + px] as f32;
                        v_acc += v_plane[py * chroma_src_w + px] as f32;
                        c_count += 1;
                    }
                }
                let (avg_u, avg_v) = if c_count > 0 {
                    (u_acc / c_count as f32, v_acc / c_count as f32)
                } else {
                    (128.0, 128.0) // neutral chroma
                };

                luma_row[col] = avg_y;
                color_row[col] = yuv_to_rgb(avg_y, avg_u, avg_v, color_space);
            }
        });
}

/// Rayon row-band-sharded Sobel gradient computation over a full-resolution luma map.
///
/// Returns a [`GradientMap`] (magnitude + angle) identical to the scalar
/// [`build_gradient_map`](crate::render::edge::build_gradient_map).
///
/// **Safety & Concurrency — the shared-read / disjoint-write structure:**
/// - Every worker reads from the *same* shared `&[f32] luma` slice (read-only,
///   never partitioned) — reads never race.
/// - Each worker writes only into its own disjoint chunk of `magnitude`/`angle`,
///   carved out by `par_chunks_mut` (each closure's slices are non-overlapping).
/// - Sobel's 3×3 neighborhood reads one row into a *neighboring* band's
///   territory at a band boundary — that is fine, because that read targets the
///   shared immutable `luma`, not another worker's in-progress output.
///
/// This is why the design needs **zero `unsafe` and zero halo-copying**: the
/// input is immutable for the duration of the pass, so partition-boundary reads
/// are safe by construction.
pub fn build_gradient_map_parallel(luma: &[f32], width: usize, height: usize) -> GradientMap {
    let mut magnitude = vec![0.0f32; luma.len()];
    let mut angle = vec![0.0f32; luma.len()];

    // Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = [-1 -2 -1; 0 0 0; 1 2 1] (y down).
    // Identical constants to the scalar kernel in edge.rs.
    const GX: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const GY: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    let w = width as i32;
    let h = height as i32;
    let band_px = BAND_ROWS * width;

    magnitude
        .par_chunks_mut(band_px)
        .zip(angle.par_chunks_mut(band_px))
        .enumerate()
        .for_each(|(band_idx, (mag_band, ang_band))| {
            let row_start = band_idx * BAND_ROWS;
            let row_end = (row_start + BAND_ROWS).min(height);
            for y in row_start..row_end {
                let row_off = (y - row_start) * width;
                let yi = y as i32;
                for x in 0..w {
                    let mut gx = 0.0f32;
                    let mut gy = 0.0f32;
                    for (ky, dy) in [-1i32, 0, 1].iter().enumerate() {
                        for (kx, dx) in [-1i32, 0, 1].iter().enumerate() {
                            let ny = (yi + dy).clamp(0, h - 1);
                            let nx = (x + dx).clamp(0, w - 1);
                            let val = luma[(ny * w + nx) as usize];
                            gx += GX[ky][kx] as f32 * val;
                            gy += GY[ky][kx] as f32 * val;
                        }
                    }
                    mag_band[row_off + x as usize] = (gx * gx + gy * gy).sqrt();
                    ang_band[row_off + x as usize] = gy.atan2(gx);
                }
            }
        });

    GradientMap {
        width: width as u32,
        height: height as u32,
        magnitude,
        angle,
    }
}

/// Loads a 3-row × 10-element window centered at column `cx` with boundary clamping.
///
/// Returns three rows of 10 `f32` values each. This is the minimum needed for
/// 8 SIMD-processed pixels starting at `cx`: the 3×3 Sobel kernel requires one
/// column of context on each side, and the `wide::f32x8` lane width demands 8
/// elements plus the 2 boundary columns.
#[inline]
fn load_sobel_3x10(luma: &[f32], w: usize, h: usize, cx: usize, cy: usize) -> [[f32; 10]; 3] {
    let mut rows = [[0.0f32; 10]; 3];
    for ky in 0..3i32 {
        let sy = (cy as i32 + ky - 1).clamp(0, h as i32 - 1) as usize;
        for kx in 0..10i32 {
            let sx = (cx as i32 + kx - 1).clamp(0, w as i32 - 1) as usize;
            rows[ky as usize][kx as usize] = luma[sy * w + sx];
        }
    }
    rows
}

/// Processes a single row band with SIMD-accelerated Sobel convolution.
///
/// `luma` is the full-resolution luma map (absolute indexing).
/// `magnitude` / `angle` are band-local slices from `par_chunks_mut`
/// (relative indexing: row 0 of the band = index 0).
/// `band_rows` is the number of rows in this band.
/// `abs_row_start` is the absolute y-coordinate of the band's first row in the
/// full image — needed for `luma` lookups and boundary clamping.
fn build_gradient_map_simd_band(
    luma: &[f32],
    magnitude: &mut [f32],
    angle: &mut [f32],
    w: usize,
    h: usize,
    band_rows: usize,
    abs_row_start: usize,
) {
    const GX_ROW: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    const GY_ROW: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    for band_y in 0..band_rows {
        let abs_y = abs_row_start + band_y;
        let row_off = band_y * w; // relative to band slice
        let mut x = 0;

        // SIMD loop: process 8 pixels at a time
        while x + 8 <= w {
            let mut gx_acc = f32x8::ZERO;
            let mut gy_acc = f32x8::ZERO;

            for ky in 0..3 {
                let win = load_sobel_3x10(luma, w, h, x, abs_y);
                // Left neighbors (indices 0..8), center (indices 1..9),
                // right neighbors (indices 2..10)
                let left_vals = f32x8::new([
                    win[ky][0], win[ky][1], win[ky][2], win[ky][3],
                    win[ky][4], win[ky][5], win[ky][6], win[ky][7],
                ]);
                let center_vals = f32x8::new([
                    win[ky][1], win[ky][2], win[ky][3], win[ky][4],
                    win[ky][5], win[ky][6], win[ky][7], win[ky][8],
                ]);
                let right_vals = f32x8::new([
                    win[ky][2], win[ky][3], win[ky][4], win[ky][5],
                    win[ky][6], win[ky][7], win[ky][8], win[ky][9],
                ]);

                // Gx[ky] = left * GX[ky][0] + center * GX[ky][1] + right * GX[ky][2]
                // (GX[ky][1] is always 0, but keep the term for symmetry.)
                gx_acc += left_vals * f32x8::new([GX_ROW[ky][0]; 8])
                    + center_vals * f32x8::new([GX_ROW[ky][1]; 8])
                    + right_vals * f32x8::new([GX_ROW[ky][2]; 8]);

                // Gy[ky] = left * GY[ky][0] + center * GY[ky][1] + right * GY[ky][2]
                gy_acc += left_vals * f32x8::new([GY_ROW[ky][0]; 8])
                    + center_vals * f32x8::new([GY_ROW[ky][1]; 8])
                    + right_vals * f32x8::new([GY_ROW[ky][2]; 8]);
            }

            // magnitude = sqrt(gx² + gy²)
            let mag = (gx_acc * gx_acc + gy_acc * gy_acc).sqrt();
            let mag_arr: [f32; 8] = mag.into();
            for i in 0..8 {
                magnitude[row_off + x + i] = mag_arr[i];
            }

            // Angle: scalar atan2 loop (wide has no portable atan2)
            let gx_arr: [f32; 8] = gx_acc.into();
            let gy_arr: [f32; 8] = gy_acc.into();
            for i in 0..8 {
                angle[row_off + x + i] = gy_arr[i].atan2(gx_arr[i]);
            }

            x += 8;
        }

        // Scalar tail for remaining columns
        for xv in x..w {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            let win = load_sobel_3x10(luma, w, h, xv, abs_y);
            for ky in 0..3 {
                gx += GX_ROW[ky][0] * win[ky][0] + GX_ROW[ky][1] * win[ky][1] + GX_ROW[ky][2] * win[ky][2];
                gy += GY_ROW[ky][0] * win[ky][0] + GY_ROW[ky][1] * win[ky][1] + GY_ROW[ky][2] * win[ky][2];
            }
            magnitude[row_off + xv] = (gx * gx + gy * gy).sqrt();
            angle[row_off + xv] = gy.atan2(gx);
        }
    }
}

/// SIMD-accelerated Sobel gradient computation (§4.9).
///
/// Row-band-sharded using Rayon, with SIMD (`wide::f32x8`) processing 8 pixels
/// per iteration within each band. The scalar tail handles widths that are not
/// multiples of 8.
///
/// Bit-exact with both the scalar [`build_gradient_map`] and the non-SIMD
/// parallel [`build_gradient_map_parallel`], verified in `tests/simd_tests.rs`.
pub fn build_gradient_map_simd(luma: &[f32], width: usize, height: usize) -> GradientMap {
    let mut magnitude = vec![0.0f32; luma.len()];
    let mut angle = vec![0.0f32; luma.len()];

    let band_px = BAND_ROWS * width;

    magnitude
        .par_chunks_mut(band_px)
        .zip(angle.par_chunks_mut(band_px))
        .enumerate()
        .for_each(|(band_idx, (mag_band, ang_band))| {
            let abs_row_start = band_idx * BAND_ROWS;
            let band_rows = (abs_row_start + BAND_ROWS).min(height) - abs_row_start;
            build_gradient_map_simd_band(
                luma,
                mag_band,
                ang_band,
                width,
                height,
                band_rows,
                abs_row_start,
            );
        });

    GradientMap {
        width: width as u32,
        height: height as u32,
        magnitude,
        angle,
    }
}

/// Rayon row-band-sharded Non-Maximum Suppression.
///
/// Reads the now-complete `gradient` (magnitude + angle) and writes the
/// suppressed output in disjoint per-band slices. Same shared-read /
/// disjoint-write reasoning as [`build_gradient_map_parallel`]: the read of
/// neighbor pixels at a band boundary goes against the shared, immutable
/// `gradient`, never another worker's in-progress output.
///
/// The asymmetric `>` / `>=` comparisons and the angle-quantization table are
/// byte-for-byte identical to the scalar [`non_max_suppress`](crate::render::edge::non_max_suppress).
pub fn non_max_suppress_parallel(gradient: &GradientMap) -> Vec<f32> {
    let w = gradient.width as i32;
    let h = gradient.height as i32;
    let mut out = vec![0.0f32; gradient.magnitude.len()];
    let band_px = BAND_ROWS * w as usize;

    out.par_chunks_mut(band_px)
        .enumerate()
        .for_each(|(band_idx, band)| {
            let row_start = band_idx * BAND_ROWS;
            let row_end = (row_start + BAND_ROWS).min(h as usize);
            for y in row_start..row_end {
                let row_off = (y - row_start) * w as usize;
                for x in 0..w {
                    let idx = (y * w as usize + x as usize) as usize;
                    let mag = gradient.magnitude[idx];
                    if mag == 0.0 {
                        continue;
                    }
                    let [a, b] = nms_bin(gradient.angle[idx]);
                    let na = ((y as i32 + a.1).clamp(0, h - 1) * w + (x + a.0).clamp(0, w - 1))
                        as usize;
                    let nb = ((y as i32 + b.1).clamp(0, h - 1) * w + (x + b.0).clamp(0, w - 1))
                        as usize;
                    if mag > gradient.magnitude[na] && mag >= gradient.magnitude[nb] {
                        band[row_off + x as usize] = mag;
                    }
                }
            }
        });

    out
}

/// Returns the 0..4 direction-bin index for an angle, matching the arms of
/// [`nms_bin`](crate::render::edge::nms_bin) exactly. Used to build the
/// per-lane SIMD selection masks in `non_max_suppress_simd`.
#[inline]
fn nms_bin_index(angle: f32) -> usize {
    let base_deg = (angle.to_degrees()).rem_euclid(180.0);
    if base_deg < 22.5 || base_deg >= 157.5 {
        0 // ~horizontal gradient: [(-1,0),(1,0)]
    } else if base_deg < 67.5 {
        1 // ~45° diagonal: [(1,1),(-1,-1)]
    } else if base_deg < 112.5 {
        2 // ~vertical gradient: [(0,-1),(0,1)]
    } else {
        3 // ~135° diagonal: [(1,-1),(-1,1)]
    }
}

/// Loads 8 contiguous `f32` values from `slice` starting at `start`.
#[inline]
fn load8(slice: &[f32], start: usize) -> f32x8 {
    f32x8::new([
        slice[start],
        slice[start + 1],
        slice[start + 2],
        slice[start + 3],
        slice[start + 4],
        slice[start + 5],
        slice[start + 6],
        slice[start + 7],
    ])
}

/// Scalar NMS evaluation for one pixel at (x, y), identical to the Phase 2
/// reference. Writes into the band-local `out` at `row_off + x`. Used for
/// boundary rows, boundary columns, and the non-multiple-of-8 tail.
#[inline]
fn nms_scalar_pixel(
    gradient: &GradientMap,
    w: i32,
    h: i32,
    x: i32,
    y: i32,
    out: &mut [f32],
    row_off: usize,
) {
    let idx = (y * w + x) as usize;
    let mag = gradient.magnitude[idx];
    if mag == 0.0 {
        return;
    }
    let [a, b] = nms_bin(gradient.angle[idx]);
    let na = ((y + a.1).clamp(0, h - 1) * w + (x + a.0).clamp(0, w - 1)) as usize;
    let nb = ((y + b.1).clamp(0, h - 1) * w + (x + b.0).clamp(0, w - 1)) as usize;
    if mag > gradient.magnitude[na] && mag >= gradient.magnitude[nb] {
        out[row_off + x as usize] = mag;
    }
}

/// SIMD NMS evaluation for 8 consecutive interior pixels in row `y` starting at
/// column `s`.
///
/// The four-way direction branch of the scalar kernel becomes a per-lane SIMD
/// *select*: for each of the 4 bins we load the two direction-selected neighbor
/// lanes as shifted contiguous windows (the same shape as the Sobel kernel in
/// §4.9), then sum them against per-lane 0/1 masks that pick the correct bin.
/// The asymmetric `>` / `>=` compare is then a single SIMD compare, and the
/// output is written via a mask `blend`.
///
/// Precondition: `s >= 1`, `s + 8 <= w - 1`, and `1 <= y <= h - 2` — i.e. the
/// 8-pixel block plus every reachable 3×3 neighbor lies strictly inside the
/// image, so no boundary clamping is needed on any lane.
fn nms_simd_block(
    gradient: &GradientMap,
    out: &mut [f32],
    w: usize,
    y: usize,
    s: usize,
    row_off: usize,
) {
    let magnitude = &gradient.magnitude;
    let angle = &gradient.angle;
    let base = y * w;
    let prev_row = (y - 1) * w;
    let next_row = (y + 1) * w;

    // Per-lane 0/1 masks selecting which of the 4 bins each lane belongs to.
    let mut m = [[0.0f32; 8]; 4];
    let mut mag_arr = [0.0f32; 8];
    for i in 0..8 {
        let idx = base + s + i;
        mag_arr[i] = magnitude[idx];
        m[nms_bin_index(angle[idx])][i] = 1.0;
    }
    let mask0 = f32x8::new(m[0]);
    let mask1 = f32x8::new(m[1]);
    let mask2 = f32x8::new(m[2]);
    let mask3 = f32x8::new(m[3]);
    let mag = f32x8::new(mag_arr);

    // Direction-selected neighbor windows for each of the 4 bins.
    //   bin0 (horizontal): a = (x-1, y), b = (x+1, y)
    //   bin1 (45°):        a = (x+1, y+1), b = (x-1, y-1)
    //   bin2 (vertical):   a = (x, y-1),   b = (x, y+1)
    //   bin3 (135°):       a = (x+1, y-1), b = (x-1, y+1)
    let bin0_na = load8(magnitude, base + s - 1);
    let bin0_nb = load8(magnitude, base + s + 1);
    let bin1_na = load8(magnitude, next_row + s + 1);
    let bin1_nb = load8(magnitude, prev_row + s - 1);
    let bin2_na = load8(magnitude, prev_row + s);
    let bin2_nb = load8(magnitude, next_row + s);
    let bin3_na = load8(magnitude, prev_row + s + 1);
    let bin3_nb = load8(magnitude, next_row + s - 1);

    // SIMD select across the four bins via the per-lane masks.
    let na = bin0_na * mask0 + bin1_na * mask1 + bin2_na * mask2 + bin3_na * mask3;
    let nb = bin0_nb * mask0 + bin1_nb * mask1 + bin2_nb * mask2 + bin3_nb * mask3;

    // Asymmetric compare, identical semantics to the scalar kernel.
    let keep = mag.cmp_gt(na) & mag.cmp_ge(nb);
    let res = keep.blend(mag, f32x8::ZERO);
    let res_arr: [f32; 8] = res.into();
    for i in 0..8 {
        out[row_off + s + i] = res_arr[i];
    }
}

/// SIMD-accelerated Non-Maximum Suppression (§4.10).
///
/// Rayon row-band-sharded, with SIMD (`wide::f32x8`) processing 8 interior
/// pixels per iteration. Boundary rows (top/bottom) and the column blocks
/// touching the left/right image edge, plus the non-multiple-of-8 tail, fall
/// back to the exact scalar kernel — so the output is bit-identical to the
/// Phase 2 reference. Differentially tested in `tests/simd_nms_tests.rs`.
pub fn non_max_suppress_simd(gradient: &GradientMap) -> Vec<f32> {
    let w = gradient.width as usize;
    let h = gradient.height as usize;
    let wi = w as i32;
    let hi = h as i32;
    let mut out = vec![0.0f32; gradient.magnitude.len()];
    let band_px = BAND_ROWS * w;

    out.par_chunks_mut(band_px)
        .enumerate()
        .for_each(|(band_idx, band)| {
            let row_start = band_idx * BAND_ROWS;
            let row_end = (row_start + BAND_ROWS).min(h);
            for y in row_start..row_end {
                let row_off = (y - row_start) * w;

                if y < 1 || y + 1 >= h {
                    // Boundary row: neighbors reach outside the image, so clamp
                    // via the scalar path for the whole row.
                    for x in 0..w {
                        nms_scalar_pixel(gradient, wi, hi, x as i32, y as i32, band, row_off);
                    }
                    continue;
                }

                // Interior row: column 0 always needs left-edge clamping.
                nms_scalar_pixel(gradient, wi, hi, 0, y as i32, band, row_off);

                // SIMD blocks while the block + its neighbors stay interior.
                let mut s = 1usize;
                while s + 8 <= w - 1 {
                    nms_simd_block(gradient, band, w, y, s, row_off);
                    s += 8;
                }

                // Scalar tail (right edge + non-multiple-of-8 remainder).
                while s < w {
                    nms_scalar_pixel(gradient, wi, hi, s as i32, y as i32, band, row_off);
                    s += 1;
                }
            }
        });

    out
}

// ---------------------------------------------------------------------------
// §4.8 — SIMD box-filter / luma summation helpers
// ---------------------------------------------------------------------------

/// Sum a slice of `u8` bytes into `f32` using `wide::f32x8` SIMD lanes, with a
/// scalar tail for the remainder.
///
/// The scalar remainder calls plain `f32` addition — the *same* operation the
/// Phase 1 scalar box-filter already does, not a second hand-written
/// "mostly the same" implementation.  This is the general pattern for every
/// SIMD kernel in Phase 5 (§4.8 of the plan).
#[inline]
fn sum_row_simd(pixels: &[u8]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut chunks = pixels.chunks_exact(8);
    for chunk in &mut chunks {
        acc += f32x8::new([
            chunk[0] as f32, chunk[1] as f32, chunk[2] as f32, chunk[3] as f32,
            chunk[4] as f32, chunk[5] as f32, chunk[6] as f32, chunk[7] as f32,
        ]);
    }
    let mut total: f32 = acc.reduce_add();
    for &px in chunks.remainder() {
        total += px as f32;
    }
    total
}

/// SIMD-accelerated box-filter downsampling of raw YUV planes into per-cell
/// luma and RGB colors.
///
/// Identical algorithm to [`downsample_yuv_planes`] but uses `sum_row_simd`
/// (§4.8) for the per-row summation within each cell's source rectangle.
/// The remainder-tail calls plain scalar addition — the exact scalar path.
///
/// Differential tested against both the scalar [`downsample_yuv_planes`] and
/// the non-SIMD parallel [`downsample_yuv_planes_parallel`] in
/// `tests/simd_tests.rs`.
pub fn downsample_yuv_planes_simd(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    src_w: usize,
    src_h: usize,
    color_space: ColorSpace,
    cols: usize,
    rows: usize,
    cell_luma: &mut [f32],
    cell_color: &mut [(u8, u8, u8)],
) {
    if cols == 0 || rows == 0 || src_w == 0 || src_h == 0 {
        return;
    }
    debug_assert!(cell_luma.len() >= cols * rows);
    debug_assert!(cell_color.len() >= cols * rows);

    let luma_target = &mut cell_luma[..cols * rows];
    let color_target = &mut cell_color[..cols * rows];

    luma_target
        .par_chunks_mut(cols)
        .zip(color_target.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(row, (luma_row, color_row))| {
            for col in 0..cols {
                let x0 = col * src_w / cols;
                let x1 = ((col + 1) * src_w / cols).max(x0 + 1);
                let y0 = row * src_h / rows;
                let y1 = ((row + 1) * src_h / rows).max(y0 + 1);

                let cx0 = x0 / 2;
                let cx1 = (x1 + 1) / 2;
                let cy0 = y0 / 2;
                let cy1 = (y1 + 1) / 2;

                // SIMD-accelerated Y summation over the luma rect
                let mut y_acc = 0.0f32;
                let mut y_count = 0u32;
                for py in y0..y1 {
                    y_acc += sum_row_simd(&y_plane[py * src_w + x0..py * src_w + x1]);
                    y_count += (x1 - x0) as u32;
                }
                let avg_y = if y_count > 0 {
                    y_acc / y_count as f32
                } else {
                    0.0
                };

                // Chroma — same scalar fallback (very few elements per cell)
                let chroma_src_w = (src_w / 2).max(1);
                let mut u_acc = 0.0f32;
                let mut v_acc = 0.0f32;
                let mut c_count = 0u32;
                for py in cy0..cy1.min(src_h / 2) {
                    for px in cx0..cx1.min(src_w / 2) {
                        u_acc += u_plane[py * chroma_src_w + px] as f32;
                        v_acc += v_plane[py * chroma_src_w + px] as f32;
                        c_count += 1;
                    }
                }
                let (avg_u, avg_v) = if c_count > 0 {
                    (u_acc / c_count as f32, v_acc / c_count as f32)
                } else {
                    (128.0, 128.0)
                };

                luma_row[col] = avg_y;
                color_row[col] = yuv_to_rgb(avg_y, avg_u, avg_v, color_space);
            }
        });
}
