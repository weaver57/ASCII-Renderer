//! Rayon-accelerated data parallelism for image downsampling and edge processing.
//!
//! Implements Milestone 5.5 and 5.6 (§4.7) of the Phase 5 Architecture Plan:
//! - Row-sharded box-filter downsampling of YUV planes.
//! - Row-band-sharded Sobel gradient computation and Non-Maximum Suppression (NMS).
//! - Zero `unsafe` and zero halo-copying: all workers read from shared, immutable input slices
//!   while writing disjoint, non-overlapping slices partitioned by `par_chunks_mut`.

use rayon::prelude::*;

use crate::render::edge::{nms_bin, GradientMap};
use crate::video::yuv::{yuv_to_rgb, ColorSpace};

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
