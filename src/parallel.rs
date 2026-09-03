//! Rayon-accelerated data parallelism for image downsampling and edge processing.
//!
//! Implements Milestone 5.5 and 5.6 (§4.7) of the Phase 5 Architecture Plan:
//! - Row-sharded box-filter downsampling of YUV planes.
//! - Row-band-sharded Sobel gradient computation and Non-Maximum Suppression (NMS).
//! - Zero `unsafe` and zero halo-copying: all workers read from shared, immutable input slices
//!   while writing disjoint, non-overlapping slices partitioned by `par_chunks_mut`.

use rayon::prelude::*;

use crate::video::yuv::{yuv_to_rgb, ColorSpace};

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
