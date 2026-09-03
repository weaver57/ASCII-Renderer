/// YUV420P frame handling: de-striding, range expansion, YUV→RGB conversion.
///
/// Phase 3's headline optimization: the Y plane is the luma map directly — no
/// RGB conversion in the hot path. Chroma→RGB conversion happens once per
/// character cell, not per pixel.

/// Whether the video uses limited (TV) or full (PC) color range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    Limited,
    Full,
}

/// Whether the video uses BT.601 or BT.709 color space coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Bt601,
    Bt709,
}

/// A decoded YUV420P frame with de-strided, tightly-packed planes.
#[derive(Debug, Clone)]
pub struct YuvFrame {
    pub width: u32,
    pub height: u32,
    /// De-strided luma plane, `width * height` bytes, full-range after expansion.
    pub y: Vec<u8>,
    /// De-strided Cb (U) plane, `(width/2) * (height/2)` bytes, full-range.
    pub u: Vec<u8>,
    /// De-strided Cr (V) plane, `(width/2) * (height/2)` bytes, full-range.
    pub v: Vec<u8>,
    pub range: ColorRange,
    pub color_space: ColorSpace,
    /// Presentation timestamp in seconds, if available from the container.
    pub pts_seconds: Option<f64>,
}

/// Copies a possibly-padded plane into a tightly packed buffer.
///
/// FFmpeg frame buffers are typically padded per-row (`stride`) to a
/// SIMD-friendly alignment boundary, which is usually *larger* than the actual
/// pixel width. Every downstream algorithm (Sobel, NMS, box filter) assumes
/// `row_pitch == width`, so we strip the padding here at the boundary.
pub fn destride_plane(src: &[u8], src_stride: usize, width: usize, height: usize, dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), width * height);
    for row in 0..height {
        let src_start = row * src_stride;
        let dst_start = row * width;
        dst[dst_start..dst_start + width].copy_from_slice(&src[src_start..src_start + width]);
    }
}

/// Expand a single value from limited range (16–235) to full range (0–255).
///
/// Most video content stores luma as "limited range" — black is 16, white is
/// 235, not 0/255. Treating limited as full makes blacks look gray and
/// reduces contrast in the brightness→character mapping.
#[inline]
pub fn expand_limited_range(value: u8) -> u8 {
    (((value as i32 - 16) * 255) / (235 - 16)).clamp(0, 255) as u8
}

/// Expand an entire plane from limited to full range in-place.
pub fn expand_plane_limited(plane: &mut [u8]) {
    for v in plane.iter_mut() {
        *v = expand_limited_range(*v);
    }
}

/// Detect color space from stream metadata or heuristic.
///
/// If the container doesn't specify, use the standard heuristic:
/// height ≥ 720 → BT.709, otherwise BT.601 (same default as browsers).
pub fn detect_color_space(height: u32, stream_color_space: Option<&str>) -> ColorSpace {
    if let Some(cs) = stream_color_space {
        let lower = cs.to_ascii_lowercase();
        if lower.contains("bt709") || lower.contains("bt.709") {
            return ColorSpace::Bt709;
        }
        // smpte170m, bt470bg, bt470, bt601, etc. → BT.601
        // Unrecognized strings → BT.601 (safe default for SD-era content)
        return ColorSpace::Bt601;
    }
    if height >= 720 {
        ColorSpace::Bt709
    } else {
        ColorSpace::Bt601
    }
}

/// Convert a single (Y, U, V) triple from full-range YUV to RGB.
///
/// Inputs must be full-range (post range expansion). Clamping is essential —
/// the linear formulas can produce slightly out-of-range values near saturated
/// colors, and an unclamped cast to `u8` would silently wrap instead of clip.
#[inline]
pub fn yuv_to_rgb(y: f32, u: f32, v: f32, space: ColorSpace) -> (u8, u8, u8) {
    let cb = u - 128.0;
    let cr = v - 128.0;
    let (r, g, b) = match space {
        ColorSpace::Bt601 => (
            y + 1.402 * cr,
            y - 0.344136 * cb - 0.714136 * cr,
            y + 1.772 * cb,
        ),
        ColorSpace::Bt709 => (
            y + 1.5748 * cr,
            y - 0.1873 * cb - 0.4681 * cr,
            y + 1.8556 * cb,
        ),
    };
    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Build a full-resolution luma map directly from the Y plane.
///
/// This is D3 made concrete: no `luma()` call, no RGB anywhere. The Y bytes
/// are already perceptual luma (Rec. 709 coefficients are baked into the
/// YCbCr encoding itself), so we just cast to f32.
pub fn build_luma_map_y(y_plane: &[u8], dst: &mut Vec<f32>) {
    dst.clear();
    dst.extend(y_plane.iter().map(|&v| v as f32));
}

/// Downsample raw Y, U, V plane slices directly into pre-allocated per-cell luma & color buffers.
pub fn downsample_yuv_planes(
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
    debug_assert!(cell_luma.len() >= cols * rows);
    debug_assert!(cell_color.len() >= cols * rows);

    for row in 0..rows {
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

            let idx = row * cols + col;
            cell_luma[idx] = avg_y;
            cell_color[idx] = yuv_to_rgb(avg_y, avg_u, avg_v, color_space);
        }
    }
}

/// Downsample a YUV420P frame to per-cell (luma, RGB color).
///
/// For each character cell:
/// 1. Box-filter average Y over the luma-space source rect → cell luma.
/// 2. Box-filter average U, V over the corresponding chroma-space rect (half res).
/// 3. Convert the single averaged (Y, U, V) triple to RGB — once per cell.
///
/// This defers chroma→RGB conversion to after per-cell averaging (D4), yielding
/// a ~778× reduction in color-space conversions at 1080p→160×90 grid.
pub fn downsample_yuv(
    frame: &YuvFrame,
    cols: usize,
    rows: usize,
) -> (Vec<f32>, Vec<(u8, u8, u8)>) {
    let mut cell_luma = vec![0.0f32; cols * rows];
    let mut cell_color = vec![(0u8, 0u8, 0u8); cols * rows];
    downsample_yuv_planes(
        &frame.y,
        &frame.u,
        &frame.v,
        frame.width as usize,
        frame.height as usize,
        frame.color_space,
        cols,
        rows,
        &mut cell_luma,
        &mut cell_color,
    );
    (cell_luma, cell_color)
}

/// Create a YuvFrame from raw FFmpeg YUV420P output.
///
/// `raw` contains the concatenated Y, U, V planes with potential row padding.
/// `y_stride`, `u_stride`, `v_stride` are the FFmpeg-reported line sizes.
pub fn create_yuv_frame(
    raw: &[u8],
    width: u32,
    height: u32,
    y_stride: usize,
    u_stride: usize,
    v_stride: usize,
    range: ColorRange,
    color_space: ColorSpace,
    pts_seconds: Option<f64>,
) -> YuvFrame {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = w / 2;
    let chroma_h = h / 2;

    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; chroma_w * chroma_h];
    let mut v = vec![0u8; chroma_w * chroma_h];

    // Calculate offsets into the raw buffer
    let y_offset = 0;
    let u_offset = y_stride * h;
    let v_offset = u_offset + u_stride * chroma_h;

    // De-stride each plane
    if raw.len() >= v_offset + v_stride * chroma_h {
        destride_plane(&raw[y_offset..], y_stride, w, h, &mut y);
        destride_plane(&raw[u_offset..], u_stride, chroma_w, chroma_h, &mut u);
        destride_plane(&raw[v_offset..], v_stride, chroma_w, chroma_h, &mut v);
    }

    // Expand limited range to full range in-place
    if range == ColorRange::Limited {
        expand_plane_limited(&mut y);
        expand_plane_limited(&mut u);
        expand_plane_limited(&mut v);
    }

    YuvFrame {
        width,
        height,
        y,
        u,
        v,
        range,
        color_space,
        pts_seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_limited_range_black() {
        assert_eq!(expand_limited_range(16), 0);
    }

    #[test]
    fn test_expand_limited_range_white() {
        assert_eq!(expand_limited_range(235), 255);
    }

    #[test]
    fn test_expand_limited_range_midpoint() {
        // Midpoint of 16..235 is 125.5, maps to ~127 in full range
        let expanded = expand_limited_range(125);
        assert!((126..=129).contains(&expanded), "got {}", expanded);
    }

    #[test]
    fn test_expand_limited_range_clamps_low() {
        assert_eq!(expand_limited_range(0), 0);
    }

    #[test]
    fn test_expand_limited_range_clamps_high() {
        assert_eq!(expand_limited_range(255), 255);
    }

    #[test]
    fn test_yuv_to_rgb_black() {
        // Y=0, U=128, V=128 → should be black (all near 0)
        let (r, g, b) = yuv_to_rgb(0.0, 128.0, 128.0, ColorSpace::Bt601);
        assert!(r < 10 && g < 10 && b < 10, "black: ({}, {}, {})", r, g, b);
    }

    #[test]
    fn test_yuv_to_rgb_white() {
        // Y=255, U=128, V=128 → should be white (all near 255)
        let (r, g, b) = yuv_to_rgb(255.0, 128.0, 128.0, ColorSpace::Bt601);
        assert!(r > 245 && g > 245 && b > 245, "white: ({}, {}, {})", r, g, b);
    }

    #[test]
    fn test_yuv_to_rgb_saturated_red_bt601() {
        // Approximate red in YUV: Y≈76, U≈84, V≈255 (BT.601)
        let (r, g, b) = yuv_to_rgb(76.0, 84.0, 255.0, ColorSpace::Bt601);
        assert!(r > 230, "red channel should be high: {}", r);
        assert!(g < 50, "green channel should be low: {}", g);
        assert!(b < 80, "blue channel should be low: {}", b);
    }

    #[test]
    fn test_yuv_to_rgb_clamping() {
        // Feed values that would mathematically produce >255
        let (r, g, b) = yuv_to_rgb(255.0, 0.0, 255.0, ColorSpace::Bt601);
        assert!(r <= 255, "must clamp, got {}", r);
        assert!(g <= 255, "must clamp, got {}", g);
        assert!(b <= 255, "must clamp, got {}", b);
    }

    #[test]
    fn test_destride_plane() {
        let src_stride = 16;
        let width = 10;
        let height = 3;
        // Create source with stride > width, padding bytes = 0xFF
        let mut src = vec![0u8; src_stride * height];
        for row in 0..height {
            for col in 0..width {
                src[row * src_stride + col] = (row * width + col) as u8;
            }
            // Padding bytes
            for col in width..src_stride {
                src[row * src_stride + col] = 0xFF;
            }
        }

        let mut dst = vec![0u8; width * height];
        destride_plane(&src, src_stride, width, height, &mut dst);

        // Verify tightly packed output
        for row in 0..height {
            for col in 0..width {
                let expected = (row * width + col) as u8;
                assert_eq!(
                    dst[row * width + col],
                    expected,
                    "mismatch at ({}, {})",
                    row,
                    col
                );
            }
        }
        // No 0xFF sentinel bytes should leak through
        assert!(!dst.contains(&0xFF));
    }

    #[test]
    fn test_destride_single_row() {
        let src_stride = 20;
        let width = 5;
        let mut src = vec![0u8; src_stride];
        for i in 0..width {
            src[i] = (i + 10) as u8;
        }
        let mut dst = vec![0u8; width];
        destride_plane(&src, src_stride, width, 1, &mut dst);
        assert_eq!(&dst, &[10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_create_yuv_frame() {
        let width = 4u32;
        let height = 4u32;
        let y_stride = 8; // padded
        let u_stride = 4;
        let v_stride = 4;

        // Build raw YUV420P buffer with padding
        let mut raw = vec![0u8; y_stride * height as usize + u_stride * 2 + v_stride * 2];

        // Fill Y plane with value 128
        for row in 0..height as usize {
            for col in 0..width as usize {
                raw[row * y_stride + col] = 128;
            }
        }
        // Fill U plane (offset = y_stride * height)
        let u_off = y_stride * height as usize;
        for row in 0..2usize {
            for col in 0..2usize {
                raw[u_off + row * u_stride + col] = 128;
            }
        }
        // Fill V plane
        let v_off = u_off + u_stride * 2;
        for row in 0..2usize {
            for col in 0..2usize {
                raw[v_off + row * v_stride + col] = 128;
            }
        }

        let frame = create_yuv_frame(
            &raw, width, height, y_stride, u_stride, v_stride,
            ColorRange::Limited, ColorSpace::Bt601, None,
        );

        // Y=128 limited → expanded ≈ 127; U=V=128 neutral
        assert_eq!(frame.y.len(), 16);
        assert_eq!(frame.u.len(), 4);
        assert_eq!(frame.v.len(), 4);
        // All Y values should be expanded from limited 128 → ~127
        for &y_val in &frame.y {
            assert!((129..=131).contains(&y_val), "Y expanded to {}", y_val);
        }
    }

    #[test]
    fn test_detect_color_space() {
        // No stream metadata → height heuristic
        assert_eq!(detect_color_space(480, None), ColorSpace::Bt601);
        assert_eq!(detect_color_space(720, None), ColorSpace::Bt709);
        assert_eq!(detect_color_space(1080, None), ColorSpace::Bt709);
        // Explicit BT.709 strings → BT.709
        assert_eq!(detect_color_space(480, Some("bt709")), ColorSpace::Bt709);
        assert_eq!(detect_color_space(480, Some("BT.709")), ColorSpace::Bt709);
        assert_eq!(detect_color_space(480, Some(" bt709 ")), ColorSpace::Bt709);
        // BT.601-related strings → BT.601 (NOT BT.709)
        assert_eq!(detect_color_space(1080, Some("smpte170m")), ColorSpace::Bt601);
        assert_eq!(detect_color_space(1080, Some("bt601")), ColorSpace::Bt601);
        assert_eq!(detect_color_space(1080, Some("bt470bg")), ColorSpace::Bt601);
        // Unrecognized string → BT.601 (safe default)
        assert_eq!(detect_color_space(1080, Some("unknown_cs")), ColorSpace::Bt601);
    }

    #[test]
    fn test_build_luma_map_y() {
        let y_plane = vec![0u8, 64, 128, 192, 255];
        let mut dst = Vec::new();
        build_luma_map_y(&y_plane, &mut dst);
        assert_eq!(dst.len(), 5);
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[1], 64.0);
        assert_eq!(dst[2], 128.0);
        assert_eq!(dst[3], 192.0);
        assert_eq!(dst[4], 255.0);
    }

    #[test]
    fn test_downsample_yuv_single_cell() {
        // 4x4 YUV420P frame downsampled to 1x1 grid
        let frame = YuvFrame {
            width: 4,
            height: 4,
            y: vec![128u8; 16],
            u: vec![128u8; 4],
            v: vec![128u8; 4],
            range: ColorRange::Full,
            color_space: ColorSpace::Bt601,
            pts_seconds: None,
        };
        let (cell_luma, cell_color) = downsample_yuv(&frame, 1, 1);
        assert_eq!(cell_luma.len(), 1);
        assert_eq!(cell_color.len(), 1);
        assert!((cell_luma[0] - 128.0).abs() < 0.01);
        // Y=128, U=128, V=128 → neutral gray
        let (r, g, b) = cell_color[0];
        assert!((r as i32 - 128).abs() < 10, "gray r={}", r);
        assert!((g as i32 - 128).abs() < 10, "gray g={}", g);
        assert!((b as i32 - 128).abs() < 10, "gray b={}", b);
    }

    #[test]
    fn test_downsample_yuv_2x2_grid() {
        // 4x4 frame → 2x2 grid, each cell covers 2x2 luma pixels
        let mut y = vec![0u8; 16];
        // Top-left cell: Y=50
        y[0] = 50; y[1] = 50; y[4] = 50; y[5] = 50;
        // Top-right cell: Y=200
        y[2] = 200; y[3] = 200; y[6] = 200; y[7] = 200;
        // Bottom-left cell: Y=100
        y[8] = 100; y[9] = 100; y[12] = 100; y[13] = 100;
        // Bottom-right cell: Y=150
        y[10] = 150; y[11] = 150; y[14] = 150; y[15] = 150;

        let frame = YuvFrame {
            width: 4,
            height: 4,
            y,
            u: vec![128u8; 4],
            v: vec![128u8; 4],
            range: ColorRange::Full,
            color_space: ColorSpace::Bt601,
            pts_seconds: None,
        };

        let (cell_luma, _) = downsample_yuv(&frame, 2, 2);
        assert_eq!(cell_luma[0], 50.0); // top-left
        assert_eq!(cell_luma[1], 200.0); // top-right
        assert_eq!(cell_luma[2], 100.0); // bottom-left
        assert_eq!(cell_luma[3], 150.0); // bottom-right
    }
}
