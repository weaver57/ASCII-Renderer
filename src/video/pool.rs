/// Pre-allocated buffers reused across every frame to avoid per-frame heap
/// allocations. This is the "we own pooling for buffers we derive" part of
/// design decision D6 — FFmpeg owns its own decoded-frame allocation, but
/// everything we compute on top of it gets pooled here.

use crate::render::edge::EdgeCellInfo;

pub struct FramePipelineBuffers {
    /// Full-resolution luma map (Y plane cast to f32).
    pub luma_map: Vec<f32>,
    /// Per-cell averaged luma values.
    pub cell_luma: Vec<f32>,
    /// Per-cell averaged RGB colors.
    pub cell_color: Vec<(u8, u8, u8)>,
    /// Per-cell edge info (None = brightness shading, Some = edge glyph).
    pub cell_edges: Vec<Option<EdgeCellInfo>>,
    /// Output byte buffer for ANSI terminal output.
    pub output_bytes: Vec<u8>,
    /// Last dimensions to detect when reallocation is needed.
    last_dims: Option<(u32, u32, usize, usize)>,
}

impl FramePipelineBuffers {
    pub fn new() -> Self {
        Self {
            luma_map: Vec::new(),
            cell_luma: Vec::new(),
            cell_color: Vec::new(),
            cell_edges: Vec::new(),
            output_bytes: Vec::new(),
            last_dims: None,
        }
    }

    /// Ensure all buffers are large enough for the given dimensions.
    /// Only reallocates when dimensions actually change.
    pub fn ensure_capacity(&mut self, src_w: u32, src_h: u32, cols: usize, rows: usize) {
        let new_dims = (src_w, src_h, cols, rows);
        if self.last_dims == Some(new_dims) {
            return;
        }
        self.last_dims = Some(new_dims);

        let src_pixels = src_w as usize * src_h as usize;
        let cells = cols * rows;

        self.luma_map.resize(src_pixels, 0.0);
        self.cell_luma.resize(cells, 0.0);
        self.cell_color.resize(cells, (0, 0, 0));
        self.cell_edges.resize(cells, None);
        // Output buffer: generous pre-allocation (roughly 40 bytes per cell)
        self.output_bytes
            .reserve(cells.saturating_sub(self.output_bytes.capacity()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_new_is_empty() {
        let pool = FramePipelineBuffers::new();
        assert!(pool.luma_map.is_empty());
        assert!(pool.cell_luma.is_empty());
        assert!(pool.last_dims.is_none());
    }

    #[test]
    fn test_pool_ensure_capacity() {
        let mut pool = FramePipelineBuffers::new();
        pool.ensure_capacity(100, 100, 10, 10);
        assert_eq!(pool.luma_map.len(), 10000);
        assert_eq!(pool.cell_luma.len(), 100);
        assert_eq!(pool.cell_color.len(), 100);
        assert_eq!(pool.cell_edges.len(), 100);
        assert_eq!(pool.last_dims, Some((100, 100, 10, 10)));
    }

    #[test]
    fn test_pool_same_dims_no_realloc() {
        let mut pool = FramePipelineBuffers::new();
        pool.ensure_capacity(100, 100, 10, 10);
        let cap_before = pool.luma_map.capacity();
        // Same dims → no realloc
        pool.ensure_capacity(100, 100, 10, 10);
        assert_eq!(pool.luma_map.capacity(), cap_before);
    }

    #[test]
    fn test_pool_different_dims_realloc() {
        let mut pool = FramePipelineBuffers::new();
        pool.ensure_capacity(100, 100, 10, 10);
        pool.ensure_capacity(200, 200, 20, 20);
        assert_eq!(pool.luma_map.len(), 40000);
        assert_eq!(pool.cell_luma.len(), 400);
    }
}
