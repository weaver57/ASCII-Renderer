pub mod ascii;
pub mod edge;
pub mod luminance;
pub mod temporal;
pub mod ramp;

pub use ascii::{AsciiRenderer, ColorMode, BLOCK_RAMP, DETAILED_RAMP, SHORT_RAMP};
pub use edge::{EdgeCellInfo, direction_to_char, compute_frame_edges, compute_frame_edges_from_luma, aggregate_cell_edges, circular_lerp_deg, normalize_deg};
pub use ramp::RAMP;
pub use temporal::TemporalEdgeSmoother;
