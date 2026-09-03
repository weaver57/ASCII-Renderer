pub mod ascii;
pub mod edge;
pub mod grid;
pub mod luminance;
pub mod temporal;
pub mod ramp;

pub use ascii::{AsciiRenderer, ColorMode, BLOCK_RAMP, DETAILED_RAMP, SHORT_RAMP};
pub use edge::{EdgeCellInfo, direction_to_char, compute_frame_edges, compute_frame_edges_from_luma};
// This is the library's public grid-type surface. The bin crate (a separate
// compilation unit over these same files) only names `build_char_grid_into`,
// so the value types warn there as unused — they are intentionally exported
// for library consumers, not dead imports.
#[allow(unused_imports)]
pub use grid::{CharCell, CharGrid, Rgb, SENTINEL_CELL, build_char_grid_into};
pub use ramp::RAMP;
pub use temporal::TemporalEdgeSmoother;
