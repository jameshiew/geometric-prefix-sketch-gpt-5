//! # Geometric Prefix Sketch
//!
//! `geometric-prefix-sketch` implements the structure described in
//! `DESIGN.md`: an unbiased, merge-friendly prefix sketch that touches only a
//! geometrically sampled number of trie prefixes per update. Each update costs
//! `O(1)` expected time while preserving exactness at shallow depths and a
//! controllable error curve deeper in the trie.
//!
//! ## Quick start
//!
//! ```rust
//! use geometric_prefix_sketch::GpsSketch;
//!
//! let mut sketch = GpsSketch::default();
//! sketch.add("/api/v1/users", 1.0);
//! sketch.add("/api/v1/orders", 1.0);
//! sketch.add("/api/v2/users", 1.0);
//!
//! // Estimate traffic for a top-level prefix (depth 1 is exact).
//! let root_calls = sketch.estimate("/");
//! assert_eq!(root_calls, 3.0);
//!
//! // Merge sketches built on different shards (same alpha/seed).
//! let mut shard = GpsSketch::default();
//! shard.add("/api/v1/users", 1.0);
//! sketch.merge_from(&shard);
//! ```
//!
//! See [`GpsSketch`] for the full API including heavy-hitter tracking, pruning,
//! and prefix iteration utilities.
mod sketch;
mod tree;
mod util;

pub use sketch::GpsSketch;
