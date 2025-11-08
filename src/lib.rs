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
//! // Pick a shared seed when you plan to merge sketches across shards.
//! let mut sketch = GpsSketch::with_seed(0.5, 42);
//! sketch.add("/api/v1/users", 1.0);
//! sketch.add("/api/v1/orders", 1.0);
//! sketch.add("/api/v2/users", 1.0);
//!
//! // Estimate traffic for a top-level prefix (depth 1 is exact).
//! let root_calls = sketch.estimate("/");
//! assert_eq!(root_calls, 3.0);
//!
//! // Merge sketches built on different shards (same alpha/seed).
//! let mut shard = GpsSketch::with_seed(0.5, 42);
//! shard.add("/api/v1/users", 1.0);
//! sketch.merge_from(&shard);
//! ```
//!
//! ## Seeding & mergeability
//!
//! - **Default:** [`GpsSketch::default`] now uses a random seed so adversaries
//!   cannot predict sampling depths. Use this when you **do not** need to merge
//!   sketches across shards or processes.
//! - **Merging:** to merge sketches, every shard must share the same
//!   `(alpha, seed[, heavy_hitters_capacity])` when constructing sketches:
//!   ```rust
//!   # use geometric_prefix_sketch::GpsSketch;
//!   let mut a = GpsSketch::with_seed(0.5, 12345);
//!   let mut b = GpsSketch::with_seed(0.5, 12345);
//!   // ... ingest on each ...
//!   a.merge_from(&b);
//!   ```
//! - [`GpsSketch::new`] remains for compatibility and uses a fixed seed `0`.
//!   Prefer [`with_random_seed`](GpsSketch::with_random_seed) for standalone use
//!   and [`with_seed`](GpsSketch::with_seed) for deterministic merging.
//!
//! See [`GpsSketch`] for the full API including heavy-hitter tracking, pruning,
//! and prefix iteration utilities.
mod sketch;
mod tree;
mod util;

pub use sketch::{GpsSketch, MergeError};
