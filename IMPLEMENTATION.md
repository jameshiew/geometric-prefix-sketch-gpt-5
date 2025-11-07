# Implementation Notes for Geometric Prefix Sketch

This document bridges `DESIGN.md` with the current Rust implementation. It explains
which parts of the design are implemented, what compromises were made, and why.

## High-level Architecture

The crate exposes a single public type, [`GpsSketch`](src/sketch.rs), layered on
three modules:

- `util`: deterministic hashing (XXH3) and inclusion-probability helpers.
- `tree`: compressed radix trie (nodes, edges, heavy-hitter SpaceSaving, merge
  helpers). Most data-structure heavy lifting lives here.
- `sketch`: user-facing API, sampling, deterministic merge wiring, and tests.

This separation mirrors DESIGN.md’s “implementation sketch” section but keeps the
public API surface minimal.

## Design elements and their status

| DESIGN element | Status | Notes |
| -------------- | ------ | ----- |
| Geometric depth sampling (Horvitz–Thompson rescaling) | ✅ Implemented. | `GpsSketch::add` samples a max depth using XXH3-derived uniform bits and `sample_level_for_hash`; queries rescale by `alpha^(depth-1)`.
| Deterministic hashing | ✅ Implemented. | `util::deterministic_hash` uses `xxh3_128_with_seed` so shards merge safely.
| Compressed radix trie with mid-edge mass | ✅ Implemented (see `tree.rs`). | Nodes hold `sum` plus edges with `label` and `mid_sums`. Node promotion depth currently hard-coded at 4 (`PROMOTION_DEPTH`).
| Heavy-hitter sketches per node | ✅ Optional. | `SpaceSaving` stores top suffixes; `GpsSketch::with_heavy_hitters` toggles them. Implementation keeps a `HashMap<Vec<u8>, usize>` index for `O(1)` updates.
| Merge via deterministic sampling | ✅ Implemented structurally. | `tree::merge_nodes` walks both tries, summing nodes/edges in place instead of replaying inserts. HH sketches are merged entry-wise.
| Pruning low-signal prefixes | ✅ `prune_by_estimate`. | Recurses through the trie, dropping subtrees once their scaled estimate falls below threshold.
| Examples/benchmarks | ✅ `examples/` + Criterion benches. | Provide runnable demos and performance harnesses.
| Accuracy/integration testing | ✅ `tests/accuracy.rs`. | Ensures estimates align with exact counts on random data (with a tolerance for deep prefixes).

## Not-yet/Partially implemented design ideas

- **Configurable promotion depth:** currently a constant (`PROMOTION_DEPTH = 4`). DESIGN.md suggests adapting it; this is future work.
- **Alternate alphabets / arenas:** the trie stores `Vec<u8>` for edge labels and allocates per insertion. Switching to arenas or slices would reduce fragmentation but adds complexity.
- **Advanced heavy-hitter logic (e.g., Count-Min per node):** the current SpaceSaving is simple but sufficient for small `k`.
- **Precision safeguards:** for very deep prefixes (depth > ~40) the inclusion probability underflows toward `f64::MIN_POSITIVE`. DESIGN.md hints at using arbitrary precision or per-depth scaling tables; presently we clamp to `MIN_POSITIVE`, which inflates variance but keeps numbers finite.

## Justifications / trade-offs

### 1. Compressing edges but keeping early promotion
The design recommends a “compressed radix node.” We adopt a hybrid: for the
first `PROMOTION_DEPTH` bytes, every character gets its own node (fast lookups,
precise control at shallow depths). Beyond that point, we store compressed edge
labels with `mid_sums`. Rationale:

- Depth 1–4 prefixes are common reporting surfaces; keeping them explicit
  preserves exactness and fast `contains_prefix` checks.
- Compression beyond depth 4 dramatically reduces node count on path-heavy
  datasets (URLs, file paths).

### 2. Deterministic hashing via XXH3
DESIGN.md left hashing “implementation-specific.” We chose XXH3 (128-bit) for:

- High avalanche quality; minimal bias in sampling levels.
- Efficient pure-Rust implementation via `xxhash-rust`.
- Seed control for shard merge compatibility.

### 3. SpaceSaving index map
Original DESIGN only said “tiny heavy-hitter sketch”. Instead of a multi-row
Count-Min, we use classic SpaceSaving with an auxiliary `HashMap` to keep
updates `O(1)` despite storing arbitrary byte suffixes. This trades tiny extra
memory (one hash entry per tracked suffix) for predictable speed.

### 4. Merge strategy
The first implementation replayed every prefix via `add_raw_sum`, which was
O(n log n). The current version merges tries structurally:

- Node sums are added directly.
- Mid-edge accumulators are summed when labels match.
- Children are cloned when unique, or recursed into when shared.

This keeps merge cost linear in the number of realized prefixes and aligns with
DESIGN.md’s “merge by addition” statement.

### 5. Accuracy guardrails
DESIGN.md touts unbiased estimators but doesn’t specify testing methodology. We
added `tests/accuracy.rs`, which:

- Builds a large sketch from random keys.
- Tracks exact counts for shallow prefixes.
- Asserts that depth-1 prefixes are exact (because `alpha=0.5` ⇒ `q(1)=1`).
- Enforces <15% relative error for high-support prefixes (≥200 true count).

This ensures probabilistic behavior matches theory under a representative
workload.

### 6. Examples & documentation
To make the crate usable without reading the entire design paper, we added:

- `IMPLEMENTATION.md` (this doc) and `examples/basic.rs` / `examples/heavy_hitters.rs`.
- Crate-level doc comment with runnable snippet.
- Extensive rustdoc on each public method (detailing scaling, merges, etc.).

## Running the project

- `cargo test` – runs unit, doctest, and integration suites.
- `cargo bench` – Criterion benchmarks for add/estimate/merge/top-k/prune.
- `cargo run --example basic` – small end-to-end demo.
- `cargo run --example heavy_hitters` – heavy-hitter query sample.

## Future Work

1. Configurable promotion depth & arena allocation.
2. More numerically stable inclusion probabilities (per-depth table).
3. Alternative heavy-hitter sketches (Count-Min, Misra-Gries) plug-ins.
4. Streaming/backpressure interface for pruning/compaction.

For now, this implementation matches the majority of DESIGN.md’s promises and
keeps the public API ergonomic for application developers.
