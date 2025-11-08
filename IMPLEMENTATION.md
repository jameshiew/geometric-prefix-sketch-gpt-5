# Implementation Notes for Geometric Prefix Sketch

This document bridges `DESIGN.md` with the current Rust implementation. It explains
which parts of the design are implemented, what compromises were made, and why.

## High-level Architecture

The crate exposes a single public type, [`GpsSketch`](src/sketch.rs), layered on
three modules:

- `util`: deterministic hashing (XXH3) and inclusion-probability helpers.
- `tree`: compressed radix trie (nodes, edges, heavy-hitter top-k compactor summaries, merge
  helpers). Most data-structure heavy lifting lives here.
- `sketch`: user-facing API, sampling, deterministic merge wiring, and tests.

This separation mirrors DESIGN.md’s “implementation sketch” section but keeps the
public API surface minimal.

## Design elements and their status

| DESIGN element | Status | Notes |
| -------------- | ------ | ----- |
| Geometric depth sampling (Horvitz–Thompson rescaling) | ✅ Implemented. | `GpsSketch::add` samples a max depth using XXH3-128 leading-zero counts (`α=0.5` realizable depths stop at 129) and, for general `α`, a precomputed inclusion table capped at 200 000 entries where deeper levels are defined to have `q=0`; queries rescale by `alpha^(depth-1)`.
| Deterministic hashing | ✅ Implemented. | `util::deterministic_hash` uses `xxh3_128_with_seed` so shards merge safely. `GpsSketch::default()` now sources a secure random seed; use `with_seed(alpha, seed)` when deterministic merging is required.
| Compressed radix trie with mid-edge mass | ✅ Implemented (see `tree.rs`). | Nodes hold `sum` plus edges with `label` and `mid_sums`. Node promotion depth currently hard-coded at 4 (`PROMOTION_DEPTH`).
| Heavy-hitter sketches per node | ✅ Optional. | A truncate-to-capacity top-k compactor stores the heaviest suffixes; `GpsSketch::with_heavy_hitters` toggles them. Implementation keeps a `HashMap<Vec<u8>, usize>` index for `O(1)` updates and merges summaries by summing shared suffixes then truncating back to capacity. Only positive deltas feed the HH stream so completions never inherit negative mass.
| Merge via deterministic sampling | ✅ Implemented structurally. | `tree::merge_nodes` walks both tries, summing nodes/edges in place instead of replaying inserts. Heavy hitters reuse the same sum-then-truncate compactor merge so results are order-invariant. Callers must still match `alpha`, `hash_seed`, and HH capacity (the code `assert!`s every merge).
| Pruning low-signal prefixes | ✅ `prune_by_estimate`. | Recurses through the trie, dropping subtrees once their scaled estimate falls below threshold.
| Examples/benchmarks | ✅ `examples/` + Criterion benches. | Provide runnable demos and performance harnesses.
| Accuracy/integration testing | ✅ `tests/accuracy.rs`. | Ensures estimates align with exact counts on random data (with a tolerance for deep prefixes).

## Not-yet/Partially implemented design ideas

- **Configurable promotion depth:** currently a constant (`PROMOTION_DEPTH = 4`). DESIGN.md suggests adapting it; this is future work.
- **Alternate alphabets / arenas:** the trie stores `Vec<u8>` for edge labels and allocates per insertion. Switching to arenas or slices would reduce fragmentation but adds complexity.
- **Advanced heavy-hitter logic (e.g., Count-Min per node):** the current truncate-to-`k` compactor is simple but sufficient for small `k`.
- **Precision safeguards:** for very deep prefixes (depth > ~40) the inclusion probability underflows toward `f64::MIN_POSITIVE`. DESIGN.md hints at using arbitrary precision or per-depth scaling tables; presently we clamp to `MIN_POSITIVE`, which inflates variance but keeps numbers finite, and we cap the precomputed inclusion table at 200 000 depths so high-`α` sketches cannot allocate unbounded memory (deeper levels are treated as `q = 0`).

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

### 2. Deterministic hashing via XXH3 (and sampler hygiene)
DESIGN.md left hashing “implementation-specific.” We chose XXH3 (128-bit) for:

- High avalanche quality; minimal bias in sampling levels.
- Efficient pure-Rust implementation via `xxhash-rust`.
- Seed control for shard merge compatibility.

When building shards independently, every sketch that will eventually merge
must share both `alpha` and the `hash_seed`. Pick a non-guessable seed whenever
adversarial keys might try to predict sampling depths; `GpsSketch::with_seed`
is the intended entry point for that configuration.

`GpsSketch::default()` now calls `with_random_seed`, which sources entropy via
`getrandom` to defend against adversarial key selection. Switch to
`with_seed(alpha, seed)` (and share that tuple across shards) whenever
mergeability matters.

Sampling now mirrors the design’s deterministic story exactly:

- For `α = 0.5` (the default), `sample_level_for_hash` uses the
  number of leading zeros in the 128-bit hash (`1 + clz(hash)`), capped by the
  key length and by the 128-bit limit (max realizable depth = 129). No floating
  point math, no branches. Extending beyond 129 via a streamed “extended
  sampler” is still on the future-work list.
- For general `α`, we precompute a monotone table of Q128 thresholds derived
  from `α^(ℓ-1)` so runtime sampling boils down to comparing `hash < q128[ℓ]`
  with zero floating-point work. The table itself is capped at 200 000 depths to
  keep memory bounded when `α → 1`; deeper levels are defined to have `q = 0`.
  Every shard therefore makes identical keep/drop decisions without depending on
  platform rounding quirks.

### 3. Top-k compactor bookkeeping & merges
Original DESIGN only said “tiny heavy-hitter sketch”. Instead of a multi-row
Count-Min, we keep a bounded top-k compactor backed by an auxiliary
`HashMap<Vec<u8>, usize>` so updates stay `O(1)` even for arbitrary byte suffixes.
Whenever the compactor grows to `2k` entries we select the `k` heaviest via
`select_nth_unstable` and truncate. During shard merges we sum shared suffixes,
append any new ones, then run the same truncate step, so merge order can’t
perturb the heavy-hitter winners even though no strict Misra-Gries guarantees
are claimed.

### 4. Merge strategy
The first implementation replayed every prefix via `add_raw_sum`, which was
O(n log n). The current version merges tries structurally and now handles
compressed-edge divergence correctly:

- Node sums are added directly.
- When two compressed edges share a partial prefix, the destination edge is
  split at the LCP so the shared bytes’ `mid_sums` and node sums can be added
  before recursing into the remainder.
- Children are cloned only when truly unique; otherwise we recurse into the
  shared node after rescaling the boundary accumulator.
- Per-node heavy hitters reuse the same sum-then-truncate compactor merge,
  keeping the HH summaries deterministic and order-invariant.

This keeps merge cost linear in the number of realized prefixes, guarantees the
resulting trie stays compressed/canonical, and aligns with DESIGN.md’s “merge by
addition” statement even when shards see diverging suffixes beyond
`PROMOTION_DEPTH`.

### 5. Accuracy guardrails
DESIGN.md touts unbiased estimators but doesn’t specify testing methodology. We
added `tests/accuracy.rs`, which:

- Builds a large sketch from random keys.
- Tracks exact counts for shallow prefixes.
- Asserts that depth-1 prefixes are exact (because `alpha=0.5` ⇒ `q(1)=1`).
- Enforces <15% relative error for high-support prefixes (≥200 true count).
- Adds targeted regression coverage (`merge_splits_compressed_edges_when_needed`)
  so merges stay correct even when long suffixes diverge past
  `PROMOTION_DEPTH`.
- Validates heavy-hitter queries on compressed edges
  (`heavy_hitters_cover_mid_edge_prefixes`) so autocomplete doesn’t break when a
  prefix ends mid-edge.
- Guards sampler determinism via `alpha_point_five_sampler_matches_leading_zeros`
  and HH merge determinism via `top_k_compactor_merge_is_order_invariant`.

This ensures probabilistic behavior matches theory under a representative
workload.

### 6. Examples & documentation
To make the crate usable without reading the entire design paper, we added:

- `IMPLEMENTATION.md` (this doc) and `examples/basic.rs` / `examples/heavy_hitters.rs`
  / `examples/merge_shards.rs` / `examples/range_counts.rs` to cover core use
  cases (ingest + merge + heavy hitters + integer ranges).
- Crate-level doc comment with runnable snippet.
- Extensive rustdoc on each public method (detailing scaling, merges, etc.).

## Toolchain

The crate targets Rust 2024 (see `Cargo.toml`). Make sure your toolchain is
new enough to enable that edition before running builds or tests.

## Running the project

- `cargo test` – runs unit, doctest, and integration suites.
- `cargo bench` – Criterion benchmarks for add/estimate/merge/top-k/prune.
- `cargo run --example basic` – small end-to-end demo.
- `cargo run --example heavy_hitters` – heavy-hitter query sample.

## Future Work

1. Configurable promotion depth & arena allocation.
2. More numerically stable inclusion probabilities (per-depth table).
3. Alternative heavy-hitter sketches (Count-Min, etc.) plug-ins.
4. Streaming/backpressure interface for pruning/compaction.
5. Extended sampler that chains additional deterministic blocks when all 128 bits are zero, removing the 129-depth ceiling for `α = 0.5`.

For now, this implementation matches the majority of DESIGN.md’s promises and
keeps the public API ergonomic for application developers.

### 7. Heavy hitters along compressed edges

Users shouldn’t need to know whether a prefix stops at a node or mid-edge. The
new `locate_prefix` helper lets `top_completions` stitch the remainder of a
compressed edge onto the requested prefix before applying the downstream top-k
compactor. Callers can now ask for completions of `"abcde"` even if the trie
stores `"def"` as a single edge. Only suffixes observed at that child node are
surfaced—updates that stopped exactly at the mid-edge accumulator never enter
that compactor—so mid-edge HH output always has the form
`prefix + remaining_edge + hh_suffix`.

### 8. Pointer safety in trie navigation

Earlier versions relied on raw pointers in `ensure_unit_child` and
`ensure_edge`. These helpers now work purely with safe Rust borrows by looking
up indices first and re-borrowing after any structural edits. This removes the
latent risk of invalidated pointers during merges or edge splits.
