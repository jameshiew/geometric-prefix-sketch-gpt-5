# Geometric Prefix Sketch (GPS)

*A constant‑update, mergeable, unbiased sketch for fast prefix (or hierarchical) aggregation.*

### What it’s for

Whenever you have **string or hierarchical keys** (paths, URLs, DNS names, tokens, IP prefixes, category trees, binary integers for range queries) and you need, in real time:

* `count("foo*")` or `sum(value | key has prefix "foo")`
* “top prefixes” at any depth
* prefix analytics in distributed systems (merge sketches from many machines)
* approximate **range counts** for integers (by treating binary prefixes as a trie)

Existing exact structures (tries, wavelet trees, Fenwick-on-tries, hierarchical Count‑Min) require **O(|key|)** work per update because every inserted key contributes to all of its prefixes. GPS brings that down to **O(1) expected** work per update while keeping unbiased estimates and making distributed merging trivial.

---

## Core idea (one-line intuition)

When a key arrives, choose a **random maximum prefix length (L)** (geometric distribution) **once**, then update **only the first (L)** prefix nodes along its path—**but scale the contribution** so that **any queried prefix** gets an **unbiased** estimate of its total.

This replaces “touch every prefix” with “touch a **randomly truncated** set of prefixes of expected constant size,” while compensating statistically so that queries remain correct in expectation.

---

## Data model

Consider keys as strings over an alphabet (bytes, UTF‑8 code units, bits for integers). Let the **prefix trie** of all observed keys be the conceptual index. We do **not** populate the full trie; we populate only the nodes that get touched by sampled updates.

Each realized trie node `P` stores:

* `S(P)`: a **sampled (unscaled)** accumulator for the sum/count under prefix `P`. We add `Δ` to `S(P)` whenever `P` is visited; the unbiased estimate divides by the inclusion probability \(q(|P|)\) at query time.
* (Optional) a tiny **heavy-hitter summary** implemented as a bounded top-k compactor (think “keep the `k` heaviest suffixes seen so far”). It merges by summing shared keys and re-truncating to capacity, and it only sees the Bernoulli subsample induced by depth `|P|`. It is **not** a strict Misra-Gries guarantee; if you need deterministically bounded error, swap in an actual MG/SpaceSaving variant.
* Child pointers (we recommend a **compressed trie**/radix node to pack runs of characters). Every compressed edge stores `mid_sums` for each interior prefix so queries that stop mid-edge still recover their sampled accumulator without forcing a structural split.

We fix a geometric parameter `α ∈ (0,1)` that tunes cost vs. accuracy (think `α = 1/2` by default).

Define

* Sampling distribution over nonnegative depths:
  [
\Pr[L \ge 0] = q(0) = 1, \quad q(1) = 1,\quad
\Pr[L \ge \ell] = q(\ell) = \alpha^{\ell-1}\; (\ell>1),\quad
\Pr[L=\ell] = (1-\alpha)\,\alpha^{\ell-1} \text{ for } \ell \ge 1.
  ]
* For a prefix of length (\ell), we’ll need the factor (q(\ell)).

**Deterministic sampling from the key.** To keep updates repeatable and shard-mergeable, we compute (L) **solely** from a keyed 128-bit hash of the key. The implementation interprets the hash as an unsigned Q128 value and walks a **precomputed monotone threshold table** `q128[d] ≈ ⌊α^{d-1} · 2^128⌋`. Starting at `d = 1`, advance while `hash < q128[d+1]`, clamping to both the key length and the realizable depth cap (129 for `α=0.5`, or the inclusion-table length for general `α`). This avoids runtime logs, stays integer-only, and guarantees identical behavior across platforms and shards (`sample_level_for_hash` in `src/sketch.rs` mirrors this logic).

* **Fast path for `α = 0.5`.** We still keep the branchless **1 + clz128(hash)** shortcut, capped at 129, because `α^{d-1}` aligns perfectly with powers of two. The full Q128 table is used for every other α.
* **Executable checks.** `depth_one_counts_are_exact_for_general_alpha` and `alpha_point_five_sampler_matches_leading_zeros` assert the sampler’s determinism and depth limits in the shipped crate.

> **Per-key deterministic sampling only.** The sampler hashes the *key* (plus seed) exactly once, so all occurrences of that key share the same `L`. Variance therefore follows the Horvitz–Thompson expression with \(\sum_s w_s^2\); there is currently no per-occurrence mode.

> Use a **keyed** XXH3-128 hash so every shard agrees on inclusion decisions. Share `(α, seed)` across shards to remain deterministic and adversarially safe.

### Constructors & seeding semantics

The public constructors make those seeding guarantees explicit:

- `GpsSketch::default()` → draws a fresh random seed (great for adversarial resistance; capture the seed before merging).
- `GpsSketch::with_seed(alpha, seed)` → explicit seed for deterministic shard merges.
- `GpsSketch::new(alpha)` → fixed seed `0` for legacy/compatibility scenarios (deterministic but adversarially predictable).

`with_heavy_hitters` composes with the above and forwards the seed unchanged; callers decide whether they prefer safety (`default`) or reproducible merges (`with_seed`/`new`).

---

## Operations

### Insert / add(key, Δ)

Adds value `Δ` (often 1 for counting). Let `s` be the key, length `|s|`.

1. Compute `h = Hash128(s)`.
2. Compute `L = sample_level(h, α)` as above; set `L = min(L, |s|)`.
3. Walk the trie from the root for the **first `L` characters** of `s`, creating compressed nodes if missing. At each visited prefix node `P` (depth `ℓ`):

   * Update the **sampled** sum:
     `S(P) += Δ`  (we store the raw sampled total and scale only at query time)
   * (Optional) Update `P`’s top-k compactor with the **remaining suffix after `P`**, weight `Δ`. Only **positive** weights are inserted (`Δ \le 0` is ignored) so the summary stays insertion-only. The sketch therefore tracks a Bernoulli subsample at this depth; multiply reported weights by the inclusion factor used by that summary when displaying. Because the compactor is nonlinear (truncate-to-capacity), we simply add sampled weights and scale on readback; it does **not** inherit Misra-Gries frequency guarantees, only “keep the heaviest things we saw.” The stored HH labels are suffixes relative to the node; concatenate `prefix + suffix` to reconstruct the original key.

**Why is this unbiased?**
For any fixed prefix (P) of length (\ell), a key under (P) contributes to `S(P)` **iff** (L \ge \ell), which happens with probability (q(\ell)=\alpha^{\ell-1}). If we define the estimated sum at query time as ( \widehat{A}(P) = S(P)/q(\ell) ), then for each key with contribution (\Delta),
[
\mathbb{E}\left[\frac{\mathbf{1}[L\ge \ell] \cdot \Delta}{q(\ell)}\right] = \Delta,
]
so the estimate is **unbiased**.

> **Cost per update:** ( \mathbb{E}[L] = \sum_{\ell\ge 1} \Pr[L \ge \ell] = \sum_{\ell\ge 1} \alpha^{\ell-1} = 1/(1-\alpha) ).
> With (\alpha = 1/2), ( \mathbb{E}[L] = 2). So **O(1) expected** trie steps and node touches per update.

### Delete / add(key, −Δ)

Same as insert (deterministic `L`), but subtract. Heavy-hitter summaries stay insertion-only: negative deltas still update the node’s sampled sum but do **not** remove entries from the compactor. This matches the library implementation; if deletions are frequent and you need HH parity, supply a signed sketch variant (e.g., SpaceSaving) per node instead.

### Query: sum/count under a prefix P

1. Traverse the compressed trie while tracking whether the prefix lands **on a node** or **inside an edge**. If the traversal fails, return 0.
2. Let `(raw, depth)` be the match: nodes return their stored `S(P)` with `depth = |P|`; mid-edge matches return the appropriate `mid_sums[idx]` bucket with `depth` equal to the matched prefix length (child depth of that bucket).
3. Return `raw / q(depth)`. By definition `q(0) = q(1) = 1`; for deeper levels the table gives `α^{depth-1}` until the realizable cap, after which `q(depth)=0` and the estimate is defined to be 0.

**Variance & relative error.** Let (q = q(|P|) = \alpha^{|P|-1}).

*If sampling is per occurrence* (each arrival has its own stable event ID and hash), arrivals with values (\Delta_i) behave like standard Bernoulli HT sampling:
[
\mathrm{Var}[\widehat{A}(P)] = \frac{1-q}{q}\sum_i \Delta_i^2.
\]
For pure counts (\Delta_i=1), this collapses to (N_P \cdot \frac{1-q}{q}) and relative RMSE (\approx \sqrt{\frac{1-q}{qN_P}}).

*In this library’s default per‑key deterministic sampling*, every occurrence of key (s) under (P) moves together. Let (w_s) be the total weight collected by that key (counts ⇒ (w_s=f_s)). Then
[
\mathrm{Var}[\widehat{A}(P)] = \frac{1-q}{q}\sum_s w_s^2,\qquad
\mathrm{rRMSE} \approx \sqrt{\frac{1-q}{q\,N_{\text{eff}}}},
\]
where (N_{\text{eff}} = (\sum_s w_s)^2 / \sum_s w_s^2) is the **effective** number of equally weighted keys. The simpler (N_P) formula only holds when each key appears once.

Because (q(1)=1) for any (\alpha), depth‑1 totals are always exact. Raising (\alpha) improves accuracy for deeper depths at the cost of higher expected update work ((1/(1-\alpha))); lowering (\alpha) does the opposite.

**Iteration & root totals.** `iter_estimates()` traverses both node prefixes and mid-edge buckets, applying the same \(1/q(depth)\) scaling, so users see every realized prefix without reimplementing trie walks. `iter_estimates_reports_mid_edge_prefixes` and `iter_estimates_match_raw_traversal` keep that guarantee executable. The depth-0 root is always materialized with `q(0)=1`, so `total()` is exactly `estimate("")`; `empty_key_only_touches_root` covers that invariant by inserting the empty key.

### Top-k under a prefix

At node `P`, query the local heavy-hitter sketch (updated only when `P` was touched, i.e., with probability (q(|P|)) per key). When the query ends **mid-edge**, append the remaining edge label first and then consult the downstream child’s summary; the summary’s depth is the child depth, not the user-specified prefix. Scale reported weights by \(1/q(d_{\text{summary}})\), where `d_summary` equals the depth of the node that owns the heavy-hitter sketch. Because these sketches are insertion-only, weights reported after deletions may diverge from the unbiased `estimate(P)` result.

If a node-level query lands exactly on a node that either lacks a heavy-hitter sketch or whose compactor is empty **and** the node has exactly one child, we “fall through” to that child: append the solo edge label, reuse the child’s HH summary, and scale by the child’s depth before returning completions. This avoids silent gaps for prefixes that collapse into deterministic chains while keeping branching nodes localized.

> Note: The bounded top-k compactor is **nonlinear** and insertion-only. Scaling its reported weights by \(1/q(d_{\text{summary}})\) adjusts magnitudes but does not make the top-k set itself unbiased. In practice, heavy items stay heavy under Bernoulli subsampling, and the compactor remains mergeable because we add shared suffixes, allow the merged compactor to grow to at most about `2×capacity - 1`, and only then truncate (or whenever the compactor is explicitly compressed). That deferred compression keeps merges order-invariant while avoiding churn from repeated truncations.

### Heavy-hitter caveats (current behavior)

- **Insertion-only summaries.** Negative deltas update the unbiased node counters but are ignored by the top-k compactor, so completions can temporarily overstate frequency after deletions. This matches `TopKCompactor::update` and is documented in the public API (`GpsSketch::add` docs).
- **Mid-edge completions scale at the child depth.** When a query stops mid-edge, we append the remaining edge label, read the child’s summary, and multiply by \(1/q(\text{child depth})\). `mid_edge_top_completions_use_child_depth` asserts this scaling so completions stay consistent with `estimate()`.
- **Merge = sum, then defer compression until ≈ `2×capacity`.** Heavy-hitter summaries merge by adding shared suffixes, appending any new suffixes, and only trimming once the compactor reaches `2·capacity` entries (or when callers explicitly compress). This keeps merges order-invariant (`top_k_compactor_merge_is_order_invariant`) while guaranteeing the working set never exceeds `2×capacity - 1` entries per node. Expect small divergences from strict Misra-Gries guarantees; GPS intentionally prioritizes “keep the heaviest suffixes we saw” behavior.

### Merge (distributed)

Because inclusion decisions and levels are **deterministic from the key hash**, two GPS structures built on disjoint streams can be merged by **pointwise addition** of:

* `S` at corresponding trie nodes
* (optional) heavy-hitter sketches (mergeable sketches or simple sum of Count-Min tables)

When the tries are compressed, merging also has to respect **edge boundaries**:

1. For every incoming compressed edge, find the destination edge that shares the same first byte.
2. Walk down while the labels match. If they match exactly, add `mid_sums` elementwise and recurse into the child.
3. If the match ends mid-label (partial overlap), **split** the destination edge at the longest common prefix. Add overlapping `mid_sums` for the shared interior buckets.
4. **Add** (do not rescale) the overlapping boundary accumulator (`mid_sums[lcp-1]`) into the child node’s `S` before recursing so that the promoted node stores the correct raw mass.

This keeps the trie canonical and preserves unbiased estimates even when merges introduce new branching points along a compressed edge.

No global coordination is required beyond agreeing on `(\alpha,\text{seed}[,\text{hh capacity}])`. Remember the crate’s default constructor randomizes the seed; call `with_seed`/`with_heavy_hitters` when you intend to merge shards.

`merge_boundary_mid_carries_overlap` exercises the entire split/add path to ensure merges transfer boundary mass exactly once.

> **Promotion caveat / future work:** the shipped crate only implements **structural promotion** (unit-byte nodes up to `PROMOTION_DEPTH`). Per-node “exact maintenance” (forcing \(q=1\) for specific prefixes) remains future work. If or when you add it, the promotion policy must be deterministic and shared across shards to avoid bias.

---

## Implementation sketch

### Node layout

Use a **compressed radix trie** (Patricia) with explicit edges:

```text
struct Node {
  double S;                 // sampled (unscaled) sum; int64 works if counts only
  Edge[] children;          // sorted small vector or array-mapped (HAMT-like)
  Optional<HHSketch> hh;    // tiny bounded top-k compactor (optional)
}

struct Edge {
  string label;             // compressed run (slice into arena)
  double mid_sums[label.len()-1]; // raw sums for every interior prefix on this edge
  Node child;               // subtree after the label
}
```

Every realized prefix lives either on a node (`S`) or inside an edge (`mid_sums[i]`). Mid-edge accumulators are crucial: a query that stops in the middle of a compressed label must still return the sampled mass collected at that interior depth, so we materialize that accumulator in `mid_sums` rather than forcing a node split.

* **Arena allocation** for node/edge storage.
* **Children**: for byte alphabets, a tiny 256-bit bitmap + packed child array is fast; for UTF-8 or general alphabets, a sorted small vector (binary search) is usually fine.

**Hybrid trie (unit-byte promotion + compression).** Depths `0..PROMOTION_DEPTH` (the Rust crate fixes `PROMOTION_DEPTH = 4`) are stored as **unit-byte nodes** so every character has its own node. That keeps shallow prefixes exact and simplifies merges. Beyond that depth the trie switches to **compressed edges** with `mid_sums`, drastically shrinking memory for long suffixes. This is the only form of “promotion” implemented today—it’s **structural**, not a “set q(ℓ) = 1” toggle. Merges still preserve correctness but can materialize compressed edges even inside the promoted depths whenever the destination shard lacks a matching child for that byte. Inserts continue to enforce unit-byte nodes, and a post-merge normalization pass is optional if a workload needs the structural invariant restored.

> Exact-maintenance (q=1) promotion is **future work**. Any design that maintains per-node q=1 must share the promotion set across shards; the current crate does not ship this feature and always uses the geometric sampler for every node beyond the structural promotion depth.

### Sampling function (α=0.5 fast path; general-α threshold table)

We hash every key with XXH3-128 (keyed) and count leading zeros across the full
128 bits:

```c
uint128_t h = xxh3_128(key_bytes, seed);
int L = 1 + clz128(h);   // number of leading zeros until the first 1 bit
L = min(L, key_length, 129);  // 128 bits ⇒ max realizable depth 129
```

This exactly matches the implementation: the sampler can realize depths 1‥=129
when `α = 0.5`, and anything deeper is treated as depth 129. Longer prefixes are
still stored (the trie can grow arbitrarily deep) but they rely on observations
from other shards or higher α to get sampled mass.

> **Extended sampler (future work).** Earlier drafts suggested streaming extra
> deterministic blocks (e.g., from a keyed XOF) whenever all 128 bits are zero
> so the geometric tail never truncates. We have not implemented that yet; it's
> listed as future work so this document stays honest about the current cap.

For general (α), use the same keyed 128-bit hash as a Q128 integer and advance through a precomputed monotone threshold array `q128[d] ≈ ⌊α^{d-1} · 2^128⌋`, stopping at the first depth where `hash ≥ q128[d+1]`. This yields deterministic, platform-stable sampling without floating point.

### Depth caps & zeroed tails

* **α = 0.5 (fast path):** Counting leading zeros across 128 hash bits yields at most 128 zeros, so the realizable depth is \(L \le 129\). Any sampled depth beyond that is clamped to 129, and the sampler also caps by the key length. Depth‑1 remains exact for every α; `depth_one_counts_are_exact_for_general_alpha` locks this in the test suite.
* **General α:** The inclusion table stores `q(ℓ)` for **up to 200 000** entries (capped) in the reference crate and defines `q(ℓ) = 0` beyond that. Insertions clamp `L` to `min(|key|, table_len, depth_cap)` so high-α configurations cannot allocate unbounded tables.
* **Queries:** Because `q(depth)=0` past the realizable cap, estimates deeper than the table deterministically return 0. `unreachable_depth_estimates_to_zero` exercises this specification so applications are never surprised by a silent underflow.

### Insert pseudocode

```pseudo
function insert(key, Δ, α=0.5):
    h  = Hash128(key)
    L  = sample_geometric_level(h, α)
    L  = min(L, len(key))
    node = root
    depth = 0
    off = 0
    while depth < L:
        // descend or create next component (compressed)
        (node, consumed) = descend_or_create(node, key[off:])
        off   += consumed
        depth += number_of_chars_in(consumed)
        node.S += Δ                 // sampled (unscaled) accumulator
        if node.hh exists and Δ > 0:
            node.hh.update(remaining_suffix_after_this_node(key, off), Δ)
            // HH sketch stores suffix relative to this node
```

### Query pseudocode

```pseudo
function estimate_prefix_sum(prefix, α=0.5):
    match = locate(prefix)
    if match == null: return 0
    if match is Node:
        depth = length_in_chars(prefix)
        raw = match.node.S
    else:
        depth = match.depth
        raw = match.mid_bucket
    q = α^(depth-1)
    if q == 0: return 0
    return raw / q
```

> **Note on scaling:** We store `S(P)`/`mid_sums` as raw sampled totals. At query time we divide by the inclusion probability of the matched depth `q(depth)`. If you need integer-only outputs, precompute reciprocals (or fixed-point factors) per depth and apply them on read.

> **Depth cap trade-off:** See “Depth caps & zeroed tails” for the precise limits. Practically, the table stops after ~200 000 entries (general α) or 129 entries (α=0.5), so `q(depth)=0` beyond that and estimates are fixed at zero.

---

## Why this works & what’s new

* It’s a **linear sketch** on the trie incidence structure with a **random truncated path** per key—so per‑key update cost is **constant in expectation**, not (O(|key|)).
* Estimates are **unbiased** for every prefix length simultaneously.
* The sketch is **mergeable** and **deterministic** across shards.
* With (\alpha=1/2), depth‑1 counts are **exact** (a useful anchor), and deeper levels trade accuracy for speed in a controllable way.
* I’m not aware of a published structure that chooses a **single randomized depth** per key and **pushes scaled mass up the prefix path** to create unbiased prefix estimates with **O(1)** expected update work while keeping a concrete, material trie that supports fast prefix lookup and heavy hitters.

---

## Practical uses (high impact)

1. **Autocomplete & search suggestions.** Maintain live counts for any prefix while ingesting millions of queries/sec.
2. **URL/API telemetry.** Get `GET /api/v1/users/*` counts instantly; drill into deeper path prefixes on demand.
3. **DNS & security.** Count subdomain activity under `*.example.com` or IP /24, /20, /16 prefixes in stream.
4. **Retail taxonomy analytics.** Live rollups for category trees (`Electronics > Laptops > 14"`).
5. **Integer range counts.** Treat integers as bit strings; any range ([a,b]) decomposes into (O(\log U)) prefixes—sum those nodes’ estimates.

---

## How to verify (step‑by‑step plan)

1. **Reference implementation** (200–400 LOC) in C++/Rust/Go:

   * Compressed radix trie + GPS logic.
   * Configurable `α`.
   * Optional bounded top-k compactor per node (capacity 8–16).
2. **Datasets:**

   * Public query logs (e.g., AOL 2006), Wikipedia page titles, real URL paths, synthetic Zipfian strings.
3. **Ground truth:** Build an exact prefix‑count trie (or hash map keyed by prefixes).
4. **Experiments:**

   * Vary `α ∈ {0.4, 0.5, 0.6, 0.7}`; measure **update throughput**, **memory**, and **prefix error** by depth.
   * Plot relative RMSE vs. depth (\ell) and **effective size** \(N_{\text{eff}} = (\sum_s f_s)^2 / \sum_s f_s^2\); confirm (\sim \sqrt{(1-q)/(q N_{\text{eff}})}) under per-key sampling.
   * **Latency** of `estimate_prefix_sum` is just trie lookup (microseconds).
   * **Merging:** build two sketches on split halves, merge, compare to single‑pass sketch and ground truth.
5. **A/B vs. baselines:**

   * Exact compressed trie (same code without sampling) — check speedup vs. error.
   * Hierarchical Count‑Min (updates every prefix): compare throughput/memory/error (GPS should dominate on update cost for similar accuracy).
   * Bloom/CBF cannot answer prefix queries directly—use as a control.

---

## Complexity & memory

* **Insert/Delete:** (O(1)) expected node touches; worst‑case (O(|key|)) if `L` hits full length (rare, geometric tail).
* **Query:** (O(|prefix|)) to traverse the compressed path; answer in (O(1)) after lookup.
* **Memory:** Number of realized nodes is the number of **distinct prefixes that were ever sampled**, which is (O(n \cdot \mathbb{E}[L]) = O(n/(1-\alpha))) in the worst case (typically far less thanks to shared prefixes). Here (n) is the number of **distinct keys** ever observed. Each node stores a counter and small metadata; optional tiny sketches add fixed overhead.

---

## Practical tips & variants

* **Parameter choice.** Start with (\alpha=0.5). If you need higher fidelity at deeper levels (e.g., 3–5 chars), try 0.6–0.7; watch update cost ((1/(1-\alpha))).
* **Integer ranges.** Implement a bit‑trie front end: a 64‑bit unsigned splits naturally into 64 levels; GPS updates only a constant expected number of levels; range queries decompose into (\le 2\log U) prefixes—sum their estimates.
* **Bias/variance tuning.** You can make depth‑dependent (\alpha_\ell) (e.g., slower decay beyond depth 4) by reading more bits from the hash to sample from a *piecewise geometric* distribution; same analysis applies with (q(\ell)=\Pr[L\ge\ell]).
* **Heavy hitters.** A truncate-to-capacity top-k compactor with capacity 8–16 per node gives good completions; it only trims once entries reach `≈ 2×capacity`, capping memory at `2×capacity - 1` items between compressions. Scale estimates by \(1/q(d_{\text{summary}})\) where `d_summary` is the depth of the node that owns the summary (child depth for mid-edge prefixes). The summary is nonlinear and insertion-only, so scaling fixes magnitudes but not the set; if you need deterministic frequency guarantees or deletions, plug in a true Misra-Gries/SpaceSaving variant per node.

*Mid-edge HH behavior.* When a query prefix ends in the middle of a compressed edge, we first append the remaining edge label so we can consult the downstream node’s summary. Only the suffixes observed at that child participate—updates that stopped exactly at the mid-edge accumulator never enter that HH summary—so the completions you see are always of the form `prefix + remaining_edge + hh_suffix`.
* **Promotion policy.** If you “promote” certain prefixes to exact maintenance (force \(q=1\)), keep the policy deterministic and shared across shards so merges stay unbiased.
* **Persistence.** Because decisions are hash‑deterministic, you can **replay** updates idempotently and snapshot/restore cheaply.
* **Concurrency.** Shard by first byte/edge; merge node‑local counters with lock‑free atomics; the per‑node work is tiny.

---

## Limitations (and how to mitigate)

* **Unbounded variance for very deep, very small prefixes.** True: when (N_P) is tiny and (\ell) large, relative error can be high. Mitigate by raising (\alpha) (spend more CPU) or by building a secondary exact cache for the hottest prefixes. The previously sketched **“set \(q(\ell)=1\) for this node” promotion** is not implemented in this crate yet; keep it as future work unless every shard can deterministically agree on the promoted set.
* **Repeated keys inflate variance under per-key sampling.** When the same key appears many times, all of its occurrences are either sampled or not together, so variance scales with \(\sum_s w_s^2\). Mitigate by bumping \(\alpha\), keeping a **secondary exact cache** for the hottest prefixes (per-node `q=1` promotion is future work), or—if you have a stable per-event identifier—hashing on `(key, event_id)` to approximate per-occurrence sampling.
* **Counts only (sums of nonnegative values).** For signed updates (turnstile with cancellations), the estimator remains unbiased, but variance adds; use wider counters or float.
* **Insertion-only heavy hitters.** The default HH compactor never subtracts weight when you apply negative deltas. After deletions, its reported weights can lag the unbiased `estimate(P)` values. If this matters, either disable HH summaries or replace them with a signed sketch (e.g., SpaceSaving with explicit decrements).

---

## “Is it really new?”

This structure fuses:

* geometric‑depth sampling à la HyperLogLog,
* unbiased Horvitz–Thompson scaling,
* a material (compressed) trie to support **direct prefix addressing**, and
* per‑node optional heavy‑hitter summaries,

to deliver **unbiased prefix sums with O(1) expected update work** and trivial distributed merging. I’m not aware of prior art with exactly these ingredients and guarantees. If you plan to publish or patent, you should still run a formal prior‑art search around hierarchical sketches, sample‑and‑hold on tries, and prefix analytics.

---

## Minimal, testable reference (pseudo‑code completeness)

```pseudo
// Globals
α in (0,1)
MAX_DEPTH = realizable_depth(α)  // 129 when α=0.5, else inclusion-table size (~200k)
q(depth) =
    if depth <= 1: 1
    else if depth > MAX_DEPTH: 0
    else α^(depth-1)

function sample_level(hash128 h, α):
    if α == 0.5:
        L = 1 + clz128(h)             // capped at 129 via MAX_DEPTH
    else:
        // Precompute q128[d] = floor(α^(d-1) * 2^128) for d ≤ MAX_DEPTH
        L = 1
        while L < MAX_DEPTH and h < q128[L+1]:
            L += 1
    return L

function add(key s, Δ):
    // Hybrid trie: single-byte nodes for depths < PROMOTION_DEPTH (4), then compressed edges with `mid_sums`
    h = Hash128(s)
    L = min(sample_level(h, α), |s|)
    node = root
    depth = 0
    i = 0
    // Root (depth 0) keeps q(0)=1, so update it unconditionally to keep total() exact.
    node.S += Δ
    while depth < L:
        (node, consumed_chars) = descend_or_create(node, s, i)
        i     += bytes(consumed_chars)
        depth += consumed_chars
        node.S += Δ                  // sampled accumulator
        if node.hh exists and Δ > 0:
            node.hh.update(remaining_suffix_after_this_node(s, i), Δ)

function estimate(prefix p):
    match = locate(prefix p)
    if match == null: return 0
    if match is Node:
        depth = |p|
        raw   = match.node.S
    else:
        depth = match.depth
        raw   = match.mid_bucket
    if q(depth) == 0: return 0
    return raw / q(depth)
```

You can implement this directly and validate it today.
