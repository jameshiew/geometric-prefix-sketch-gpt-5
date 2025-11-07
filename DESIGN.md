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

* `B_sum(P)`: a **rescaled** accumulator for the sum/count under prefix `P`.
* (Optional) a tiny **heavy‑hitter sketch** (e.g., Misra-Gries or a 3–4 row Count‑Min) to extract frequent *extensions* under `P` for autocomplete.
* Child pointers (we recommend a **compressed trie**/radix node to pack runs of characters).

We fix a geometric parameter `α ∈ (0,1)` that tunes cost vs. accuracy (think `α = 1/2` by default).

Define

* Sampling distribution over positive integers (L \ge 1):
  [
\Pr[L \ge \ell] = q(\ell) = \alpha^{\ell-1}, \quad
\Pr[L=\ell] = (1-\alpha)\,\alpha^{\ell-1}.
  ]
* For a prefix of length (\ell), we’ll need the factor (q(\ell)).

**Deterministic sampling from the key.** To make updates repeatable and shard‑mergeable, we compute (L) **deterministically** from a 64‑bit hash of the key: e.g., let `u` be the hash viewed as a fixed‑point uniform in (0,1); then
[
L = 1 + \left\lfloor \frac{\ln u}{\ln \alpha} \right\rfloor,
]
truncated at (|key|). (With (\alpha=1/2), this is just **1 + number‑of‑leading‑zeros** in the hash until the first 1‑bit—exactly like HyperLogLog’s `rho` function—capped by the key length. That makes `L` sampling **branchless** and fast.)

---

## Operations

### Insert / add(key, Δ)

Adds value `Δ` (often 1 for counting). Let `s` be the key, length `|s|`.

1. Compute `h = Hash64(s)`.
2. Compute `L = sample_level(h, α)` as above; set `L = min(L, |s|)`.
3. Walk the trie from the root for the **first `L` characters** of `s`, creating compressed nodes if missing. At each visited prefix node `P` (depth `ℓ`):

   * Update the **rescaled** sum:
     `B_sum(P) += Δ`  (note: we store the *rescaled* quantity directly; see “Query”)
   * (Optional) Update `P`’s heavy‑hitter sketch with the **remaining suffix** (or the full key), weight `Δ`.

**Why is this unbiased?**
For any fixed prefix (P) of length (\ell), a key under (P) contributes to `B_sum(P)` **iff** (L \ge \ell), which happens with probability (q(\ell)=\alpha^{,\ell-1}). If we define the estimated sum at query time as ( \widehat{A}(P) = B_sum(P)/q(\ell) ), then for each key with contribution (\Delta),
[
\mathbb{E}\left[\frac{\mathbf{1}[L\ge \ell] \cdot \Delta}{q(\ell)}\right] = \Delta,
]
so the estimate is **unbiased**.

> **Cost per update:** ( \mathbb{E}[L] = \sum_{\ell\ge 1} \Pr[L \ge \ell] = \sum_{\ell\ge 1} \alpha^{\ell-1} = 1/(1-\alpha) ).
> With (\alpha = 1/2), ( \mathbb{E}[L] = 2). So **O(1) expected** trie steps and node touches per update.

### Delete / add(key, −Δ)

Same as insert (deterministic `L`), but subtract.

### Query: sum/count under a prefix P

1. Traverse the trie to node `P` (if missing ⇒ estimate 0).
2. Return
   [
   \widehat{A}(P) = \frac{B_sum(P)}{q(|P|)} = \frac{B_sum(P)}{\alpha^{,|P|-1}}.
   ]

**Variance & relative error.**
If `Δ=1` (counting), let (N_P) be the true number of keys under prefix (P) (length (\ell)). Each contributes a Bernoulli with inclusion prob (q(\ell)) and weight (1/q(\ell)). Thus
[
\mathrm{Var}[\widehat{A}(P)] = N_P \cdot \frac{1-q(\ell)}{q(\ell)}.
]
Relative RMSE (\approx \sqrt{\frac{1-q(\ell)}{q(\ell) , N_P}}).

* For (\alpha=1/2): (q(\ell) = 2^{-(\ell-1)}).

  * Depth 1: (q=1) ⇒ **exact** (variance 0) — nice property for top‑level categories.
  * Depth 2: relative RMSE (\approx 1/\sqrt{N_P}).
  * Depth 3: relative RMSE (\approx \sqrt{3}/\sqrt{N_P}), etc.

You can pick (\alpha) to tune the depth‑accuracy curve vs. per‑update cost:

* Higher (\alpha) (e.g., 0.7): better accuracy at deeper prefixes but higher update cost ((1/(1-\alpha))).
* Lower (\alpha) (e.g., 0.4): cheaper updates, less accuracy for deep prefixes.

### Top‑k under a prefix

At node `P`, query the local heavy‑hitter sketch (updated only when `P` was touched, i.e., with probability (q(|P|)) per key). The sketch naturally tracks frequent **completions** under `P`. You can scale counts by (1/q(|P|)) to unbias.

### Merge (distributed)

Because inclusion decisions and levels are **deterministic from the key hash**, two GPS structures built on disjoint streams can be merged by **pointwise addition** of:

* `B_sum` at corresponding trie nodes
* (optional) heavy‑hitter sketches (mergeable sketches or simple sum of Count‑Min tables)

No global coordination is required.

---

## Implementation sketch

### Node layout

Use a **compressed radix node** (like a succinct Patricia trie):

```text
struct Node {
  string edge_label;        // compressed run (could be slice into an arena)
  double B_sum;             // or fixed-point / 64-bit int if counts are big
  Children children;        // sorted small vector or array-mapped (HAMT-like)
  Optional<HHSketch> hh;    // tiny Misra-Gries or CM-Sketch (optional)
}
```

* **Arena allocation** for node/edge storage.
* **Children**: for byte alphabets, a tiny 256‑bit bitmap + packed child array is fast; for UTF‑8 or general alphabets, a sorted small vector (binary search) is usually fine.

### Sampling function (branchless, (\alpha=1/2))

```c
uint64_t h = splitmix64(key_bytes);
int L = 1 + clz( (~h) & (h-1) );  // equivalent to number of leading zeros before first 1
L = min(L, key_length);
```

(Or use standard “position of first 1” trick; any deterministic geometric sampler works.)

### Insert pseudocode

```pseudo
function insert(key, Δ, α=0.5):
    h  = Hash64(key)
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
        node.B_sum += Δ             // rescaled accumulator
        if node.hh exists:
            node.hh.update(key, Δ)  // optional heavy hitters
```

### Query pseudocode

```pseudo
function estimate_prefix_sum(prefix, α=0.5):
    node = find(prefix)
    if node == null: return 0
    depth = length_in_chars(prefix)
    q = α^(depth-1)
    return node.B_sum / q
```

> **Note on scaling:** To avoid large divisions, you can store `B_sum_scaled(P) = B_sum(P) * (α^(|P|-1))` and return `B_sum_scaled(P) / (α^(|P|-1)) / (α^(|P|-1))`—but that complicates integers. In practice, using `double` for `B_sum` is fine for analytics. If exact integer arithmetic is required, maintain per‑depth integer accumulators and multiply at query time by a precomputed reciprocal table.

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
   * Optional Misra-Gries sketch per node (capacity 8–16).
2. **Datasets:**

   * Public query logs (e.g., AOL 2006), Wikipedia page titles, real URL paths, synthetic Zipfian strings.
3. **Ground truth:** Build an exact prefix‑count trie (or hash map keyed by prefixes).
4. **Experiments:**

   * Vary `α ∈ {0.4, 0.5, 0.6, 0.7}`; measure **update throughput**, **memory**, and **prefix error** by depth.
   * Plot relative RMSE vs. true count (N_P) and depth (\ell); confirm ( \sim \sqrt{(1-q)/ (q N_P)}).
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
* **Memory:** Number of realized nodes is the number of **distinct prefixes that were ever sampled**, which is (O(n \cdot \mathbb{E}[L]) = O(n/(1-\alpha))) in the worst case (typically far less due to shared prefixes). Each node stores a double and small metadata; optional tiny sketches add fixed overhead.

---

## Practical tips & variants

* **Parameter choice.** Start with (\alpha=0.5). If you need higher fidelity at deeper levels (e.g., 3–5 chars), try 0.6–0.7; watch update cost ((1/(1-\alpha))).
* **Integer ranges.** Implement a bit‑trie front end: a 64‑bit unsigned splits naturally into 64 levels; GPS updates only a constant expected number of levels; range queries decompose into (\le 2\log U) prefixes—sum their estimates.
* **Bias/variance tuning.** You can make depth‑dependent (\alpha_\ell) (e.g., slower decay beyond depth 4) by reading more bits from the hash to sample from a *piecewise geometric* distribution; same analysis applies with (q(\ell)=\Pr[L\ge\ell]).
* **Heavy hitters.** Misra-Gries with capacity 8–16 per node gives good top‑k completions; scale estimates by (1/q(|P|)).
* **Persistence.** Because decisions are hash‑deterministic, you can **replay** updates idempotently and snapshot/restore cheaply.
* **Concurrency.** Shard by first byte/edge; merge node‑local counters with lock‑free atomics; the per‑node work is tiny.

---

## Limitations (and how to mitigate)

* **Unbounded variance for very deep, very small prefixes.** True: when (N_P) is tiny and (\ell) large, relative error can be high. Mitigate by:

  * Raising (\alpha) (spend more CPU), or
  * **On‑demand promotion:** if a prefix is frequently queried, start *exactly* maintaining that node: on future updates, always touch it (set (q(\ell)=1) for that specific node).
* **Counts only (sums of nonnegative values).** For signed updates (turnstile with cancellations), the estimator remains unbiased, but variance adds; use wider counters or float.

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
q(depth) = α^(depth-1)

function sample_level(hash64 h, α):
    // Convert to uniform u in (0,1)
    u = (h + 1) / 2^64
    // Geometric tail: floor( ln(u)/ln(α) ) + 1
    L = 1 + floor(log(u) / log(α))
    return max(1, L)

function add(key s, Δ):
    h = Hash64(s)
    L = min(sample_level(h, α), |s|)
    node = root
    depth = 0
    i = 0
    while depth < L:
        (node, consumed_chars) = descend_or_create(node, s, i)
        i     += bytes(consumed_chars)
        depth += consumed_chars
        node.B_sum += Δ              // rescaled accumulator
        // optional: node.hh.update(s, Δ)

function estimate(prefix p):
    node = find(prefix p)
    if node == null: return 0
    depth = |p|
    return node.B_sum / q(depth)
```

You can implement this directly and validate it today.
