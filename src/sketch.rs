//! Geometric Prefix Sketch (GPS) implementation with a compressed radix trie.
//!
//! The sketch follows the design laid out in `DESIGN.md`: every update touches a
//! geometrically sampled prefix depth, accumulators remain unbiased, sketches
//! merge additively, and (optional) per-node heavy-hitter summaries expose top
//! completions. The trie now uses *compressed edges* so long, single-child runs
//! past a configurable promotion depth occupy a single node/edge instead of one
//! heap allocation per byte.

use crate::tree::{
    Node, PrefixMatch, add_inner, locate_prefix, locate_raw_sum, merge_nodes, prune_node,
};
#[cfg(test)]
use crate::tree::{PROMOTION_DEPTH, TopKCompactor, ensure_edge, visit_raw};
#[cfg(test)]
use crate::util::inclusion_prob;
use crate::util::{InclusionTable, deterministic_hash};

/// Geometric Prefix Sketch with deterministic per-key sampling.
///
/// `GpsSketch` stores a compressed prefix trie whose nodes are populated by a
/// geometric (per-key) sampling scheme. Updates touch only a constant expected
/// number of prefixes while estimates remain unbiased. Instances are
/// merge-friendly as long as they share the same [`alpha`](Self::alpha) and
/// [`hash_seed`](Self::hash_seed).
#[derive(Clone, Debug)]
pub struct GpsSketch {
    depth_table: InclusionTable,
    hash_seed: u64,
    heavy_capacity: Option<usize>,
    root: Node,
}

impl Default for GpsSketch {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl GpsSketch {
    /// Creates a new sketch with the provided geometric parameter `alpha`.
    ///
    /// `alpha` controls the per-depth inclusion probability (higher values
    /// sample more prefixes per update but reduce variance on deep prefixes).
    pub fn new(alpha: f64) -> Self {
        Self::with_seed(alpha, 0)
    }

    /// Creates a sketch with explicit hash seed.
    ///
    /// Use this when multiple shards must produce identical sampling decisions
    /// so their sketches can be merged later. For adversarial workloads, pick a
    /// non-guessable seed so attackers cannot predict sampling depths.
    pub fn with_seed(alpha: f64, hash_seed: u64) -> Self {
        Self::with_heavy_hitters_internal(alpha, hash_seed, None)
    }

    /// Creates a sketch with per-node heavy-hitter capacity.
    ///
    /// Heavy hitters are tracked via a bounded top-k compactor stored on each
    /// visited prefix node. Only **positive** updates contribute to the heavy
    /// hitter stream.
    pub fn with_heavy_hitters(alpha: f64, hash_seed: u64, hh_capacity: usize) -> Self {
        assert!(hh_capacity > 0);
        Self::with_heavy_hitters_internal(alpha, hash_seed, Some(hh_capacity))
    }

    fn with_heavy_hitters_internal(
        alpha: f64,
        hash_seed: u64,
        heavy_capacity: Option<usize>,
    ) -> Self {
        assert!(alpha.is_finite());
        assert!((0.0..1.0).contains(&alpha));
        let depth_table = InclusionTable::new(alpha);
        Self {
            depth_table,
            hash_seed,
            heavy_capacity,
            root: Node::new(heavy_capacity),
        }
    }

    /// Returns the configured `alpha`.
    ///
    /// `alpha` corresponds to the geometric inclusion probability `q(ℓ) =
    /// alpha^(ℓ-1)` used when sampling prefixes.
    pub fn alpha(&self) -> f64 {
        self.depth_table.alpha()
    }

    /// Returns the deterministic hash seed.
    ///
    /// All sketches that need to merge must share this seed (and the same
    /// `alpha`).
    pub fn hash_seed(&self) -> u64 {
        self.hash_seed
    }

    /// Returns the heavy-hitter capacity, if enabled.
    pub fn heavy_hitters_capacity(&self) -> Option<usize> {
        self.heavy_capacity
    }

    /// Adds `delta` to `key`.
    ///
    /// Keys may be any byte slice (strings, paths, binary prefixes, …). Passing
    /// a negative `delta` removes mass, which is useful for turnstile updates.
    pub fn add<K: AsRef<[u8]>>(&mut self, key: K, delta: f64) {
        if delta == 0.0 {
            return;
        }
        let bytes = key.as_ref();
        if bytes.is_empty() {
            self.root.sum += delta;
            self.root.update_heavy_hitters(bytes, delta);
            return;
        }
        let limit = self.prefix_budget(bytes);
        self.root.sum += delta;
        self.root.update_heavy_hitters(bytes, delta);
        add_inner(&mut self.root, bytes, 0, limit, delta, self.heavy_capacity);
    }

    /// Returns the Horvitz–Thompson estimate of the sum under `prefix`.
    ///
    /// The estimator is unbiased for every prefix simultaneously. Nonexistent
    /// prefixes return `0`.
    pub fn estimate<K: AsRef<[u8]>>(&self, prefix: K) -> f64 {
        match self.raw_sum(prefix.as_ref()) {
            Some((raw, depth)) => self.scale_raw_estimate(raw, depth),
            None => 0.0,
        }
    }

    /// Returns the global total estimate (root prefix).
    pub fn total(&self) -> f64 {
        self.estimate(Vec::<u8>::new())
    }

    /// Removes all data from the sketch.
    pub fn clear(&mut self) {
        self.root = Node::new(self.heavy_capacity);
    }

    /// Returns the number of stored prefixes (materialized nodes + compressed interior ones).
    pub fn node_count(&self) -> usize {
        self.root.count_prefixes()
    }

    /// Returns whether `prefix` currently exists in the trie.
    ///
    /// A prefix is considered present if it has ever been sampled and assigned a
    /// node or mid-edge accumulator, regardless of its current estimated value.
    pub fn contains_prefix<K: AsRef<[u8]>>(&self, prefix: K) -> bool {
        self.raw_sum(prefix.as_ref()).is_some()
    }

    /// Returns iterator over all materialized prefixes and their estimates.
    ///
    /// Every prefix is cloned into an owned buffer so the iterator can outlive
    /// the borrowed sketch.
    pub fn iter_estimates(&self) -> impl Iterator<Item = (Vec<u8>, f64)> {
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        self.collect_estimates(&self.root, 0, &mut prefix, &mut out);
        out.into_iter()
    }

    /// Returns up to `k` heavy-hitter completions under `prefix`.
    ///
    /// Heavy hitters are available only when the sketch was created via
    /// [`with_heavy_hitters`](Self::with_heavy_hitters). Results are sorted by
    /// estimated weight. Prefixes that end mid-edge in the compressed trie are
    /// automatically completed through that edge before heavy-hitter suffixes
    /// are appended. Updates that sampled only up to the mid-edge (without a
    /// suffix) never enter the downstream node's summary, so they're omitted
    /// from the heavy-hitter output unless per-position HH sketches are
    /// enabled. Mid-edge completions therefore only include suffixes observed
    /// at the downstream node; pruned or unvisited tails are intentionally
    /// absent. Returned strings always equal `prefix` plus any remaining edge
    /// label (when the prefix ends mid-edge) plus the heavy-hitter suffix, and
    /// their weights are scaled using the inclusion probability at the summary
    /// depth of that downstream node.
    pub fn top_completions<K: AsRef<[u8]>>(&self, prefix: K, k: usize) -> Vec<(Vec<u8>, f64)> {
        if k == 0 {
            return Vec::new();
        }
        let prefix = prefix.as_ref();
        let Some(hit) = locate_prefix(&self.root, prefix, 0) else {
            return Vec::new();
        };
        match hit {
            PrefixMatch::Node { node, depth } => {
                let direct = self.completions_from_summary(prefix, &[], node, depth, k);
                if !direct.is_empty() {
                    return direct;
                }
                let missing_summary = node.heavy.as_ref().map_or(true, |hh| hh.is_empty());
                if node.children.len() == 1 && (node.heavy.is_none() || missing_summary) {
                    let edge = &node.children[0];
                    return self.completions_from_summary(
                        prefix,
                        &edge.label,
                        &edge.child,
                        depth + edge.label.len(),
                        k,
                    );
                }
                direct
            }
            PrefixMatch::MidEdge {
                child,
                remaining_label,
                depth,
            } => {
                let child_depth = depth + remaining_label.len();
                self.completions_from_summary(prefix, remaining_label, child, child_depth, k)
            }
        }
    }

    /// Removes entire subtrees whose root estimate is below `min_abs_estimate`.
    ///
    /// This is useful for reclaiming memory from low-signal prefixes while
    /// keeping the rest of the trie intact.
    pub fn prune_by_estimate(&mut self, min_abs_estimate: f64) -> usize {
        if min_abs_estimate <= 0.0 {
            return 0;
        }
        prune_node(
            &self.depth_table,
            self.heavy_capacity,
            &mut self.root,
            0,
            min_abs_estimate,
        )
    }

    /// Merges `other` into `self` by streaming raw sums from its trie.
    ///
    /// Both sketches must share the same [`alpha`](Self::alpha),
    /// [`hash_seed`](Self::hash_seed), and heavy-hitter capacity. The merge is
    /// linear in the number of prefixes realized by `other`.
    ///
    /// # Panics
    ///
    /// Panics if `alpha`, `hash_seed`, or heavy-hitter capacity differ because
    /// mismatched samplers would otherwise corrupt the estimates.
    pub fn merge_from(&mut self, other: &GpsSketch) {
        assert!((self.alpha() - other.alpha()).abs() < 1e-12);
        assert_eq!(self.hash_seed, other.hash_seed, "hash seed mismatch");
        assert_eq!(
            self.heavy_capacity, other.heavy_capacity,
            "HH capacity mismatch"
        );
        merge_nodes(&mut self.root, &other.root, self.heavy_capacity);
    }

    fn raw_sum(&self, prefix: &[u8]) -> Option<(f64, usize)> {
        locate_raw_sum(&self.root, prefix, 0)
    }

    fn scale_raw_estimate(&self, raw: f64, depth: usize) -> f64 {
        if depth > self.depth_table.max_realizable_depth() {
            return 0.0;
        }
        let inv = self.depth_table.inv_prob(depth);
        if inv == 0.0 { 0.0 } else { raw * inv }
    }

    fn completions_from_summary(
        &self,
        base_prefix: &[u8],
        forced_suffix: &[u8],
        node: &Node,
        summary_depth: usize,
        k: usize,
    ) -> Vec<(Vec<u8>, f64)> {
        let Some(sketch) = &node.heavy else {
            return Vec::new();
        };
        let inv = if summary_depth > self.depth_table.max_realizable_depth() {
            0.0
        } else {
            self.depth_table.inv_prob(summary_depth)
        };
        let mut items: Vec<_> = sketch
            .top_k(k)
            .into_iter()
            .map(|(suffix, weight)| {
                let mut full =
                    Vec::with_capacity(base_prefix.len() + forced_suffix.len() + suffix.len());
                full.extend_from_slice(base_prefix);
                full.extend_from_slice(forced_suffix);
                full.extend_from_slice(&suffix);
                let est = if inv == 0.0 { 0.0 } else { weight * inv };
                (full, est)
            })
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items
    }

    fn collect_estimates(
        &self,
        node: &Node,
        depth: usize,
        prefix: &mut Vec<u8>,
        out: &mut Vec<(Vec<u8>, f64)>,
    ) {
        let estimate = self.scale_raw_estimate(node.sum, depth);
        out.push((prefix.clone(), estimate));
        for edge in &node.children {
            for (i, &byte) in edge.label.iter().enumerate() {
                prefix.push(byte);
                let new_depth = depth + i + 1;
                if i + 1 == edge.label.len() {
                    self.collect_estimates(&edge.child, new_depth, prefix, out);
                } else {
                    let raw = edge.mid_sums[i];
                    let est = self.scale_raw_estimate(raw, new_depth);
                    out.push((prefix.clone(), est));
                }
                prefix.pop();
            }
        }
    }

    #[cfg(test)]
    fn add_raw_sum(&mut self, prefix: &[u8], delta: f64) {
        if prefix.is_empty() {
            self.root.sum += delta;
            return;
        }
        let mut node = &mut self.root;
        let mut depth = 0;
        while depth < prefix.len() {
            if depth < PROMOTION_DEPTH {
                let byte = prefix[depth];
                node = node.ensure_unit_child(byte, self.heavy_capacity);
                depth += 1;
                if depth == prefix.len() {
                    node.sum += delta;
                    return;
                }
                continue;
            }

            let remaining = &prefix[depth..];
            let edge_idx = ensure_edge(node, remaining, self.heavy_capacity);
            let edge_len = node.children[edge_idx].label.len();
            let consume = (prefix.len() - depth).min(edge_len);
            {
                let edge = &mut node.children[edge_idx];
                edge.add_weight(consume, delta);
            }
            depth += consume;
            if consume == edge_len {
                let edge = &mut node.children[edge_idx];
                node = edge.child.as_mut();
                if depth == prefix.len() {
                    node.sum += delta;
                    return;
                }
            } else {
                return;
            }
        }
    }

    #[cfg(test)]
    fn visit_raw<F>(&self, prefix: &mut Vec<u8>, f: &mut F)
    where
        F: FnMut(&[u8], f64),
    {
        visit_raw(&self.root, prefix, f);
    }

    fn prefix_budget(&self, key: &[u8]) -> usize {
        if key.is_empty() {
            return 0;
        }
        let hash = deterministic_hash(key, self.hash_seed);
        self.sample_level_for_hash(hash, key.len())
    }

    fn sample_level_for_hash(&self, hash: u128, max_depth: usize) -> usize {
        if max_depth == 0 {
            return 0;
        }
        let depth_cap = max_depth.min(self.depth_table.max_realizable_depth());
        if depth_cap == 0 {
            return 0;
        }
        if depth_cap == 1 {
            return 1;
        }
        if self.depth_table.is_half_sampler() {
            let leading = hash.leading_zeros() as usize;
            return (1 + leading).min(depth_cap);
        }
        let mut limit = 1;
        for depth in 2..=depth_cap {
            let threshold = self.depth_table.q128(depth);
            if threshold == 0 {
                break;
            }
            if hash < threshold {
                limit = depth;
            } else {
                break;
            }
        }
        limit
    }

    #[cfg(test)]
    fn raw_sum_for_test(&self, prefix: &[u8]) -> Option<f64> {
        self.raw_sum(prefix).map(|(raw, _)| raw)
    }

    #[cfg(test)]
    fn debug_prefix_budget(&self, key: &[u8]) -> usize {
        self.prefix_budget(key)
    }

    #[cfg(test)]
    fn debug_sample_level_from_bits(&self, bits: u128) -> usize {
        const DEBUG_MAX_DEPTH: usize = 10_000;
        self.sample_level_for_hash(bits, DEBUG_MAX_DEPTH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use std::collections::HashMap;

    fn force_full_depth_insert(sketch: &mut GpsSketch, key: &[u8], delta: f64) {
        if delta == 0.0 {
            return;
        }
        sketch.root.sum += delta;
        sketch.root.update_heavy_hitters(key, delta);
        super::add_inner(
            &mut sketch.root,
            key,
            0,
            key.len(),
            delta,
            sketch.heavy_capacity,
        );
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
    }

    fn zero_out_tree(node: &mut Node, heavy_capacity: Option<usize>) {
        node.sum = 0.0;
        node.heavy = heavy_capacity.map(TopKCompactor::new);
        for edge in &mut node.children {
            for raw in &mut edge.mid_sums {
                *raw = 0.0;
            }
            zero_out_tree(edge.child.as_mut(), heavy_capacity);
        }
    }

    fn sort_completions(entries: &mut Vec<(Vec<u8>, f64)>) {
        entries.sort_by(|a, b| {
            let key_cmp = a.0.cmp(&b.0);
            if key_cmp == std::cmp::Ordering::Equal {
                a.1.partial_cmp(&b.1).unwrap()
            } else {
                key_cmp
            }
        });
    }

    #[test]
    fn unreachable_depth_estimates_to_zero() {
        let mut sketch = GpsSketch::with_seed(0.9, 7);
        let max_depth = sketch.depth_table.max_realizable_depth();
        let target_depth = max_depth + 5;
        let key = vec![b'x'; target_depth];
        sketch.add_raw_sum(&key, 12.0);
        assert_eq!(sketch.estimate(&key), 0.0);
    }

    #[test]
    fn depth_one_counts_are_exact() {
        let mut sketch = GpsSketch::default();
        sketch.add("apple", 1.0);
        sketch.add("apricot", 2.0);
        sketch.add("banana", 3.0);
        assert_close(sketch.estimate("a"), 3.0, 1e-9);
        assert_close(sketch.estimate("b"), 3.0, 1e-9);
    }

    #[test]
    fn depth_one_counts_are_exact_for_general_alpha() {
        let configs = [(0.7, 11u64), (0.9, 29u64)];
        for &(alpha, seed) in &configs {
            let mut sketch = GpsSketch::with_seed(alpha, seed);
            let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(17));
            let mut counts: HashMap<u8, f64> = HashMap::new();
            for _ in 0..4_000 {
                let mut key = vec![0u8; 16];
                rng.fill_bytes(&mut key);
                sketch.add(&key, 1.0);
                *counts.entry(key[0]).or_insert(0.0) += 1.0;
            }
            for (byte, count) in counts {
                assert_close(sketch.estimate([byte]), count, 1e-12);
            }
        }
    }

    #[test]
    fn merge_matches_single_pass() {
        let mut left = GpsSketch::with_seed(0.5, 42);
        let mut right = GpsSketch::with_seed(0.5, 42);
        let mut combined = GpsSketch::with_seed(0.5, 42);
        let keys = [
            "alpha", "alpine", "beta", "betamax", "gamma", "garden", "alphabet",
        ];
        for (i, key) in keys.iter().enumerate() {
            if i % 2 == 0 {
                left.add(key, 1.0);
            } else {
                right.add(key, 1.0);
            }
            combined.add(key, 1.0);
        }
        let mut merged = left.clone();
        merged.merge_from(&right);
        for prefix in ["", "a", "al", "b", "ga", "alphabet"] {
            assert_close(merged.estimate(prefix), combined.estimate(prefix), 1e-9);
        }
    }

    #[test]
    fn merge_splits_compressed_edges_when_needed() {
        let mut left = GpsSketch::with_seed(0.5, 7);
        let mut right = GpsSketch::with_seed(0.5, 7);
        let mut combined = GpsSketch::with_seed(0.5, 7);

        force_full_depth_insert(&mut left, b"abcdefg", 1.0);
        force_full_depth_insert(&mut right, b"abcdexy", 1.0);
        force_full_depth_insert(&mut combined, b"abcdefg", 1.0);
        force_full_depth_insert(&mut combined, b"abcdexy", 1.0);

        let mut merged = left.clone();
        merged.merge_from(&right);

        for prefix in ["abcde", "abcdef", "abcdex"] {
            assert_close(merged.estimate(prefix), combined.estimate(prefix), 1e-9);
        }
    }

    #[test]
    fn merge_boundary_mid_carries_overlap() {
        let mut left = GpsSketch::with_seed(0.5, 19);
        let mut right = GpsSketch::with_seed(0.5, 19);
        let mut combined = GpsSketch::with_seed(0.5, 19);

        force_full_depth_insert(&mut left, b"abcdefg", 2.0);
        force_full_depth_insert(&mut right, b"abcdexy", 3.0);
        force_full_depth_insert(&mut combined, b"abcdefg", 2.0);
        force_full_depth_insert(&mut combined, b"abcdexy", 3.0);

        let mut merged = left.clone();
        merged.merge_from(&right);

        for prefix in ["abcd", "abcde", "abcdef", "abcdex"] {
            assert_close(merged.estimate(prefix), combined.estimate(prefix), 1e-9);
        }
    }

    #[test]
    fn empty_key_only_touches_root() {
        let mut sketch = GpsSketch::default();
        sketch.add(Vec::<u8>::new(), 5.0);
        assert_close(sketch.total(), 5.0, 1e-12);
        assert_eq!(sketch.node_count(), 1);
    }

    #[test]
    fn sampling_is_deterministic() {
        let sketch = GpsSketch::new(0.6);
        let key = b"deterministic";
        let budget = sketch.debug_prefix_budget(key);
        for _ in 0..10 {
            assert_eq!(sketch.debug_prefix_budget(key), budget);
        }
    }

    #[test]
    fn prefix_budget_matches_across_instances() {
        let mut rng = StdRng::seed_from_u64(99);
        let sketch_a = GpsSketch::with_seed(0.73, 123);
        let sketch_b = GpsSketch::with_seed(0.73, 123);
        for _ in 0..200 {
            let len = rng.gen_range(4..32);
            let mut key = vec![0u8; len];
            rng.fill_bytes(&mut key);
            let budget_a = sketch_a.debug_prefix_budget(&key);
            let budget_b = sketch_b.debug_prefix_budget(&key);
            assert_eq!(budget_a, budget_b);
        }
    }

    #[test]
    fn raw_sums_match_manual_reference() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut sketch = GpsSketch::with_seed(0.65, 123);
        let mut records = Vec::new();
        for _ in 0..400 {
            let len = rng.gen_range(1..=8);
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                bytes.push(rng.gen_range(b'a'..=b'f'));
            }
            let delta = rng.gen_range(-3..=5) as f64;
            sketch.add(&bytes, delta);
            records.push((bytes, delta));
        }
        let prefixes = [
            &b""[..],
            &b"a"[..],
            &b"ab"[..],
            &b"abc"[..],
            &b"abcd"[..],
            &b"abcde"[..],
        ];
        for prefix in prefixes {
            let expected = manual_truncated_sum(prefix, &records, &sketch);
            let actual = sketch.raw_sum_for_test(prefix).unwrap_or(0.0);
            assert_close(actual, expected, 1e-9);
            let est = sketch.estimate(prefix);
            let q = inclusion_prob(sketch.alpha(), prefix.len());
            if q == 0.0 {
                assert_close(est, 0.0, 1e-9);
            } else {
                assert_close(est, expected / q, 1e-6);
            }
        }
    }

    fn manual_truncated_sum(prefix: &[u8], records: &[(Vec<u8>, f64)], sketch: &GpsSketch) -> f64 {
        let depth = prefix.len();
        let mut acc = 0.0;
        for (key, delta) in records {
            if depth > key.len() {
                continue;
            }
            if !key.starts_with(prefix) {
                continue;
            }
            let limit = sketch.debug_prefix_budget(key);
            if limit >= depth {
                acc += delta;
            }
        }
        acc
    }

    #[test]
    fn prune_drops_small_prefixes() {
        let mut sketch = GpsSketch::default();
        sketch.add("a", 0.25);
        sketch.add("b", 2.0);
        sketch.add("c", 0.2);
        let removed = sketch.prune_by_estimate(0.5);
        assert!(removed >= 1);
        assert!(!sketch.contains_prefix("a"));
        assert!(sketch.contains_prefix("b"));
    }

    #[test]
    fn prune_removes_compressed_subtrees() {
        let sketch = GpsSketch::default();
        let keeper = find_key_with_budget_at_least(&sketch, 4);
        let mut sketch = sketch;
        sketch.add_raw_sum(b"abcdefghijkl", 1e-6);
        sketch.add(&keeper, 10.0);
        assert!(sketch.contains_prefix(&keeper[..1]));
        let before = sketch.node_count();
        let removed = sketch.prune_by_estimate(0.5);
        assert!(removed > 0);
        assert!(sketch.node_count() < before);
        assert!(!sketch.contains_prefix("abcdefghijkl"));
        assert!(sketch.contains_prefix(&keeper[..1]));
    }

    #[test]
    fn iter_estimates_reports_mid_edge_prefixes() {
        let mut sketch = GpsSketch::default();
        sketch.add_raw_sum(b"abcd", 1.0);
        let entries: HashMap<_, _> = sketch.iter_estimates().collect();
        assert!(entries.contains_key(b"abcd" as &[u8]));
    }

    #[test]
    fn iter_estimates_match_raw_traversal() {
        let mut sketch = GpsSketch::with_seed(0.55, 7);
        for (i, key) in ["alpha", "alphabet", "alpine", "beta", "betamax", "gamma"]
            .iter()
            .enumerate()
        {
            sketch.add(key, (i + 1) as f64);
        }

        let iter_map: HashMap<Vec<u8>, f64> = sketch.iter_estimates().collect();
        let mut visit_map = HashMap::new();
        let mut prefix = Vec::new();
        sketch.visit_raw(&mut prefix, &mut |pref, raw| {
            let depth = pref.len();
            let q = inclusion_prob(sketch.alpha(), depth);
            let est = if q == 0.0 { 0.0 } else { raw / q };
            visit_map.insert(pref.to_vec(), est);
        });

        assert_eq!(iter_map.len(), visit_map.len());
        for (k, v) in visit_map {
            let iter_v = *iter_map
                .get(&k)
                .expect("missing prefix from iterator traversal");
            assert_close(iter_v, v, 1e-9);
        }
    }

    #[test]
    fn contains_prefix_handles_mid_edge() {
        let sketch = GpsSketch::default();
        let key = find_key_with_budget_at_least(&sketch, 8);
        let mut sketch = sketch;
        sketch.add(&key, 1.0);
        assert!(sketch.contains_prefix(&key[..4]));
        assert!(sketch.contains_prefix(&key[..]));
        assert!(!sketch.contains_prefix("xyz"));
    }

    #[test]
    fn heavy_hitters_track_top_suffixes() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        for _ in 0..50 {
            sketch.add("dog", 1.0);
        }
        for _ in 0..20 {
            sketch.add("door", 1.0);
        }
        for _ in 0..10 {
            sketch.add("doll", 1.0);
        }
        let tops = sketch.top_completions("d", 3);
        assert_eq!(tops.len(), 3);
        assert_eq!(tops[0].0, b"dog".to_vec());
        assert!(tops[0].1 >= tops[1].1);
    }

    #[test]
    fn heavy_hitters_scale_by_inclusion_prob() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        for _ in 0..50 {
            sketch.add("xyza", 1.0);
        }
        let tops = sketch.top_completions("x", 1);
        assert_eq!(tops[0].0, b"xyza".to_vec());
        assert_close(tops[0].1, 50.0, 1e-6);
    }

    #[test]
    fn merge_with_heavy_hitters_matches_combined() {
        let mut left = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        let mut right = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        let mut combined = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        let samples = [
            ("cat", 5.0),
            ("car", 3.0),
            ("cart", 4.0),
            ("dog", 2.0),
            ("cape", 1.0),
        ];
        for (idx, &(key, weight)) in samples.iter().enumerate() {
            if idx % 2 == 0 {
                left.add(key, weight);
            } else {
                right.add(key, weight);
            }
            combined.add(key, weight);
        }

        let mut merged = left.clone();
        merged.merge_from(&right);

        let merged_top = merged.top_completions("ca", 4);
        let combined_top = combined.top_completions("ca", 4);
        assert_eq!(merged_top.len(), combined_top.len());
        for (m, c) in merged_top.iter().zip(combined_top.iter()) {
            assert_eq!(m.0, c.0);
            assert_close(m.1, c.1, 1e-9);
        }
    }

    #[test]
    fn heavy_hitters_cover_mid_edge_prefixes() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        force_full_depth_insert(&mut sketch, b"abcdefg", 4.0);
        force_full_depth_insert(&mut sketch, b"abcdefgh", 2.0);
        let tops = sketch.top_completions("abcde", 4);
        let keys: Vec<_> = tops.into_iter().map(|(k, _)| k).collect();
        assert!(keys.contains(&b"abcdefg"[..].to_vec()));
        assert!(keys.contains(&b"abcdefgh"[..].to_vec()));
    }

    #[test]
    fn heavy_hitters_return_original_keys_for_prefix() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.5, 21, 16);
        let keys: &[&[u8]] = &[
            b"/abmid_edge_alpha",
            b"/abmid_edge_beta",
            b"/abshort",
            b"/abdeeper_branch_value",
        ];
        for (idx, key) in keys.iter().enumerate() {
            force_full_depth_insert(&mut sketch, key, (idx + 1) as f64);
        }
        let completions = sketch.top_completions("/ab", keys.len());
        assert_eq!(completions.len(), keys.len());
        for (full, _) in completions {
            assert!(full.starts_with(b"/ab"));
            assert!(
                keys.iter()
                    .any(|original| original.as_ref() == full.as_slice()),
                "missing completion {:?}",
                full
            );
        }
    }

    #[test]
    fn heavy_hitter_labels_survive_merge() {
        let keys: &[(&[u8], f64)] = &[
            (b"/abcommon_tail_cat", 5.0),
            (b"/abcommon_tail_dog", 4.0),
            (b"/abbranch_left", 3.0),
            (b"/abbranch_right", 2.5),
            (b"/abdepth_mid_edge_suffix", 2.0),
            (b"/abx_trailing", 1.5),
        ];
        let mut left = GpsSketch::with_heavy_hitters(0.5, 8, 16);
        let mut right = GpsSketch::with_heavy_hitters(0.5, 8, 16);
        let mut combined = GpsSketch::with_heavy_hitters(0.5, 8, 16);
        for (idx, &(key, weight)) in keys.iter().enumerate() {
            if idx % 2 == 0 {
                force_full_depth_insert(&mut left, key, weight);
            } else {
                force_full_depth_insert(&mut right, key, weight);
            }
            force_full_depth_insert(&mut combined, key, weight);
        }
        let mut merged = left.clone();
        merged.merge_from(&right);

        let mut merged_top = merged.top_completions("/ab", keys.len());
        let mut combined_top = combined.top_completions("/ab", keys.len());
        sort_completions(&mut merged_top);
        sort_completions(&mut combined_top);
        assert_eq!(merged_top.len(), combined_top.len());
        for (lhs, rhs) in merged_top.iter().zip(combined_top.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert_close(lhs.1, rhs.1, 1e-9);
        }
    }

    #[test]
    fn mid_edge_top_completions_use_child_depth() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.5, 11, 8);
        force_full_depth_insert(&mut sketch, b"abcdefgh", 8.0);
        let tops = sketch.top_completions("abcde", 1);
        assert_eq!(tops.len(), 1);
        assert_eq!(tops[0].0, b"abcdefgh".to_vec());
        let point_estimate = sketch.estimate("abcdefgh");
        assert_close(tops[0].1, point_estimate, 1e-9);
    }

    #[test]
    fn mid_edge_scaling_holds_for_general_alpha() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.7, 3, 8);
        force_full_depth_insert(&mut sketch, b"mnopqrst", 10.0);
        let tops = sketch.top_completions("mnopq", 1);
        assert_eq!(tops.len(), 1);
        assert_eq!(tops[0].0, b"mnopqrst".to_vec());
        assert_close(tops[0].1, sketch.estimate("mnopqrst"), 1e-9);
    }

    #[test]
    fn mid_edge_completions_concatenate_remaining_label() {
        let mut sketch = GpsSketch::with_heavy_hitters(0.6, 5, 8);
        force_full_depth_insert(&mut sketch, b"abcdefgh", 6.0);
        let prefix = b"abcde";
        let tops = sketch.top_completions(prefix, 2);
        assert_eq!(tops.len(), 1);
        assert!(tops[0].0.starts_with(prefix));
        assert_eq!(tops[0].0, b"abcdefgh".to_vec());
        assert_close(tops[0].1, sketch.estimate("abcdefgh"), 1e-9);
    }

    #[test]
    fn top_completions_survive_structural_split() {
        let seed = 13;
        let hh_cap = 8;
        let mut sketch = GpsSketch::with_heavy_hitters(0.5, seed, hh_cap);
        force_full_depth_insert(&mut sketch, b"abcdefgh", 5.0);
        let prefix = b"abcde";
        let mut before = sketch.top_completions(prefix, 2);
        assert!(!before.is_empty());
        sort_completions(&mut before);

        let mut splitter = GpsSketch::with_heavy_hitters(0.5, seed, hh_cap);
        force_full_depth_insert(&mut splitter, b"abcde", 1.0);
        zero_out_tree(&mut splitter.root, splitter.heavy_capacity);

        sketch.merge_from(&splitter);

        let mut after = sketch.top_completions(prefix, 2);
        sort_completions(&mut after);
        assert_eq!(before.len(), after.len());
        for (lhs, rhs) in before.iter().zip(after.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert_close(lhs.1, rhs.1, 1e-9);
        }
    }

    #[test]
    fn top_k_compactor_merge_is_order_invariant() {
        let mut a = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        let mut b = GpsSketch::with_heavy_hitters(0.5, 0, 4);
        for _ in 0..100 {
            a.add("apricot", 1.0);
        }
        for _ in 0..60 {
            a.add("apple", 1.0);
        }
        for _ in 0..55 {
            b.add("apricot", 1.0);
        }
        for _ in 0..70 {
            b.add("ape", 1.0);
        }

        let mut m1 = a.clone();
        m1.merge_from(&b);
        let mut m2 = b.clone();
        m2.merge_from(&a);

        let t1 = m1.top_completions("a", 3);
        let t2 = m2.top_completions("a", 3);
        assert_eq!(t1.len(), t2.len());
        for (lhs, rhs) in t1.iter().zip(t2.iter()) {
            assert_eq!(lhs.0, rhs.0);
        }
    }

    #[test]
    fn alpha_point_five_sampler_matches_leading_zeros() {
        let sketch = GpsSketch::new(0.5);
        for bitpos in 0..=128 {
            let hash = if bitpos == 128 {
                0
            } else {
                1u128 << (127 - bitpos)
            };
            let level = sketch.debug_sample_level_from_bits(hash);
            assert_eq!(level, 1 + bitpos);
        }
    }

    #[test]
    fn sampler_can_hit_very_deep_levels() {
        let sketch = GpsSketch::new(0.5);
        let level = sketch.debug_sample_level_from_bits(0);
        assert!(level >= 120);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn general_alpha_sampler_matches_tail_probabilities() {
        const TRIALS: usize = 200_000;
        for &(alpha, seed) in &[(0.7, 31u64), (0.9, 37u64)] {
            let sketch = GpsSketch::new(alpha);
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tail_counts = vec![0usize; 65];
            for _ in 0..TRIALS {
                let bits = ((rng.next_u64() as u128) << 64) | (rng.next_u64() as u128);
                let level = sketch.debug_sample_level_from_bits(bits).min(64);
                for depth in 1..=level {
                    tail_counts[depth] += 1;
                }
            }
            for depth in 1..=64 {
                let empirical = tail_counts[depth] as f64 / TRIALS as f64;
                let expected = alpha.powi((depth - 1) as i32);
                assert!(
                    (empirical - expected).abs() <= 2e-3,
                    "alpha={alpha}, depth={depth}, empirical={empirical}, expected={expected}"
                );
            }
            let mut prev = u128::MAX;
            for depth in 1..=64 {
                let threshold = sketch.depth_table.q128(depth);
                assert!(
                    threshold <= prev,
                    "q128 not monotone at depth {depth}: {} > {}",
                    threshold,
                    prev
                );
                prev = threshold;
            }
        }
    }

    #[test]
    fn negative_updates_cancel_out() {
        let sketch = GpsSketch::default();
        let key = find_key_with_budget_at_least(&sketch, 5);
        let mut sketch = sketch;
        sketch.add(&key, 10.0);
        sketch.add(&key, -10.0);
        assert_close(sketch.raw_sum_for_test(&key).unwrap_or(0.0), 0.0, 1e-9);
        assert_close(sketch.total(), 0.0, 1e-9);
    }

    #[test]
    #[should_panic(expected = "hash seed mismatch")]
    fn merge_requires_matching_seeds() {
        let mut a = GpsSketch::with_seed(0.5, 1);
        let b = GpsSketch::with_seed(0.5, 2);
        a.merge_from(&b);
    }

    #[test]
    #[should_panic(expected = "HH capacity mismatch")]
    fn merge_requires_matching_hh_capacity() {
        let mut a = GpsSketch::with_heavy_hitters(0.5, 0, 2);
        let b = GpsSketch::with_seed(0.5, 0);
        a.merge_from(&b);
    }

    fn find_key_with_budget_at_least(sketch: &GpsSketch, min_budget: usize) -> Vec<u8> {
        for idx in 0..50_000 {
            let candidate = format!("key-{idx}-payload");
            if sketch.debug_prefix_budget(candidate.as_bytes()) >= min_budget {
                return candidate.into_bytes();
            }
        }
        panic!("unable to find key with budget >= {min_budget}");
    }
}
