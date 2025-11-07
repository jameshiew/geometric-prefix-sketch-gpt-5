//! Geometric Prefix Sketch (GPS) implementation with a compressed radix trie.
//!
//! The sketch follows the design laid out in `DESIGN.md`: every update touches a
//! geometrically sampled prefix depth, accumulators remain unbiased, sketches
//! merge additively, and (optional) per-node heavy-hitter summaries expose top
//! completions. The trie now uses *compressed edges* so long, single-child runs
//! past a configurable promotion depth occupy a single node/edge instead of one
//! heap allocation per byte.

use crate::tree::{
    Node, PROMOTION_DEPTH, SpaceSaving, add_inner, ensure_edge, ensure_node, locate_node,
    locate_raw_sum, prune_node, visit_heavy, visit_raw,
};
use crate::util::{HASH128_TO_UNIT, deterministic_hash, inclusion_prob};

/// Geometric Prefix Sketch with deterministic per-key sampling.
///
/// `GpsSketch` stores a compressed prefix trie whose nodes are populated by a
/// geometric (per-key) sampling scheme. Updates touch only a constant expected
/// number of prefixes while estimates remain unbiased. Instances are
/// merge-friendly as long as they share the same [`alpha`](Self::alpha) and
/// [`hash_seed`](Self::hash_seed).
#[derive(Clone, Debug)]
pub struct GpsSketch {
    alpha: f64,
    log_alpha: f64,
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
    /// so their sketches can be merged later.
    pub fn with_seed(alpha: f64, hash_seed: u64) -> Self {
        Self::with_heavy_hitters_internal(alpha, hash_seed, None)
    }

    /// Creates a sketch with per-node heavy-hitter capacity.
    ///
    /// Heavy hitters are tracked via a SpaceSaving summary stored on each
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
        Self {
            alpha,
            log_alpha: alpha.ln(),
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
        self.alpha
    }

    /// Returns the deterministic hash seed.
    ///
    /// All sketches that need to merge must share this seed.
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
            Some((raw, depth)) => raw / inclusion_prob(self.alpha, depth),
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
    /// estimated weight.
    pub fn top_completions<K: AsRef<[u8]>>(&self, prefix: K, k: usize) -> Vec<(Vec<u8>, f64)> {
        if k == 0 {
            return Vec::new();
        }
        let prefix = prefix.as_ref();
        let Some((node, depth)) = self.find_node(prefix) else {
            return Vec::new();
        };
        let Some(sketch) = &node.heavy else {
            return Vec::new();
        };
        let q = inclusion_prob(self.alpha, depth);
        let mut items: Vec<_> = sketch
            .top_k(k)
            .into_iter()
            .map(|(suffix, weight)| {
                let mut full = Vec::with_capacity(prefix.len() + suffix.len());
                full.extend_from_slice(prefix);
                full.extend_from_slice(&suffix);
                (full, weight / q)
            })
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items
    }

    /// Removes entire subtrees whose root estimate is below `min_abs_estimate`.
    ///
    /// This is useful for reclaiming memory from low-signal prefixes while
    /// keeping the rest of the trie intact.
    pub fn prune_by_estimate(&mut self, min_abs_estimate: f64) -> usize {
        if min_abs_estimate <= 0.0 {
            return 0;
        }
        prune_node(self.alpha, &mut self.root, 0, min_abs_estimate)
    }

    /// Merges `other` into `self` by streaming raw sums from its trie.
    ///
    /// Both sketches must share the same [`alpha`](Self::alpha),
    /// [`hash_seed`](Self::hash_seed), and heavy-hitter capacity. The merge is
    /// linear in the number of prefixes realized by `other`.
    pub fn merge_from(&mut self, other: &GpsSketch) {
        assert!((self.alpha - other.alpha).abs() < 1e-12);
        assert_eq!(self.hash_seed, other.hash_seed, "hash seed mismatch");
        assert_eq!(
            self.heavy_capacity, other.heavy_capacity,
            "HH capacity mismatch"
        );

        let mut prefix = Vec::new();
        other.visit_raw(&mut prefix, &mut |pref, sum| {
            self.add_raw_sum(pref, sum);
        });

        if let Some(cap) = self.heavy_capacity {
            other.visit_heavy(&mut prefix, &mut |pref, sketch| {
                if let Some(node) = self.ensure_node(pref) {
                    let hh = node.heavy.get_or_insert_with(|| SpaceSaving::new(cap));
                    for (suffix, weight) in sketch.iter_entries() {
                        hh.update(suffix, weight);
                    }
                }
            });
        }
    }

    fn raw_sum(&self, prefix: &[u8]) -> Option<(f64, usize)> {
        locate_raw_sum(&self.root, prefix, 0)
    }

    fn find_node(&self, prefix: &[u8]) -> Option<(&Node, usize)> {
        locate_node(&self.root, prefix, 0)
    }

    fn ensure_node(&mut self, prefix: &[u8]) -> Option<&mut Node> {
        ensure_node(&mut self.root, prefix, self.heavy_capacity)
    }

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
            let edge = ensure_edge(node, remaining, self.heavy_capacity);
            let consume = (prefix.len() - depth).min(edge.label.len());
            edge.add_weight(consume, delta);
            depth += consume;
            if consume == edge.label.len() {
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

    fn collect_estimates(
        &self,
        node: &Node,
        depth: usize,
        prefix: &mut Vec<u8>,
        out: &mut Vec<(Vec<u8>, f64)>,
    ) {
        let estimate = node.sum / inclusion_prob(self.alpha, depth);
        out.push((prefix.clone(), estimate));
        for edge in &node.children {
            for (i, &byte) in edge.label.iter().enumerate() {
                prefix.push(byte);
                let new_depth = depth + i + 1;
                if i + 1 == edge.label.len() {
                    self.collect_estimates(&edge.child, new_depth, prefix, out);
                } else {
                    let raw = edge.mid_sums[i];
                    let est = raw / inclusion_prob(self.alpha, new_depth);
                    out.push((prefix.clone(), est));
                }
                prefix.pop();
            }
        }
    }

    fn visit_raw<F>(&self, prefix: &mut Vec<u8>, f: &mut F)
    where
        F: FnMut(&[u8], f64),
    {
        visit_raw(&self.root, prefix, f);
    }

    fn visit_heavy<F>(&self, prefix: &mut Vec<u8>, f: &mut F)
    where
        F: FnMut(&[u8], &SpaceSaving),
    {
        visit_heavy(&self.root, prefix, f);
    }

    fn prefix_budget(&self, key: &[u8]) -> usize {
        if key.is_empty() {
            return 0;
        }
        let hash = deterministic_hash(key, self.hash_seed);
        let level = self.sample_level_for_hash(hash);
        level.min(key.len())
    }

    fn sample_level_for_hash(&self, hash: u128) -> usize {
        let uniform = ((hash as f64) + 1.0) * HASH128_TO_UNIT;
        let raw = 1.0 + (uniform.ln() / self.log_alpha).floor();
        if raw.is_finite() && raw >= 1.0 {
            raw as usize
        } else {
            1
        }
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
        self.sample_level_for_hash(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::collections::HashMap;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
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
    fn merge_matches_single_pass() {
        let mut left = GpsSketch::default();
        let mut right = GpsSketch::default();
        let mut combined = GpsSketch::default();
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
            assert_close(est, expected / q, 1e-6);
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
            let est = raw / inclusion_prob(sketch.alpha(), depth);
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
    fn sampler_can_hit_very_deep_levels() {
        let sketch = GpsSketch::new(0.5);
        let level = sketch.debug_sample_level_from_bits(0);
        assert!(level >= 120);
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
