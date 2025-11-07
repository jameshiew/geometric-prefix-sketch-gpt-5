//! Geometric Prefix Sketch (GPS) implementation.
//!
//! This crate implements the sketch described in `DESIGN.md`. The sketch keeps
//! unbiased prefix aggregates while touching only a geometrically sampled
//! portion of each key's prefixes, so updates take constant expected time.

use std::collections::HashMap;

use xxhash_rust::xxh3::xxh3_128_with_seed;

const HASH128_TO_UNIT: f64 = 1.0 / ((u128::MAX as f64) + 1.0);

#[derive(Clone, Debug, Default)]
struct Node {
    sum: f64,
}

/// Geometric Prefix Sketch with deterministic per-key sampling.
#[derive(Clone, Debug)]
pub struct GpsSketch {
    alpha: f64,
    log_alpha: f64,
    hash_seed: u64,
    nodes: HashMap<Vec<u8>, Node>,
}

impl Default for GpsSketch {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl GpsSketch {
    /// Creates a new sketch with the provided geometric parameter `alpha`.
    ///
    /// `alpha` must lie strictly in `(0, 1)`.
    pub fn new(alpha: f64) -> Self {
        Self::with_seed(alpha, 0)
    }

    /// Creates a new sketch with an explicit deterministic hash seed.
    /// All sketches that need to merge must share the same seed.
    pub fn with_seed(alpha: f64, hash_seed: u64) -> Self {
        assert!(alpha.is_finite());
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        let mut nodes = HashMap::new();
        nodes.insert(Vec::new(), Node::default());
        Self {
            alpha,
            log_alpha: alpha.ln(),
            hash_seed,
            nodes,
        }
    }

    /// Returns the configured `alpha`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the seed used by the deterministic hash function.
    pub fn hash_seed(&self) -> u64 {
        self.hash_seed
    }

    /// Adds `delta` to `key`.
    pub fn add<K: AsRef<[u8]>>(&mut self, key: K, delta: f64) {
        if delta == 0.0 {
            return;
        }

        let key = key.as_ref();
        let limit = self.prefix_budget(key);
        self.root_mut().sum += delta;

        if limit == 0 {
            return;
        }

        let mut prefix = Vec::with_capacity(limit);
        for (depth, &byte) in key.iter().enumerate() {
            prefix.push(byte);
            let prefix_depth = depth + 1;
            if prefix_depth > limit {
                break;
            }

            let node = self.nodes.entry(prefix.clone()).or_default();
            node.sum += delta;

            if prefix_depth == limit {
                break;
            }
        }
    }

    /// Returns the Horvitz–Thompson estimate of the sum under `prefix`.
    pub fn estimate<K: AsRef<[u8]>>(&self, prefix: K) -> f64 {
        let prefix = prefix.as_ref();
        let depth = prefix.len();
        match self.nodes.get(prefix) {
            Some(node) => {
                let q = self.prefix_inclusion_prob(depth);
                node.sum / q
            }
            None => 0.0,
        }
    }

    /// Returns the global total (estimate at the root prefix).
    pub fn total(&self) -> f64 {
        self.estimate(Vec::<u8>::new())
    }

    /// Removes all data from the sketch.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.insert(Vec::new(), Node::default());
    }

    /// Number of materialized prefixes (including root).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether `prefix` currently has a node in the trie.
    pub fn contains_prefix<K: AsRef<[u8]>>(&self, prefix: K) -> bool {
        self.nodes.contains_key(prefix.as_ref())
    }

    /// Returns an iterator over all materialized prefixes along with their estimates.
    pub fn iter_estimates(&self) -> impl Iterator<Item = (&[u8], f64)> + '_ {
        self.nodes.iter().map(move |(prefix, node)| {
            let estimate = node.sum / self.prefix_inclusion_prob(prefix.len());
            (prefix.as_slice(), estimate)
        })
    }

    /// Removes all non-root prefixes whose estimated absolute value is below `min_abs_estimate`.
    /// Returns the number of nodes removed.
    pub fn prune_by_estimate(&mut self, min_abs_estimate: f64) -> usize {
        if min_abs_estimate <= 0.0 {
            return 0;
        }
        let mut to_remove = Vec::new();
        for (prefix, node) in &self.nodes {
            if prefix.is_empty() {
                continue;
            }
            let estimate = node.sum / self.prefix_inclusion_prob(prefix.len());
            if estimate.abs() < min_abs_estimate {
                to_remove.push(prefix.clone());
            }
        }
        let removed = to_remove.len();
        for prefix in to_remove {
            self.nodes.remove(&prefix);
        }
        removed
    }

    /// Merges `other` into `self` by pointwise addition.
    pub fn merge_from(&mut self, other: &GpsSketch) {
        assert!(
            (self.alpha - other.alpha).abs() < 1e-12 && self.hash_seed == other.hash_seed,
            "cannot merge sketches with mismatched alpha or hash seed"
        );
        for (prefix, node) in &other.nodes {
            let entry = self.nodes.entry(prefix.clone()).or_default();
            entry.sum += node.sum;
        }
    }

    fn root_mut(&mut self) -> &mut Node {
        self.nodes.entry(Vec::new()).or_default()
    }

    fn prefix_budget(&self, key: &[u8]) -> usize {
        if key.is_empty() {
            return 0;
        }
        let hash = deterministic_hash(key, self.hash_seed);
        let level = self.sample_level_for_hash(hash);
        level.min(key.len())
    }

    fn prefix_inclusion_prob(&self, depth: usize) -> f64 {
        if depth <= 1 {
            return 1.0;
        }
        let value = self.alpha.powf((depth - 1) as f64);
        if value < f64::MIN_POSITIVE {
            f64::MIN_POSITIVE
        } else {
            value
        }
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
    fn raw_node_sum(&self, prefix: &[u8]) -> Option<f64> {
        self.nodes.get(prefix).map(|node| node.sum)
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

fn deterministic_hash(bytes: &[u8], seed: u64) -> u128 {
    xxh3_128_with_seed(bytes, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::collections::HashMap;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "values differ: {a} vs {b} (tol {tol})"
        );
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
            assert_close(merged.estimate(prefix), combined.estimate(prefix), 1e-12);
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
    fn raw_node_sums_match_manual_reference() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut sketch = GpsSketch::with_seed(0.65, 123);
        let mut records = Vec::new();

        for _ in 0..400 {
            let len = rng.gen_range(1..=6);
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                bytes.push(rng.gen_range(b'a'..=b'd'));
            }
            let delta = rng.gen_range(-3..=5) as f64;
            sketch.add(&bytes, delta);
            records.push((bytes, delta));
        }

        let prefixes: [&[u8]; 6] = [
            &[],
            &b"a"[..],
            &b"ab"[..],
            &b"abc"[..],
            &b"c"[..],
            &b"cd"[..],
        ];

        for prefix in prefixes {
            let expected = manual_truncated_sum(prefix, &records, &sketch);
            let actual = sketch.raw_node_sum(prefix).unwrap_or(0.0);
            assert_close(actual, expected, 1e-9);

            let ht = manual_ht_estimate(prefix, &records, &sketch);
            assert_close(sketch.estimate(prefix), ht, 1e-9);
        }
    }

    fn manual_truncated_sum(prefix: &[u8], records: &[(Vec<u8>, f64)], sketch: &GpsSketch) -> f64 {
        let mut acc = 0.0;
        for (key, delta) in records {
            if !prefix.is_empty() && !key.starts_with(prefix) {
                continue;
            }
            if prefix.is_empty() {
                acc += delta;
                continue;
            }
            let limit = sketch.debug_prefix_budget(key);
            if limit >= prefix.len() {
                acc += delta;
            }
        }
        acc
    }

    fn manual_ht_estimate(prefix: &[u8], records: &[(Vec<u8>, f64)], sketch: &GpsSketch) -> f64 {
        let depth = prefix.len();
        if prefix.is_empty() {
            return records.iter().map(|(_, delta)| *delta).sum();
        }
        let q = sketch.prefix_inclusion_prob(depth);
        let mut acc = 0.0;
        for (key, delta) in records {
            if !key.starts_with(prefix) {
                continue;
            }
            let limit = sketch.debug_prefix_budget(key);
            if limit >= depth {
                acc += delta / q;
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
        assert_eq!(removed, 2);
        assert!(!sketch.contains_prefix("a"));
        assert!(sketch.contains_prefix("b"));
    }

    #[test]
    fn iter_estimates_exposes_prefixes() {
        let mut sketch = GpsSketch::default();
        sketch.add("dog", 2.0);
        sketch.add("door", 3.0);

        let mut seen = HashMap::new();
        for (prefix, estimate) in sketch.iter_estimates() {
            seen.insert(String::from_utf8(prefix.to_vec()).unwrap(), estimate);
        }

        assert_close(*seen.get("d").unwrap(), 5.0, 1e-9);
        assert!(seen.contains_key(""));
    }

    #[test]
    fn sampler_can_hit_very_deep_levels() {
        let sketch = GpsSketch::new(0.5);
        let level = sketch.debug_sample_level_from_bits(0);
        assert!(level >= 120);
    }

    #[test]
    #[should_panic(expected = "hash seed")]
    fn merge_requires_matching_seeds() {
        let mut a = GpsSketch::with_seed(0.5, 1);
        let mut b = GpsSketch::with_seed(0.5, 2);
        a.add("foo", 1.0);
        b.add("bar", 2.0);
        a.merge_from(&b);
    }
}
