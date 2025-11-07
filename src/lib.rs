//! Geometric Prefix Sketch (GPS) implementation.
//!
//! This crate implements the sketch described in `DESIGN.md`. The sketch keeps
//! unbiased prefix aggregates while touching only a geometrically sampled
//! portion of each key's prefixes, so updates take constant expected time. It
//! now also offers optional per-node heavy-hitter summaries (SpaceSaving) so you
//! can extract likely completions under any prefix.

use xxhash_rust::xxh3::xxh3_128_with_seed;

const HASH128_TO_UNIT: f64 = 1.0 / ((u128::MAX as f64) + 1.0);

#[derive(Clone, Debug)]
struct Node {
    sum: f64,
    children: Vec<Child>,
    heavy: Option<SpaceSaving>,
}

#[derive(Clone, Debug)]
struct Child {
    byte: u8,
    node: Box<Node>,
}

impl Node {
    fn new(hh_capacity: Option<usize>) -> Self {
        Self {
            sum: 0.0,
            children: Vec::new(),
            heavy: hh_capacity.map(SpaceSaving::new),
        }
    }

    fn ensure_child(&mut self, byte: u8, hh_capacity: Option<usize>) -> &mut Node {
        if let Some(pos) = self.children.iter().position(|child| child.byte == byte) {
            return &mut self.children[pos].node;
        }
        self.children.push(Child {
            byte,
            node: Box::new(Node::new(hh_capacity)),
        });
        let idx = self.children.len() - 1;
        &mut self.children[idx].node
    }

    fn get_child(&self, byte: u8) -> Option<&Node> {
        self.children
            .iter()
            .find(|child| child.byte == byte)
            .map(|child| child.node.as_ref())
    }

    fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|child| child.node.node_count())
            .sum::<usize>()
    }

    fn update_heavy_hitters(&mut self, suffix: &[u8], delta: f64) {
        if delta <= 0.0 {
            return;
        }
        if let Some(sketch) = &mut self.heavy {
            sketch.update(suffix, delta);
        }
    }
}

#[derive(Clone, Debug)]
struct SpaceSaving {
    capacity: usize,
    entries: Vec<HeavyEntry>,
}

#[derive(Clone, Debug)]
struct HeavyEntry {
    key: Vec<u8>,
    weight: f64,
}

impl SpaceSaving {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SpaceSaving capacity must be positive");
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
        }
    }

    fn update(&mut self, key: &[u8], delta: f64) {
        if delta <= 0.0 {
            return;
        }
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
            entry.weight += delta;
            return;
        }
        if self.entries.len() < self.capacity {
            self.entries.push(HeavyEntry {
                key: key.to_vec(),
                weight: delta,
            });
            return;
        }
        let (min_idx, min_weight) = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap())
            .map(|(idx, entry)| (idx, entry.weight))
            .expect("entries non-empty");
        self.entries[min_idx] = HeavyEntry {
            key: key.to_vec(),
            weight: min_weight + delta,
        };
    }

    fn top_k(&self, k: usize) -> Vec<(Vec<u8>, f64)> {
        if k == 0 {
            return Vec::new();
        }
        let mut items: Vec<_> = self
            .entries
            .iter()
            .map(|entry| (entry.key.clone(), entry.weight))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items.truncate(k.min(items.len()));
        items
    }

    fn iter_entries(&self) -> impl Iterator<Item = (&[u8], f64)> {
        self.entries
            .iter()
            .map(|entry| (entry.key.as_slice(), entry.weight))
    }
}

/// Geometric Prefix Sketch with deterministic per-key sampling.
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
    /// `alpha` must lie strictly in `(0, 1)`.
    pub fn new(alpha: f64) -> Self {
        Self::with_seed(alpha, 0)
    }

    /// Creates a new sketch with an explicit deterministic hash seed.
    /// All sketches that need to merge must share the same seed.
    pub fn with_seed(alpha: f64, hash_seed: u64) -> Self {
        Self::with_heavy_hitters_internal(alpha, hash_seed, None)
    }

    /// Creates a sketch with local heavy-hitter tracking of capacity `hh_capacity`
    /// entries per node (SpaceSaving). Only positive `delta` updates contribute.
    pub fn with_heavy_hitters(alpha: f64, hash_seed: u64, hh_capacity: usize) -> Self {
        assert!(hh_capacity > 0, "heavy hitter capacity must be > 0");
        Self::with_heavy_hitters_internal(alpha, hash_seed, Some(hh_capacity))
    }

    fn with_heavy_hitters_internal(
        alpha: f64,
        hash_seed: u64,
        heavy_capacity: Option<usize>,
    ) -> Self {
        assert!(alpha.is_finite());
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        Self {
            alpha,
            log_alpha: alpha.ln(),
            hash_seed,
            heavy_capacity,
            root: Node::new(heavy_capacity),
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

    /// Returns the heavy-hitter capacity (if any).
    pub fn heavy_hitters_capacity(&self) -> Option<usize> {
        self.heavy_capacity
    }

    /// Adds `delta` to `key`.
    pub fn add<K: AsRef<[u8]>>(&mut self, key: K, delta: f64) {
        if delta == 0.0 {
            return;
        }

        let key = key.as_ref();
        let limit = self.prefix_budget(key);
        self.root.sum += delta;
        self.root.update_heavy_hitters(key, delta);

        let mut node = &mut self.root;
        for depth in 0..limit {
            let byte = key[depth];
            node = node.ensure_child(byte, self.heavy_capacity);
            node.sum += delta;
            node.update_heavy_hitters(&key[depth + 1..], delta);
        }
    }

    /// Returns the Horvitz–Thompson estimate of the sum under `prefix`.
    pub fn estimate<K: AsRef<[u8]>>(&self, prefix: K) -> f64 {
        let prefix = prefix.as_ref();
        let depth = prefix.len();
        match self.find_node(prefix) {
            Some(node) => {
                let q = inclusion_prob(self.alpha, depth);
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
        self.root = Node::new(self.heavy_capacity);
    }

    /// Number of materialized prefixes (including root).
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Returns whether `prefix` currently has a node in the trie.
    pub fn contains_prefix<K: AsRef<[u8]>>(&self, prefix: K) -> bool {
        self.find_node(prefix.as_ref()).is_some()
    }

    /// Returns an iterator over all materialized prefixes along with their estimates.
    pub fn iter_estimates(&self) -> impl Iterator<Item = (Vec<u8>, f64)> {
        let mut all = Vec::new();
        let mut prefix = Vec::new();
        self.collect_estimates(&self.root, 0, &mut prefix, &mut all);
        all.into_iter()
    }

    /// Returns up to `k` heavy-hitter completions under `prefix` (sorted desc by estimate).
    pub fn top_completions<K: AsRef<[u8]>>(&self, prefix: K, k: usize) -> Vec<(Vec<u8>, f64)> {
        let prefix = prefix.as_ref();
        if k == 0 {
            return Vec::new();
        }
        let node = match self.find_node(prefix) {
            Some(node) => node,
            None => return Vec::new(),
        };
        let sketch = match &node.heavy {
            Some(sketch) => sketch,
            None => return Vec::new(),
        };
        let q = inclusion_prob(self.alpha, prefix.len());
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

    /// Removes all non-root prefixes whose estimated absolute value is below `min_abs_estimate`.
    /// Returns the number of nodes removed (including descendants).
    pub fn prune_by_estimate(&mut self, min_abs_estimate: f64) -> usize {
        if min_abs_estimate <= 0.0 {
            return 0;
        }
        prune_node(self.alpha, &mut self.root, 0, min_abs_estimate)
    }

    /// Merges `other` into `self` by pointwise addition.
    pub fn merge_from(&mut self, other: &GpsSketch) {
        assert!(
            (self.alpha - other.alpha).abs() < 1e-12
                && self.hash_seed == other.hash_seed
                && self.heavy_capacity == other.heavy_capacity,
            "cannot merge sketches with mismatched alpha, hash seed, or HH capacity"
        );
        merge_nodes(self.heavy_capacity, &mut self.root, &other.root);
    }

    fn find_node(&self, prefix: &[u8]) -> Option<&Node> {
        let mut node = &self.root;
        for &byte in prefix {
            node = node.get_child(byte)?;
        }
        Some(node)
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
        for child in &node.children {
            prefix.push(child.byte);
            self.collect_estimates(&child.node, depth + 1, prefix, out);
            prefix.pop();
        }
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
    fn raw_node_sum(&self, prefix: &[u8]) -> Option<f64> {
        self.find_node(prefix).map(|node| node.sum)
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

fn inclusion_prob(alpha: f64, depth: usize) -> f64 {
    if depth <= 1 {
        return 1.0;
    }
    let value = alpha.powf((depth - 1) as f64);
    if value < f64::MIN_POSITIVE {
        f64::MIN_POSITIVE
    } else {
        value
    }
}

fn prune_node(alpha: f64, node: &mut Node, depth: usize, threshold: f64) -> usize {
    let mut removed = 0;
    let mut idx = 0;
    while idx < node.children.len() {
        let child_depth = depth + 1;
        let estimate = node.children[idx].node.sum / inclusion_prob(alpha, child_depth);
        if estimate.abs() < threshold {
            removed += node.children[idx].node.node_count();
            node.children.remove(idx);
        } else {
            removed += prune_node(alpha, &mut node.children[idx].node, child_depth, threshold);
            idx += 1;
        }
    }
    removed
}

fn merge_nodes(capacity: Option<usize>, dst: &mut Node, src: &Node) {
    dst.sum += src.sum;
    match (&mut dst.heavy, &src.heavy) {
        (Some(dst_hh), Some(src_hh)) => {
            for (suffix, weight) in src_hh.iter_entries() {
                dst_hh.update(suffix, weight);
            }
        }
        (None, Some(src_hh)) => {
            if let Some(cap) = capacity {
                let mut hh = SpaceSaving::new(cap);
                for (suffix, weight) in src_hh.iter_entries() {
                    hh.update(suffix, weight);
                }
                dst.heavy = Some(hh);
            }
        }
        _ => {}
    }

    for child in &src.children {
        if let Some(existing) = dst
            .children
            .iter_mut()
            .find(|candidate| candidate.byte == child.byte)
        {
            merge_nodes(capacity, &mut existing.node, &child.node);
        } else {
            dst.children.push(child.clone());
        }
    }
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
        let q = inclusion_prob(sketch.alpha(), depth);
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
            let key = String::from_utf8(prefix).unwrap();
            seen.insert(key, estimate);
        }

        assert_close(*seen.get("d").unwrap(), 5.0, 1e-9);
        assert!(seen.contains_key(""));
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
        let key = find_key_with_budget(&sketch, 3);
        for _ in 0..50 {
            sketch.add(&key, 1.0);
        }

        let prefix_len = 2.min(key.len());
        let prefix = &key[..prefix_len];
        let tops = sketch.top_completions(prefix, 1);
        assert_eq!(&tops[0].0, &key);
        let q = inclusion_prob(sketch.alpha(), prefix_len);
        assert_close(tops[0].1, 50.0 / q, 1e-9);
    }

    fn find_key_with_budget(sketch: &GpsSketch, min_budget: usize) -> Vec<u8> {
        for idx in 0..10_000 {
            let candidate = format!("key-{idx}");
            if sketch
                .debug_prefix_budget(candidate.as_bytes())
                .ge(&min_budget)
            {
                return candidate.into_bytes();
            }
        }
        panic!("failed to find key with sufficient budget");
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

    #[test]
    #[should_panic(expected = "HH capacity")]
    fn merge_requires_matching_hh_capacity() {
        let mut a = GpsSketch::with_heavy_hitters(0.5, 0, 2);
        let b = GpsSketch::with_seed(0.5, 0);
        a.merge_from(&b);
    }
}
