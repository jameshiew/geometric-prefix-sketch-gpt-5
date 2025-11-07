//! Geometric Prefix Sketch (GPS) implementation with a compressed radix trie.
//!
//! The sketch follows the design laid out in `DESIGN.md`: every update touches a
//! geometrically sampled prefix depth, accumulators remain unbiased, sketches
//! merge additively, and (optional) per-node heavy-hitter summaries expose top
//! completions. The trie now uses *compressed edges* so long, single-child runs
//! past a configurable promotion depth occupy a single node/edge instead of one
//! heap allocation per byte.

use std::collections::HashMap;

use xxhash_rust::xxh3::xxh3_128_with_seed;

const HASH128_TO_UNIT: f64 = 1.0 / ((u128::MAX as f64) + 1.0);
const PROMOTION_DEPTH: usize = 4; // Always materialize nodes for the first bytes.

#[derive(Clone, Debug)]
struct Node {
    sum: f64,
    heavy: Option<SpaceSaving>,
    children: Vec<Edge>,
}

impl Node {
    fn new(heavy_capacity: Option<usize>) -> Self {
        Self {
            sum: 0.0,
            heavy: heavy_capacity.map(SpaceSaving::new),
            children: Vec::new(),
        }
    }

    fn ensure_unit_child(&mut self, byte: u8, heavy_capacity: Option<usize>) -> &mut Node {
        if let Some(idx) = self
            .children
            .iter()
            .position(|edge| edge.label.len() == 1 && edge.label[0] == byte)
        {
            let ptr: *mut Edge = &mut self.children[idx];
            return unsafe { (*ptr).child.as_mut() };
        }
        let mut edge = Edge::new(vec![byte], heavy_capacity);
        let child_ptr: *mut Node = edge.child.as_mut();
        self.children.push(edge);
        unsafe { &mut *child_ptr }
    }

    fn update_heavy_hitters(&mut self, suffix: &[u8], delta: f64) {
        if delta <= 0.0 {
            return;
        }
        if let Some(hh) = &mut self.heavy {
            hh.update(suffix, delta);
        }
    }

    fn count_prefixes(&self) -> usize {
        let mut total = 1; // this node's prefix
        for edge in &self.children {
            total += edge.mid_sums.len();
            total += edge.child.count_prefixes();
        }
        total
    }
}

#[derive(Clone, Debug)]
struct Edge {
    label: Vec<u8>,
    mid_sums: Vec<f64>,
    child: Box<Node>,
}

impl Edge {
    fn new(label: Vec<u8>, heavy_capacity: Option<usize>) -> Self {
        assert!(!label.is_empty());
        let mid_len = label.len().saturating_sub(1);
        Self {
            label,
            mid_sums: vec![0.0; mid_len],
            child: Box::new(Node::new(heavy_capacity)),
        }
    }

    fn split(&mut self, at: usize, heavy_capacity: Option<usize>) {
        assert!(at > 0 && at < self.label.len());
        let suffix_label = self.label.split_off(at);
        let mut suffix_mid = if at < self.mid_sums.len() {
            self.mid_sums.split_off(at)
        } else {
            Vec::new()
        };
        let promoted_sum = self.mid_sums.pop().unwrap_or(0.0);
        let original_child =
            std::mem::replace(&mut self.child, Box::new(Node::new(heavy_capacity)));

        let mut promoted = Node::new(heavy_capacity);
        promoted.sum = promoted_sum;
        promoted.children.push(Edge {
            label: suffix_label,
            mid_sums: suffix_mid.split_off(0),
            child: original_child,
        });

        self.child = Box::new(promoted);
    }

    fn add_weight(&mut self, bytes_to_consume: usize, delta: f64) {
        let limit = bytes_to_consume.min(self.label.len());
        for i in 0..limit.min(self.label.len().saturating_sub(1)) {
            self.mid_sums[i] += delta;
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
        assert!(capacity > 0);
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
        items.truncate(items.len().min(k));
        items
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
    pub fn new(alpha: f64) -> Self {
        Self::with_seed(alpha, 0)
    }

    /// Creates a sketch with explicit hash seed.
    pub fn with_seed(alpha: f64, hash_seed: u64) -> Self {
        Self::with_heavy_hitters_internal(alpha, hash_seed, None)
    }

    /// Creates a sketch with per-node heavy-hitter capacity.
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
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the deterministic hash seed.
    pub fn hash_seed(&self) -> u64 {
        self.hash_seed
    }

    /// Returns the heavy-hitter capacity, if enabled.
    pub fn heavy_hitters_capacity(&self) -> Option<usize> {
        self.heavy_capacity
    }

    /// Adds `delta` to `key`.
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
    pub fn estimate<K: AsRef<[u8]>>(&self, prefix: K) -> f64 {
        match self.raw_sum(prefix.as_ref()) {
            Some((raw, depth)) => raw / inclusion_prob(self.alpha, depth),
            None => 0.0,
        }
    }

    /// Returns the global total estimate.
    pub fn total(&self) -> f64 {
        self.estimate(Vec::<u8>::new())
    }

    /// Removes all data from the sketch.
    pub fn clear(&mut self) {
        self.root = Node::new(self.heavy_capacity);
    }

    /// Number of stored prefixes (materialized nodes + compressed interior ones).
    pub fn node_count(&self) -> usize {
        self.root.count_prefixes()
    }

    /// Returns whether `prefix` currently exists in the trie.
    pub fn contains_prefix<K: AsRef<[u8]>>(&self, prefix: K) -> bool {
        self.raw_sum(prefix.as_ref()).is_some()
    }

    /// Returns iterator over all materialized prefixes and their estimates.
    pub fn iter_estimates(&self) -> impl Iterator<Item = (Vec<u8>, f64)> {
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        self.collect_estimates(&self.root, 0, &mut prefix, &mut out);
        out.into_iter()
    }

    /// Returns up to `k` heavy-hitter completions under `prefix`.
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

    /// Removes entire subtrees whose root estimate is below the threshold.
    pub fn prune_by_estimate(&mut self, min_abs_estimate: f64) -> usize {
        if min_abs_estimate <= 0.0 {
            return 0;
        }
        prune_node(self.alpha, &mut self.root, 0, min_abs_estimate)
    }

    /// Merges `other` into `self` by rebuilding from combined raw sums.
    pub fn merge_from(&mut self, other: &GpsSketch) {
        assert!((self.alpha - other.alpha).abs() < 1e-12);
        assert_eq!(self.hash_seed, other.hash_seed, "hash seed mismatch");
        assert_eq!(
            self.heavy_capacity, other.heavy_capacity,
            "HH capacity mismatch"
        );

        let mut totals: HashMap<Vec<u8>, f64> = HashMap::new();
        self.collect_raw_into(&mut totals);
        other.collect_raw_into(&mut totals);

        let mut heavy: HashMap<Vec<u8>, Vec<(Vec<u8>, f64)>> = HashMap::new();
        self.collect_heavy_into(&mut heavy);
        other.collect_heavy_into(&mut heavy);

        self.clear();
        for (prefix, sum) in totals {
            self.add_raw_sum(&prefix, sum);
        }
        let hh_cap = self.heavy_capacity;
        for (prefix, entries) in heavy {
            if entries.is_empty() {
                continue;
            }
            if let Some(node) = self.ensure_node(prefix.as_slice()) {
                if hh_cap.is_none() {
                    continue;
                }
                let hh = node
                    .heavy
                    .get_or_insert_with(|| SpaceSaving::new(hh_cap.unwrap()));
                for (suffix, weight) in entries {
                    hh.update(&suffix, weight);
                }
            }
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

    fn collect_raw_into(&self, out: &mut HashMap<Vec<u8>, f64>) {
        let mut prefix = Vec::new();
        collect_raw(&self.root, 0, &mut prefix, out);
    }

    fn collect_heavy_into(&self, out: &mut HashMap<Vec<u8>, Vec<(Vec<u8>, f64)>>) {
        let mut prefix = Vec::new();
        collect_heavy(&self.root, &mut prefix, out);
    }
}

fn ensure_edge<'a>(
    node: &'a mut Node,
    remaining: &[u8],
    heavy_capacity: Option<usize>,
) -> &'a mut Edge {
    loop {
        if let Some(idx) = node
            .children
            .iter()
            .position(|edge| edge.label[0] == remaining[0])
        {
            let edge_ptr: *mut Edge = &mut node.children[idx];
            let edge = unsafe { &mut *edge_ptr };
            let lcp = common_prefix_len(&edge.label, remaining);
            if lcp == 0 {
                node.children
                    .push(Edge::new(remaining.to_vec(), heavy_capacity));
                let len = node.children.len();
                return &mut node.children[len - 1];
            }
            if lcp < edge.label.len() && lcp < remaining.len() {
                edge.split(lcp, heavy_capacity);
                continue;
            }
            return edge;
        }
        node.children
            .push(Edge::new(remaining.to_vec(), heavy_capacity));
        let idx = node.children.len() - 1;
        return &mut node.children[idx];
    }
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
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
    let mut i = 0;
    while i < node.children.len() {
        let child_depth = depth + node.children[i].label.len();
        let estimate = node.children[i].child.sum / inclusion_prob(alpha, child_depth);
        if estimate.abs() < threshold {
            removed += node.children[i].child.count_prefixes();
            removed += node.children[i].mid_sums.len();
            node.children.remove(i);
        } else {
            removed += prune_node(
                alpha,
                node.children[i].child.as_mut(),
                child_depth,
                threshold,
            );
            i += 1;
        }
    }
    removed
}

fn locate_raw_sum<'a>(node: &'a Node, prefix: &[u8], depth: usize) -> Option<(f64, usize)> {
    if prefix.is_empty() {
        return Some((node.sum, depth));
    }
    for edge in &node.children {
        if edge.label[0] != prefix[0] {
            continue;
        }
        let lcp = common_prefix_len(&edge.label, prefix);
        if lcp == prefix.len() {
            if lcp == edge.label.len() {
                return Some((edge.child.sum, depth + lcp));
            }
            return Some((edge.mid_sums[lcp - 1], depth + lcp));
        }
        if lcp == edge.label.len() {
            return locate_raw_sum(&edge.child, &prefix[lcp..], depth + lcp);
        }
        return None;
    }
    None
}

fn locate_node<'a>(node: &'a Node, prefix: &[u8], depth: usize) -> Option<(&'a Node, usize)> {
    if prefix.is_empty() {
        return Some((node, depth));
    }
    for edge in &node.children {
        if edge.label[0] != prefix[0] {
            continue;
        }
        let lcp = common_prefix_len(&edge.label, prefix);
        if lcp == prefix.len() {
            if lcp == edge.label.len() {
                return Some((&edge.child, depth + lcp));
            }
            return None;
        }
        if lcp == edge.label.len() {
            return locate_node(&edge.child, &prefix[lcp..], depth + lcp);
        }
        return None;
    }
    None
}

fn ensure_node<'a>(
    node: &'a mut Node,
    prefix: &[u8],
    heavy_capacity: Option<usize>,
) -> Option<&'a mut Node> {
    if prefix.is_empty() {
        return Some(node);
    }
    let mut current = node;
    let mut depth = 0;
    while depth < prefix.len() {
        if depth < PROMOTION_DEPTH {
            current = current.ensure_unit_child(prefix[depth], heavy_capacity);
            depth += 1;
            continue;
        }
        let remaining = &prefix[depth..];
        let edge = ensure_edge(current, remaining, heavy_capacity);
        let lcp = common_prefix_len(&edge.label, remaining);
        if lcp < edge.label.len() && lcp == remaining.len() {
            edge.split(lcp, heavy_capacity);
            return Some(edge.child.as_mut());
        }
        let consume = edge.label.len();
        depth += consume;
        current = edge.child.as_mut();
    }
    Some(current)
}

fn add_inner(
    node: &mut Node,
    key: &[u8],
    depth: usize,
    limit: usize,
    delta: f64,
    heavy_capacity: Option<usize>,
) {
    if depth >= limit {
        return;
    }
    if depth < PROMOTION_DEPTH {
        let byte = key[depth];
        let child = node.ensure_unit_child(byte, heavy_capacity);
        child.sum += delta;
        child.update_heavy_hitters(&key[depth + 1..], delta);
        add_inner(child, key, depth + 1, limit, delta, heavy_capacity);
        return;
    }

    let mut current = node;
    let mut offset = depth;
    while offset < limit {
        let remaining = &key[offset..];
        let edge = ensure_edge(current, remaining, heavy_capacity);
        let consume = (limit - offset).min(edge.label.len());
        edge.add_weight(consume, delta);
        offset += consume;
        if consume == edge.label.len() {
            let child = edge.child.as_mut();
            child.sum += delta;
            child.update_heavy_hitters(&key[offset..], delta);
            current = child;
        } else {
            break;
        }
    }
}

fn collect_raw(node: &Node, depth: usize, prefix: &mut Vec<u8>, out: &mut HashMap<Vec<u8>, f64>) {
    out.entry(prefix.clone())
        .and_modify(|v| *v += node.sum)
        .or_insert(node.sum);
    for edge in &node.children {
        for (i, &byte) in edge.label.iter().enumerate() {
            prefix.push(byte);
            if i + 1 == edge.label.len() {
                collect_raw(&edge.child, depth + i + 1, prefix, out);
            } else {
                let raw = edge.mid_sums[i];
                out.entry(prefix.clone())
                    .and_modify(|v| *v += raw)
                    .or_insert(raw);
            }
            prefix.pop();
        }
    }
}

fn collect_heavy(
    node: &Node,
    prefix: &mut Vec<u8>,
    out: &mut HashMap<Vec<u8>, Vec<(Vec<u8>, f64)>>,
) {
    if let Some(sketch) = &node.heavy {
        out.entry(prefix.clone())
            .or_default()
            .extend(sketch.top_k(sketch.capacity));
    }
    for edge in &node.children {
        for &byte in &edge.label {
            prefix.push(byte);
        }
        collect_heavy(&edge.child, prefix, out);
        for _ in &edge.label {
            prefix.pop();
        }
    }
}

impl GpsSketch {
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
    fn debug_prefix_budget(&self, key: &[u8]) -> usize {
        self.prefix_budget(key)
    }

    #[cfg(test)]
    fn debug_sample_level_from_bits(&self, bits: u128) -> usize {
        self.sample_level_for_hash(bits)
    }

    #[cfg(test)]
    fn raw_sum_for_test(&self, prefix: &[u8]) -> Option<f64> {
        self.raw_sum(prefix).map(|(raw, _)| raw)
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
    fn iter_estimates_reports_mid_edge_prefixes() {
        let mut sketch = GpsSketch::default();
        sketch.add_raw_sum(b"abcd", 1.0);
        let entries: HashMap<_, _> = sketch.iter_estimates().collect();
        assert!(entries.contains_key(b"abcd" as &[u8]));
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
    fn sampler_can_hit_very_deep_levels() {
        let sketch = GpsSketch::new(0.5);
        let level = sketch.debug_sample_level_from_bits(0);
        assert!(level >= 120);
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
}
