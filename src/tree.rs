use crate::util::inclusion_prob;
use std::collections::HashMap;

pub(crate) const PROMOTION_DEPTH: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) sum: f64,
    pub(crate) heavy: Option<SpaceSaving>,
    pub(crate) children: Vec<Edge>,
}

impl Node {
    pub(crate) fn new(heavy_capacity: Option<usize>) -> Self {
        Self {
            sum: 0.0,
            heavy: heavy_capacity.map(SpaceSaving::new),
            children: Vec::new(),
        }
    }

    pub(crate) fn ensure_unit_child(
        &mut self,
        byte: u8,
        heavy_capacity: Option<usize>,
    ) -> &mut Node {
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

    pub(crate) fn update_heavy_hitters(&mut self, suffix: &[u8], delta: f64) {
        if delta <= 0.0 {
            return;
        }
        if let Some(hh) = &mut self.heavy {
            hh.update(suffix, delta);
        }
    }

    pub(crate) fn count_prefixes(&self) -> usize {
        let mut total = 1;
        for edge in &self.children {
            total += edge.mid_sums.len();
            total += edge.child.count_prefixes();
        }
        total
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub(crate) label: Vec<u8>,
    pub(crate) mid_sums: Vec<f64>,
    pub(crate) child: Box<Node>,
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

    pub(crate) fn add_weight(&mut self, bytes_to_consume: usize, delta: f64) {
        let limit = bytes_to_consume.min(self.label.len());
        for i in 0..limit.min(self.label.len().saturating_sub(1)) {
            self.mid_sums[i] += delta;
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SpaceSaving {
    capacity: usize,
    entries: Vec<HeavyEntry>,
    index: HashMap<Vec<u8>, usize>,
}

#[derive(Clone, Debug)]
struct HeavyEntry {
    key: Vec<u8>,
    weight: f64,
}

impl SpaceSaving {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
        }
    }

    pub(crate) fn update(&mut self, key: &[u8], delta: f64) {
        if delta <= 0.0 {
            return;
        }
        if let Some(&idx) = self.index.get(key) {
            self.entries[idx].weight += delta;
            return;
        }
        if self.entries.len() < self.capacity {
            let entry = HeavyEntry {
                key: key.to_vec(),
                weight: delta,
            };
            self.entries.push(entry);
            self.index.insert(key.to_vec(), self.entries.len() - 1);
            return;
        }
        let (min_idx, min_weight) = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap())
            .map(|(idx, entry)| (idx, entry.weight))
            .expect("entries non-empty");
        let replacement = HeavyEntry {
            key: key.to_vec(),
            weight: min_weight + delta,
        };
        let old = std::mem::replace(&mut self.entries[min_idx], replacement);
        self.index.remove(&old.key);
        self.index
            .insert(self.entries[min_idx].key.clone(), min_idx);
    }

    pub(crate) fn top_k(&self, k: usize) -> Vec<(Vec<u8>, f64)> {
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

    pub(crate) fn iter_entries(&self) -> impl Iterator<Item = (&[u8], f64)> {
        self.entries
            .iter()
            .map(|entry| (entry.key.as_slice(), entry.weight))
    }
}

pub(crate) fn add_inner(
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

pub(crate) fn locate_raw_sum(
    node: &Node,
    prefix: &[u8],
    depth: usize,
) -> Option<(f64, usize)> {
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

pub(crate) fn locate_node<'a>(
    node: &'a Node,
    prefix: &[u8],
    depth: usize,
) -> Option<(&'a Node, usize)> {
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

pub(crate) fn prune_node(alpha: f64, node: &mut Node, depth: usize, threshold: f64) -> usize {
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

#[cfg(test)]
pub(crate) fn visit_raw<F>(node: &Node, prefix: &mut Vec<u8>, f: &mut F)
where
    F: FnMut(&[u8], f64),
{
    f(prefix, node.sum);
    for edge in &node.children {
        for (i, &byte) in edge.label.iter().enumerate() {
            prefix.push(byte);
            if i + 1 == edge.label.len() {
                visit_raw(&edge.child, prefix, f);
            } else {
                let raw = edge.mid_sums[i];
                f(prefix, raw);
            }
            prefix.pop();
        }
    }
}

pub(crate) fn merge_nodes(dst: &mut Node, src: &Node, heavy_capacity: Option<usize>) {
    dst.sum += src.sum;
    match (&mut dst.heavy, &src.heavy) {
        (Some(dst_hh), Some(src_hh)) => {
            for (suffix, weight) in src_hh.iter_entries() {
                dst_hh.update(suffix, weight);
            }
        }
        (None, Some(src_hh)) => {
            if let Some(cap) = heavy_capacity {
                let mut hh = SpaceSaving::new(cap);
                for (suffix, weight) in src_hh.iter_entries() {
                    hh.update(suffix, weight);
                }
                dst.heavy = Some(hh);
            }
        }
        _ => {}
    }

    'outer: for child in &src.children {
        for existing in &mut dst.children {
            if existing.label == child.label {
                for (a, b) in existing.mid_sums.iter_mut().zip(&child.mid_sums) {
                    *a += b;
                }
                merge_nodes(
                    existing.child.as_mut(),
                    child.child.as_ref(),
                    heavy_capacity,
                );
                continue 'outer;
            }
        }
        dst.children.push(child.clone());
    }
}

pub(crate) fn ensure_edge<'a>(
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
