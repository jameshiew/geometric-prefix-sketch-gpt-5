use crate::util::inclusion_prob;
use std::collections::HashMap;

pub(crate) const PROMOTION_DEPTH: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) sum: f64,
    pub(crate) heavy: Option<MisraGries>,
    pub(crate) children: Vec<Edge>,
}

impl Node {
    pub(crate) fn new(heavy_capacity: Option<usize>) -> Self {
        Self {
            sum: 0.0,
            heavy: heavy_capacity.map(MisraGries::new),
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
            return self.children[idx].child.as_mut();
        }
        self.children.push(Edge::new(vec![byte], heavy_capacity));
        let len = self.children.len();
        self.children[len - 1].child.as_mut()
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
pub(crate) struct MisraGries {
    capacity: usize,
    entries: Vec<HeavyEntry>,
    index: HashMap<Vec<u8>, usize>,
}

#[derive(Clone, Debug)]
struct HeavyEntry {
    key: Vec<u8>,
    weight: f64,
}

impl MisraGries {
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
        self.entries.push(HeavyEntry {
            key: key.to_vec(),
            weight: delta,
        });
        self.index.insert(key.to_vec(), self.entries.len() - 1);
        self.trim_to_capacity();
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

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn iter_entries(&self) -> impl Iterator<Item = (&[u8], f64)> {
        self.entries
            .iter()
            .map(|entry| (entry.key.as_slice(), entry.weight))
    }

    fn trim_to_capacity(&mut self) {
        if self.entries.len() > self.capacity {
            self.compress_to_capacity();
        }
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        self.index.reserve(self.entries.len());
        for (idx, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.key.clone(), idx);
        }
    }

    pub(crate) fn merge_from(&mut self, other: &MisraGries) {
        for (key, weight) in other.iter_entries() {
            if let Some(&idx) = self.index.get(key) {
                self.entries[idx].weight += weight;
            } else {
                self.entries.push(HeavyEntry {
                    key: key.to_vec(),
                    weight,
                });
            }
        }
        self.compress_to_capacity();
    }

    fn compress_to_capacity(&mut self) {
        if self.entries.is_empty() {
            self.index.clear();
            return;
        }
        if self.entries.len() <= self.capacity {
            self.rebuild_index();
            return;
        }
        self.entries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.entries.truncate(self.capacity);
        self.rebuild_index();
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
        let edge_idx = ensure_edge(current, remaining, heavy_capacity);
        let edge_len = current.children[edge_idx].label.len();
        let consume = (limit - offset).min(edge_len);
        {
            let edge = &mut current.children[edge_idx];
            edge.add_weight(consume, delta);
        }
        offset += consume;
        if consume == edge_len {
            let edge = &mut current.children[edge_idx];
            let child = edge.child.as_mut();
            child.sum += delta;
            child.update_heavy_hitters(&key[offset..], delta);
            current = child;
        } else {
            break;
        }
    }
}

pub(crate) fn locate_raw_sum(node: &Node, prefix: &[u8], depth: usize) -> Option<(f64, usize)> {
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

pub(crate) enum PrefixMatch<'a> {
    Node {
        node: &'a Node,
        depth: usize,
    },
    MidEdge {
        child: &'a Node,
        remaining_label: &'a [u8],
        depth: usize,
    },
}

pub(crate) fn locate_prefix<'a>(
    node: &'a Node,
    prefix: &[u8],
    depth: usize,
) -> Option<PrefixMatch<'a>> {
    if prefix.is_empty() {
        return Some(PrefixMatch::Node { node, depth });
    }
    for edge in &node.children {
        if edge.label[0] != prefix[0] {
            continue;
        }
        let lcp = common_prefix_len(&edge.label, prefix);
        if lcp == prefix.len() {
            if lcp == edge.label.len() {
                return Some(PrefixMatch::Node {
                    node: &edge.child,
                    depth: depth + lcp,
                });
            }
            return Some(PrefixMatch::MidEdge {
                child: &edge.child,
                remaining_label: &edge.label[lcp..],
                depth: depth + lcp,
            });
        }
        if lcp == edge.label.len() {
            return locate_prefix(&edge.child, &prefix[lcp..], depth + lcp);
        }
        return None;
    }
    None
}

pub(crate) fn prune_node(alpha: f64, node: &mut Node, depth: usize, threshold: f64) -> usize {
    let mut removed = 0;
    let mut i = 0;
    while i < node.children.len() {
        let (child_est, max_mid_est, child_depth) = {
            let edge = &node.children[i];
            let child_depth = depth + edge.label.len();
            let q_child = inclusion_prob(alpha, child_depth);
            let child_est = if q_child == 0.0 {
                0.0
            } else {
                edge.child.sum / q_child
            };
            let mut max_mid_est: f64 = 0.0;
            for (offset, raw) in edge.mid_sums.iter().enumerate() {
                let mid_depth = depth + offset + 1;
                let q_mid = inclusion_prob(alpha, mid_depth);
                if q_mid == 0.0 {
                    continue;
                }
                let estimate = raw / q_mid;
                max_mid_est = max_mid_est.max(estimate.abs());
            }
            (child_est, max_mid_est, child_depth)
        };

        if child_est.abs().max(max_mid_est) < threshold {
            let edge = node.children.remove(i);
            removed += edge.child.count_prefixes();
            removed += edge.mid_sums.len();
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
        (Some(dst_hh), Some(src_hh)) => dst_hh.merge_from(src_hh),
        (None, Some(src_hh)) => {
            if let Some(cap) = heavy_capacity {
                let mut hh = MisraGries::new(cap);
                hh.merge_from(src_hh);
                dst.heavy = Some(hh);
            }
        }
        _ => {}
    }

    for child in &src.children {
        merge_edge_into(dst, child, heavy_capacity);
    }
}

fn merge_edge_into(dst: &mut Node, src_edge: &Edge, heavy_capacity: Option<usize>) {
    merge_edge_segment(
        dst,
        &src_edge.label,
        &src_edge.mid_sums,
        src_edge.child.as_ref(),
        heavy_capacity,
    );
}

fn merge_edge_segment(
    dst: &mut Node,
    label: &[u8],
    mid_sums: &[f64],
    child: &Node,
    heavy_capacity: Option<usize>,
) {
    if label.is_empty() {
        merge_nodes(dst, child, heavy_capacity);
        return;
    }
    debug_assert_eq!(mid_sums.len(), label.len().saturating_sub(1));

    if let Some(idx) = dst
        .children
        .iter()
        .position(|edge| edge.label[0] == label[0])
    {
        loop {
            let edge = &mut dst.children[idx];
            let lcp = common_prefix_len(&edge.label, label);
            if lcp == 0 {
                break;
            }
            if lcp == edge.label.len() && lcp == label.len() {
                for (dst_mid, src_mid) in edge.mid_sums.iter_mut().zip(mid_sums.iter()) {
                    *dst_mid += src_mid;
                }
                merge_nodes(edge.child.as_mut(), child, heavy_capacity);
                return;
            }
            if lcp == edge.label.len() {
                for (dst_mid, src_mid) in edge.mid_sums.iter_mut().zip(mid_sums.iter()) {
                    *dst_mid += src_mid;
                }
                debug_assert!(lcp > 0 && lcp <= mid_sums.len());
                let boundary = mid_sums[lcp - 1];
                edge.child.sum += boundary;
                merge_edge_segment(
                    edge.child.as_mut(),
                    &label[lcp..],
                    &mid_sums[lcp..],
                    child,
                    heavy_capacity,
                );
                return;
            }
            edge.split(lcp, heavy_capacity);
            // After splitting, the same edge now carries the common prefix,
            // so re-evaluate against the source segment.
        }
    }

    dst.children.push(Edge {
        label: label.to_vec(),
        mid_sums: mid_sums.to_vec(),
        child: Box::new(child.clone()),
    });
}

pub(crate) fn ensure_edge(
    node: &mut Node,
    remaining: &[u8],
    heavy_capacity: Option<usize>,
) -> usize {
    loop {
        let match_idx = {
            let mut idx = None;
            for i in 0..node.children.len() {
                if node.children[i].label[0] == remaining[0] {
                    idx = Some(i);
                    break;
                }
            }
            idx
        };
        if let Some(idx) = match_idx {
            let split_again = {
                let edge = &mut node.children[idx];
                let lcp = common_prefix_len(&edge.label, remaining);
                debug_assert!(lcp > 0);
                if lcp < edge.label.len() && lcp < remaining.len() {
                    edge.split(lcp, heavy_capacity);
                    true
                } else {
                    return idx;
                }
            };
            if split_again {
                continue;
            }
        } else {
            node.children
                .push(Edge::new(remaining.to_vec(), heavy_capacity));
            return node.children.len() - 1;
        }
    }
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_retains_mid_edge_signal() {
        let mut root = Node::new(None);
        let mut edge = Edge {
            label: vec![b'a', b'b', b'c'],
            mid_sums: vec![10.0, 0.1],
            child: Box::new(Node::new(None)),
        };
        edge.child.sum = 0.01;
        root.children.push(edge);

        let removed = prune_node(0.5, &mut root, 0, 5.0);
        assert_eq!(removed, 0);
        assert_eq!(root.children.len(), 1);
    }

    #[test]
    fn prune_counts_include_mid_edges() {
        let mut root = Node::new(None);

        let mut keeper = Edge {
            label: vec![b'k', b'e', b'e', b'p'],
            mid_sums: vec![2.0, 0.0, 0.0],
            child: Box::new(Node::new(None)),
        };
        keeper.child.sum = 0.05;
        root.children.push(keeper);

        let mut dropped = Edge {
            label: vec![b'x', b'y'],
            mid_sums: vec![0.01],
            child: Box::new(Node::new(None)),
        };
        dropped.child.sum = 0.01;
        root.children.push(dropped);

        let removed = prune_node(0.5, &mut root, 0, 1.0);
        assert_eq!(removed, 2); // 1 mid-edge accumulator + 1 leaf node
        assert_eq!(root.children.len(), 1);
    }

    #[test]
    fn misra_gries_merge_caps_capacity() {
        let mut left = MisraGries::new(32);
        let mut right = MisraGries::new(32);
        for i in 0..64 {
            let key = vec![b'a', i as u8];
            left.update(&key, 1.0 + i as f64);
        }
        for i in 64..128 {
            let key = vec![b'b', i as u8];
            right.update(&key, 1.0 + i as f64);
        }

        left.merge_from(&right);
        assert!(left.entries.len() <= 32);
        assert_eq!(left.index.len(), left.entries.len());
    }
}
