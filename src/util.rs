use xxhash_rust::xxh3::xxh3_128_with_seed;

const HALF_MAX_DEPTH: usize = (u128::BITS as usize) + 1;
const MAX_TABLE_DEPTH: usize = 4_000_000;

pub(crate) fn deterministic_hash(bytes: &[u8], seed: u64) -> u128 {
    xxh3_128_with_seed(bytes, seed)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn inclusion_prob(alpha: f64, depth: usize) -> f64 {
    if depth <= 1 {
        return 1.0;
    }
    let prob = fast_pow(alpha, depth - 1);
    if prob < f64::MIN_POSITIVE { 0.0 } else { prob }
}

#[cfg_attr(not(test), allow(dead_code))]
fn fast_pow(mut base: f64, mut exp: usize) -> f64 {
    let mut acc = 1.0;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

#[derive(Clone, Debug)]
pub(crate) struct InclusionTable {
    alpha: f64,
    mode: SamplingMode,
    max_depth: usize,
    q_values: Vec<f64>,
    inv_q_values: Vec<f64>,
}

#[derive(Clone, Debug)]
enum SamplingMode {
    Half,
    Fixed { q128_by_depth: Vec<u128> },
}

impl InclusionTable {
    pub(crate) fn new(alpha: f64) -> Self {
        let is_half = (alpha - 0.5).abs() < f64::EPSILON;
        let max_depth = if is_half {
            HALF_MAX_DEPTH
        } else {
            compute_max_depth(alpha)
        };

        let mut q_values = Vec::with_capacity(max_depth + 1);
        let mut inv_q_values = Vec::with_capacity(max_depth + 1);
        q_values.push(1.0);
        inv_q_values.push(1.0);
        let mut prob = 1.0;
        for _ in 1..=max_depth {
            q_values.push(prob);
            inv_q_values.push(if prob == 0.0 { 0.0 } else { 1.0 / prob });
            prob *= alpha;
            if prob < f64::MIN_POSITIVE {
                prob = 0.0;
            }
        }

        let mode = if is_half {
            SamplingMode::Half
        } else {
            let mut q128_by_depth = Vec::with_capacity(max_depth + 1);
            q128_by_depth.push(u128::MAX);
            let alpha_q128 = f64_to_q128(alpha);
            let mut current = u128::MAX;
            for depth in 1..=max_depth {
                if depth == 1 {
                    q128_by_depth.push(current);
                    continue;
                }
                current = mul_q128(current, alpha_q128);
                if current > q128_by_depth[depth - 1] {
                    current = q128_by_depth[depth - 1];
                }
                q128_by_depth.push(current);
            }
            SamplingMode::Fixed { q128_by_depth }
        };

        Self {
            alpha,
            mode,
            max_depth,
            q_values,
            inv_q_values,
        }
    }

    pub(crate) fn alpha(&self) -> f64 {
        self.alpha
    }

    pub(crate) fn max_realizable_depth(&self) -> usize {
        self.max_depth
    }

    pub(crate) fn is_half_sampler(&self) -> bool {
        matches!(self.mode, SamplingMode::Half)
    }

    pub(crate) fn prob(&self, depth: usize) -> f64 {
        if depth <= 1 {
            1.0
        } else if depth < self.q_values.len() {
            self.q_values[depth]
        } else {
            0.0
        }
    }

    pub(crate) fn inv_prob(&self, depth: usize) -> f64 {
        if depth <= 1 {
            1.0
        } else if depth < self.inv_q_values.len() {
            self.inv_q_values[depth]
        } else {
            0.0
        }
    }

    pub(crate) fn q128(&self, depth: usize) -> u128 {
        match &self.mode {
            SamplingMode::Half => half_threshold(depth),
            SamplingMode::Fixed { q128_by_depth } => q128_by_depth.get(depth).copied().unwrap_or(0),
        }
    }
}

fn compute_max_depth(alpha: f64) -> usize {
    let log_alpha = alpha.ln();
    if !log_alpha.is_finite() || log_alpha >= 0.0 {
        return 1;
    }
    let raw = (f64::MIN_POSITIVE.ln() / log_alpha).floor();
    let clamped = raw.max(1.0).min((MAX_TABLE_DEPTH - 1) as f64);
    clamped as usize + 1
}

fn half_threshold(depth: usize) -> u128 {
    if depth <= 1 {
        return u128::MAX;
    }
    let shift = depth - 1;
    if shift >= u128::BITS as usize {
        0
    } else {
        u128::MAX >> shift
    }
}

fn mul_q128(a: u128, b: u128) -> u128 {
    let a_lo = a as u64 as u128;
    let a_hi = (a >> 64) as u64 as u128;
    let b_lo = b as u64 as u128;
    let b_hi = (b >> 64) as u64 as u128;

    let lo = a_lo * b_lo;
    let mid1 = a_lo * b_hi;
    let mid2 = a_hi * b_lo;
    let hi = a_hi * b_hi;

    let mid_sum = mid1 + mid2;
    let mid_low = mid_sum << 64;
    let (_lo_total, carry_low) = lo.overflowing_add(mid_low);
    let mut hi_total = hi + (mid_sum >> 64);
    if carry_low {
        hi_total = hi_total.wrapping_add(1);
    }
    hi_total
}

fn f64_to_q128(probability: f64) -> u128 {
    if !probability.is_finite() || probability <= 0.0 {
        return 0;
    }
    if probability >= 1.0 {
        return u128::MAX;
    }
    let bits = probability.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let shift = exp + 128 - 52;
    if shift <= -128 {
        0
    } else if shift < 0 {
        (mant as u128) >> (-shift as u32)
    } else if shift < 128 {
        (mant as u128) << (shift as u32)
    } else {
        u128::MAX
    }
}
