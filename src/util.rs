use xxhash_rust::xxh3::xxh3_128_with_seed;

const HASH_DOMAIN: f64 = u128::MAX as f64;

pub(crate) fn deterministic_hash(bytes: &[u8], seed: u64) -> u128 {
    xxh3_128_with_seed(bytes, seed)
}

pub(crate) fn inclusion_prob(alpha: f64, depth: usize) -> f64 {
    if depth <= 1 {
        return 1.0;
    }
    fast_pow(alpha, depth - 1).max(f64::MIN_POSITIVE)
}

pub(crate) fn probability_to_hash_threshold(probability: f64) -> u128 {
    if probability.is_nan() || probability <= 0.0 {
        0
    } else if probability >= 1.0 {
        u128::MAX
    } else {
        let scaled = probability * HASH_DOMAIN;
        if scaled >= HASH_DOMAIN {
            u128::MAX
        } else if scaled <= 0.0 {
            0
        } else {
            scaled as u128
        }
    }
}

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
