use xxhash_rust::xxh3::xxh3_128_with_seed;

pub(crate) const HASH128_TO_UNIT: f64 = 1.0 / ((u128::MAX as f64) + 1.0);

pub(crate) fn deterministic_hash(bytes: &[u8], seed: u64) -> u128 {
    xxh3_128_with_seed(bytes, seed)
}

pub(crate) fn inclusion_prob(alpha: f64, depth: usize) -> f64 {
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
