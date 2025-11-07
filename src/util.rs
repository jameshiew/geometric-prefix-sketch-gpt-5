use xxhash_rust::xxh3::xxh3_128_with_seed;

pub(crate) fn deterministic_hash(bytes: &[u8], seed: u64) -> u128 {
    xxh3_128_with_seed(bytes, seed)
}

pub(crate) fn inclusion_prob(alpha: f64, depth: usize) -> f64 {
    if depth <= 1 {
        return 1.0;
    }
    let prob = fast_pow(alpha, depth - 1);
    if prob < f64::MIN_POSITIVE { 0.0 } else { prob }
}

pub(crate) fn probability_to_hash_threshold(probability: f64) -> u128 {
    if !probability.is_finite() || probability <= 0.0 {
        return 0;
    }
    if probability >= 1.0 {
        return u128::MAX;
    }
    let bits = probability.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1023; // unbiased exponent
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52); // implicit leading 1
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
