use geometric_prefix_sketch::GpsSketch;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

#[test]
fn estimates_track_exact_counts() {
    let mut rng = StdRng::seed_from_u64(1337);
    let mut sketch = GpsSketch::default();
    let mut exact: HashMap<Vec<u8>, usize> = HashMap::new();

    for _ in 0..20_000 {
        let mut key = Vec::with_capacity(10);
        key.push(b'/');
        let len = rng.gen_range(3..7);
        for _ in 0..len {
            key.push(rng.gen_range(b'a'..=b'z'));
        }
        sketch.add(&key, 1.0);
        for depth in 1..=3 {
            if key.len() >= depth {
                exact
                    .entry(key[..depth].to_vec())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
    }

    for (prefix, &count) in &exact {
        let estimate = sketch.estimate(prefix);
        if prefix.len() == 1 {
            assert!(
                (estimate - count as f64).abs() < 1e-9,
                "depth-1 prefix {:?} should be exact",
                String::from_utf8_lossy(prefix)
            );
            continue;
        }
        if count < 200 {
            continue;
        }
        let rel_err = (estimate - count as f64).abs() / count as f64;
        assert!(
            rel_err < 0.15,
            "prefix {:?} rel error {:.3} (estimate {:.2}, exact {})",
            String::from_utf8_lossy(prefix),
            rel_err,
            estimate,
            count
        );
    }
}
