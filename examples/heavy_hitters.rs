use geometric_prefix_sketch::GpsSketch;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    // Build a sketch that tracks up to 8 heavy completions per realized prefix.
    let mut sketch = GpsSketch::with_heavy_hitters(0.55, 123, 8);

    // Feed random “query” strings into the sketch. Positive deltas contribute
    // to the bounded top-k summaries stored at each prefix.
    for _ in 0..10_000 {
        let key = random_word(&mut rng);
        sketch.add(key, 1.0);
    }

    let prefix = "ap";
    let tops = sketch.top_completions(prefix, 5);
    println!("Top completions under '{}':", prefix);
    for (key, weight) in tops {
        println!("  {:<10} -> {:.1}", String::from_utf8_lossy(&key), weight);
    }
}

fn random_word(rng: &mut StdRng) -> String {
    let len = rng.gen_range(3..8);
    (0..len)
        .map(|_| (b'a' + rng.gen_range(0..26)) as char)
        .collect()
}
