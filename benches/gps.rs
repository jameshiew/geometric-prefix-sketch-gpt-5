use criterion::{Criterion, black_box, criterion_group, criterion_main};
use geometric_prefix_sketch::GpsSketch;
use rand::{Rng, SeedableRng, rngs::StdRng};

const DEFAULT_ALPHA: f64 = 0.5;
const HASH_SEED: u64 = 42;

fn random_ascii_keys(count: usize, len: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    (0..count)
        .map(|_| {
            (0..len)
                .map(|_| rng.gen_range(b'a'..=b'z'))
                .collect::<Vec<u8>>()
        })
        .collect()
}

fn build_sketch(keys: &[Vec<u8>]) -> GpsSketch {
    let mut sketch = GpsSketch::with_seed(DEFAULT_ALPHA, HASH_SEED);
    for key in keys {
        sketch.add(key, 1.0);
    }
    sketch
}

fn bench_add(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(1234);
    let keys = random_ascii_keys(10_000, 16, &mut rng);
    c.bench_function("gps_add_10k_len16", |b| {
        b.iter(|| {
            let mut sketch = GpsSketch::with_seed(DEFAULT_ALPHA, HASH_SEED);
            for key in &keys {
                sketch.add(key, 1.0);
            }
            black_box(sketch.node_count());
        })
    });
}

fn bench_estimate(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(2024);
    let keys = random_ascii_keys(20_000, 20, &mut rng);
    let sketch = build_sketch(&keys);
    let queries = keys
        .iter()
        .take(2_000)
        .map(|k| k[..10].to_vec())
        .collect::<Vec<_>>();

    c.bench_function("gps_estimate_2k_prefixes", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for q in &queries {
                total += sketch.estimate(q);
            }
            black_box(total);
        })
    });
}

fn bench_merge(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let keys = random_ascii_keys(30_000, 24, &mut rng);
    let mut left = GpsSketch::with_seed(DEFAULT_ALPHA, HASH_SEED);
    let mut right = GpsSketch::with_seed(DEFAULT_ALPHA, HASH_SEED);
    for (idx, key) in keys.iter().enumerate() {
        if idx % 2 == 0 {
            left.add(key, 1.0);
        } else {
            right.add(key, 1.0);
        }
    }

    c.bench_function("gps_merge_15k+15k", |b| {
        b.iter(|| {
            let mut merged = left.clone();
            merged.merge_from(&right);
            black_box(merged.node_count());
        })
    });
}

fn bench_heavy_hitters(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(99);
    let keys = random_ascii_keys(15_000, 12, &mut rng);
    let mut sketch = GpsSketch::with_heavy_hitters(DEFAULT_ALPHA, HASH_SEED, 8);
    for key in &keys {
        sketch.add(key, 1.0);
    }
    let prefix = keys[0][..3].to_vec();

    c.bench_function("gps_top_completions", |b| {
        b.iter(|| {
            let tops = sketch.top_completions(&prefix, 5);
            black_box(tops);
        })
    });
}

fn bench_prune(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(555);
    let keys = random_ascii_keys(25_000, 18, &mut rng);
    let sketch = build_sketch(&keys);

    c.bench_function("gps_prune_threshold_0.5", |b| {
        b.iter(|| {
            let mut clone = sketch.clone();
            let removed = clone.prune_by_estimate(0.5);
            black_box(removed);
        })
    });
}

criterion_group!(
    benches,
    bench_add,
    bench_estimate,
    bench_merge,
    bench_heavy_hitters,
    bench_prune
);
criterion_main!(benches);
