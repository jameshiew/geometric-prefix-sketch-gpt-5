//! Demonstrates building independent sketches on different shards, then
//! merging them without replaying the original stream.

use geometric_prefix_sketch::GpsSketch;

fn main() {
    // Shard A ingests odd IDs, shard B ingests even IDs. Both share the same
    // alpha and hash seed so merges are deterministic.
    let mut shard_a = GpsSketch::with_seed(0.5, 99);
    let mut shard_b = GpsSketch::with_seed(0.5, 99);

    for id in 0..10_000 {
        let key = format!("/tenant/{}/event", id % 128);
        if id % 2 == 0 {
            shard_a.add(&key, 1.0);
        } else {
            shard_b.add(&key, 1.0);
        }
    }

    // Merge shards into a single aggregate sketch. merge_from performs a
    // structural addition, so the resulting trie is equivalent to building a
    // sketch from the unified stream.
    let mut aggregate = shard_a.clone();
    aggregate.merge_from(&shard_b);

    println!(
        "Per-tenant estimate for tenant 7: {:.2}",
        aggregate.estimate("/tenant/7")
    );
    println!("Total nodes materialized: {}", aggregate.node_count());

    // Merging is associative, so we can merge into an empty sketch when
    // reconstructing from persisted shards.
    let mut reconstructed = GpsSketch::with_seed(0.5, 99);
    reconstructed.merge_from(&shard_a);
    reconstructed.merge_from(&shard_b);
    println!(
        "Reconstructed matches original? {}",
        approx_eq(
            aggregate.estimate("/tenant/42"),
            reconstructed.estimate("/tenant/42")
        )
    );
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}
