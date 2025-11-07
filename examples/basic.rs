use geometric_prefix_sketch::GpsSketch;

fn main() {
    // Seeded sketch so different shards can be merged deterministically.
    let mut sketch = GpsSketch::with_seed(0.5, 123);

    let keys = [
        ("/api/v1/users", 1.0),
        ("/api/v1/orders", 1.0),
        ("/api/v1/users", 1.0),
        ("/api/v2/users", 1.0),
        ("/static/css", 1.0),
    ];

    // Ingest the stream. Each update performs expected O(1) work regardless of
    // key length.
    for (key, delta) in keys {
        sketch.add(key, delta);
    }

    println!("Total traffic: {:.2}", sketch.total());
    println!("/api/v1/* estimate: {:.2}", sketch.estimate("/api/v1"));

    println!("Realized prefixes (first 5):");
    // Iteration reveals which prefixes were actually materialized; useful for
    // debugging or exporting summaries.
    for (prefix, estimate) in sketch.iter_estimates().take(5) {
        println!("  {} -> {:.2}", String::from_utf8_lossy(&prefix), estimate);
    }

    // Build a second shard and merge it back. Because both sketches share alpha
    // and the seed, the merged result is equivalent to processing all events in
    // a single sketch.
    let mut other = GpsSketch::with_seed(0.5, 123);
    other.add("/api/v1/reports", 5.0);
    sketch.merge_from(&other);
    println!(
        "After merge: /api/v1 estimate = {:.2}",
        sketch.estimate("/api/v1")
    );

    // Drop any prefixes whose absolute estimate is below 0.5 to reclaim memory.
    let removed = sketch.prune_by_estimate(0.5);
    println!("Pruned {} low-signal prefixes", removed);
}
