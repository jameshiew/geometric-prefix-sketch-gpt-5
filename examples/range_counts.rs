//! Using GPS to approximate range counts on integers by treating them as binary
//! prefixes. This mimics DESGIN.md's "integer range" use case.

use geometric_prefix_sketch::GpsSketch;

fn main() {
    let mut sketch = GpsSketch::with_seed(0.55, 11);

    for value in 0u32..50_000 {
        let key = to_bit_path(value);
        sketch.add(&key, 1.0);
    }

    // Estimate how many values fell into [0, 16383] by querying the /0000 prefix
    // (top 14 bits zero). We can decompose arbitrary ranges into O(log U)
    // prefixes and sum their estimates.
    let prefix = to_bit_path_prefix(0, 14);
    let estimate = sketch.estimate(&prefix);
    println!("~[0, 16384) count: {:.0}", estimate);

    // Another range: values with leading bits 101 (roughly [0b101000.., 0b101111..)).
    let high_prefix = to_bit_path_prefix(0b1010_0000, 8);
    println!(
        "~values starting with 0b1010_0000: {:.0}",
        sketch.estimate(&high_prefix)
    );
}

fn to_bit_path(value: u32) -> Vec<u8> {
    let mut path = Vec::with_capacity(33);
    path.push(b'/');
    for bit in (0..32).rev() {
        path.push(if (value >> bit) & 1 == 1 { b'1' } else { b'0' });
    }
    path
}

fn to_bit_path_prefix(value: u32, bits: usize) -> Vec<u8> {
    let mut path = Vec::with_capacity(bits + 1);
    path.push(b'/');
    for bit in (0..bits).rev() {
        let b = (value >> bit) & 1;
        path.push(if b == 1 { b'1' } else { b'0' });
    }
    path
}
