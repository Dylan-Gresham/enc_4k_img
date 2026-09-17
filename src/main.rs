use rand::prelude::*;
use std::time::Instant;
use tfhe::prelude::*;
use tfhe::{CompactCiphertextList, CompactPublicKey, ConfigBuilder, FheUint8, generate_keys};

// Image constants
const IMAGE_WIDTH: usize = 3840;
const IMAGE_HEIGHT: usize = 2160;
const CHANNELS: usize = 1; // Assuming we're still doing grayscale
const FULL_IMAGE_BYTES: usize = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS;

// Experiment constants
const SAMPLE_SIZE: usize = 5_000;

/// Print timings
///
/// `n: usize` - number of values processed
/// `elapsed: std::time::Duration` - time taken to process those values
fn report(n: usize, elapsed: std::time::Duration) {
    let per_value = elapsed / n as u32;
    let full_image_estimate = per_value.as_secs_f64() * FULL_IMAGE_BYTES as f64;

    println!("  Sample: {} values in {:?}", n, elapsed);
    println!("  Per value: {:?}", per_value);
    println!(
        "  Extrapolated to a full 4K RGB image ({} bytes): {:.2}s (single-threaded)",
        FULL_IMAGE_BYTES, full_image_estimate
    );

    if full_image_estimate > 1.0 {
        let cores = 8.0;
        println!(
            "    ~{:.2}s if parallelized across {} cores",
            full_image_estimate / cores,
            cores
        );
    }
}

fn main() {
    println!("== TFHE-rs encryption benchmark ==");
    println!(
        "Full 4K RGB image = {} x {} x {} = {} byte-values\n",
        IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, FULL_IMAGE_BYTES
    );

    println!("Generating client key (this can take a few seconds)...");
    let config = ConfigBuilder::default().build();
    let (client_key, _server_key) = generate_keys(config);

    let mut rng = rand::rng();
    let values: Vec<u8> = (0..SAMPLE_SIZE).map(|_| rng.random()).collect();

    println!("\n[Method 1] Naive per-value ClientKey encryption");
    let start = Instant::now();
    let ciphertexts: Vec<FheUint8> = values
        .iter()
        .map(|v| FheUint8::encrypt(*v, &client_key))
        .collect();

    let elapsed = start.elapsed();
    report(SAMPLE_SIZE, elapsed);
    std::hint::black_box(&ciphertexts); // Prevent compiler optimizations

    println!("\n[Method 2] Batched CompactPublicKey encryption");
    let compact_pk = CompactPublicKey::new(&client_key);

    let start = Instant::now();
    let mut builder = CompactCiphertextList::builder(&compact_pk);
    for v in &values {
        builder.push(*v);
    }

    let compact_list = builder.build();
    let elapsed = start.elapsed();

    report(SAMPLE_SIZE, elapsed);
    std::hint::black_box(&compact_list);

    println!("\nDone.");
}
