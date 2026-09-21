use rand::prelude::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};
use tfhe::prelude::*;
use tfhe::{CompactCiphertextList, CompactPublicKey, ConfigBuilder, FheUint8, generate_keys};

// Image constants
const IMAGE_WIDTH: usize = 3840;
const IMAGE_HEIGHT: usize = 2160;
const CHANNELS: usize = 1; // Assuming grayscale
const FULL_IMAGE_BYTES: usize = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS;

// Experiment constants
const FULL_IMAGE: bool = true; // Set to `true` to process the full image, or `false` to use a sample size
//   Amount to sample for testing
//     If `FULL_IMAGE` is `false`, this value will be used to determine how many values to process for testing.
//     If `FULL_IMAGE` is `true`, this value acts as the batch size for streaming.
const SAMPLE_SIZE: usize = 5_000;

/// Print timings
///
/// `n: usize` - number of values processed
/// `elapsed: Duration` - total time taken
/// `threads: usize` - number of threads used
fn report(n: usize, elapsed: Duration, threads: usize) {
    let per_value = elapsed / n as u32;
    let full_image_estimate = per_value.as_secs_f64() * FULL_IMAGE_BYTES as f64;

    println!(
        "  Sample: {} values in {:?} ({} thread(s))",
        n, elapsed, threads
    );

    println!("  Effective per-value wall-clock time: {:?}", per_value);
    println!(
        "  Extrapolated to a full 4K RGB image ({} bytes): {:.2}s",
        FULL_IMAGE_BYTES, full_image_estimate
    );
}

/// Report the speedup
///
/// `single: Duration` - time taken for single-threaded execution
/// `parallel: Duration` - time taken for parallel execution
fn print_speedup(single: Duration, parallel: Duration) {
    let speedup = single.as_secs_f64() / parallel.as_secs_f64();
    println!("  Speedup vs. single-threaded: {:.2}x", speedup);
}

/// Progress indicator for a streaming run.
///
/// `processed: usize` - number of values processed so far.
/// `total: usize` - total number of values to process.
/// `last_pct: &mut usize` - last percentage milestone printed, updated in place.
fn print_progress(processed: usize, total: usize, last_pct: &mut usize) {
    let pct = (processed * 100) / total;
    if pct >= *last_pct + 10 || processed == total {
        println!("    ...{}% ({} / {} values)", pct, processed, total);
        *last_pct = pct - (pct % 10);
    }
}

fn run_streaming<F, P>(
    total: usize,
    batch_size: usize,
    mut make_batch: F,
    mut process_batch: P,
) -> Duration
where
    F: FnMut(usize) -> Vec<u8>,
    P: FnMut(&[u8]),
{
    let mut elapsed = Duration::ZERO;
    let mut processed = 0usize;
    let mut last_pct = 0usize;

    while processed < total {
        let this_batch = batch_size.min(total - processed);
        let batch_values = make_batch(this_batch);

        let start = Instant::now();
        process_batch(&batch_values);
        elapsed += start.elapsed();
        // `batch_values` is dropped here, freeing memory for the next batch

        processed += this_batch;
        print_progress(processed, total, &mut last_pct);
    }

    elapsed
}

fn main() {
    println!("== TFHE-rs encryption benchmark ==");
    println!(
        "Full 4K RGB image = {} x {} x {} = {} byte-values",
        IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, FULL_IMAGE_BYTES
    );

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("Detected {} available CPU cores\n", cores);

    let total_size = if FULL_IMAGE {
        FULL_IMAGE_BYTES
    } else {
        SAMPLE_SIZE
    };

    println!(
        "Using size: {} for {}{}",
        SAMPLE_SIZE,
        if FULL_IMAGE { "batch size" } else { "testing" },
        if FULL_IMAGE {
            format!(" (streaming {} total_values)", total_size)
        } else {
            String::new()
        }
    );

    println!("Generating client key (this can take a few seconds)...");
    let config = ConfigBuilder::default().build();
    let (client_key, _server_key) = generate_keys(config);
    let compact_pk = CompactPublicKey::new(&client_key);

    let mut rng = rand::rng();

    // Generates `n` random byte values.
    //
    // Used for both the small sample case and the per-batch generator when streaming.
    let mut make_values = |n: usize| -> Vec<u8> { (0..n).map(|_| rng.random()).collect() };

    println!("\n{}", "#".repeat(28).as_str());
    println!("# PART 1: single-threaded  #");
    println!("{}", "#".repeat(28).as_str());

    println!("\n[1a] Naive per-value ClientKey encryption (1 thread)");
    let single_naive = if FULL_IMAGE {
        run_streaming(total_size, SAMPLE_SIZE, &mut make_values, |batch| {
            let ciphertexts: Vec<FheUint8> = batch
                .iter()
                .map(|v| FheUint8::encrypt(*v, &client_key))
                .collect();
            std::hint::black_box(&ciphertexts);
        })
    } else {
        let values = make_values(SAMPLE_SIZE);
        let start = Instant::now();
        let ciphertexts: Vec<FheUint8> = values
            .iter()
            .map(|v| FheUint8::encrypt(*v, &client_key))
            .collect();
        let elapsed = start.elapsed();
        std::hint::black_box(&ciphertexts);
        elapsed
    };

    report(total_size, single_naive, 1);

    println!("\n[1b] Batched CompactPublicKey encryption (1 thread)");

    let single_compact = if FULL_IMAGE {
        run_streaming(total_size, SAMPLE_SIZE, &mut make_values, |batch| {
            let mut builder = CompactCiphertextList::builder(&compact_pk);

            for v in batch {
                builder.push(*v);
            }

            let compact_list = builder.build();
            std::hint::black_box(&compact_list);
        })
    } else {
        let values = make_values(SAMPLE_SIZE);
        let start = Instant::now();
        let mut builder = CompactCiphertextList::builder(&compact_pk);

        for v in &values {
            builder.push(*v);
        }

        let compact_list = builder.build();
        let elapsed = start.elapsed();
        std::hint::black_box(&compact_list);
        elapsed
    };

    report(total_size, single_compact, 1);

    println!("\n{}", "#".repeat(31).as_str());
    println!("# PART 2: parallel ({} cores) #", cores);
    println!("{}", "#".repeat(31).as_str());

    println!("\n[2a] Naive per-value ClientKey encryption (parallel)");

    let parallel_naive = if FULL_IMAGE {
        run_streaming(total_size, SAMPLE_SIZE, &mut make_values, |batch| {
            let ciphertexts: Vec<FheUint8> = batch
                .par_iter()
                .map(|v| FheUint8::encrypt(*v, &client_key))
                .collect();
            std::hint::black_box(&ciphertexts);
        })
    } else {
        let values = make_values(SAMPLE_SIZE);
        let start = Instant::now();

        let ciphertexts: Vec<FheUint8> = values
            .par_iter()
            .map(|v| FheUint8::encrypt(*v, &client_key))
            .collect();

        let elapsed = start.elapsed();
        std::hint::black_box(&ciphertexts);
        elapsed
    };

    report(total_size, parallel_naive, cores);
    print_speedup(single_naive, parallel_naive);

    println!("\n[2b] Batched CompactPublicKey encryption (parallel)");

    let parallel_compact = if FULL_IMAGE {
        run_streaming(total_size, SAMPLE_SIZE, &mut make_values, |batch| {
            let chunk_size = (batch.len() * cores - 1) / cores;
            let compact_lists: Vec<CompactCiphertextList> = batch
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut builder = CompactCiphertextList::builder(&compact_pk);

                    for v in chunk {
                        builder.push(*v);
                    }

                    builder.build()
                })
                .collect();
            std::hint::black_box(&compact_lists);
        })
    } else {
        let values = make_values(SAMPLE_SIZE);
        let chunk_size = (SAMPLE_SIZE * cores - 1) / cores;
        let start = Instant::now();

        let compact_lists: Vec<CompactCiphertextList> = values
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut builder = CompactCiphertextList::builder(&compact_pk);

                for v in chunk {
                    builder.push(*v);
                }

                builder.build()
            })
            .collect();

        let elapsed = start.elapsed();
        std::hint::black_box(&compact_lists);
        elapsed
    };

    report(total_size, parallel_compact, cores);
    print_speedup(single_compact, parallel_compact);

    println!("\nDone.")
}
