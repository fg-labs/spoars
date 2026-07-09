//! Sanity wall-clock benchmark: [`SimdEngine`] vs [`SisdEngine`] on a larger synthetic corpus.
//!
//! This is NOT a rigorous perf study (no warmup iterations, no statistical repeats, no criterion)
//! — it exists only to confirm the dispatching [`SimdEngine`] (SIMD kernels plan Task 16) actually
//! runs a vectorized kernel end to end and is plausibly faster than the scalar [`SisdEngine`] on an
//! input large enough for the DP fill to dominate over per-alignment fixed overhead. On this Mac,
//! `SimdEngine` dispatches to NEON.
//!
//! Not run by default (`#[ignore]`): run explicitly with
//! `cargo test --release --test simd_bench -- --ignored --nocapture` to see the timings (release
//! is important here — a debug build's constant-factor overhead swamps any SIMD/SISD difference).

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SimdEngine, SisdEngine};
use spoars::graph::Graph;

/// A tiny, dependency-free xorshift64 PRNG, seeded fixed for reproducible benchmark input (the
/// benchmark cares about relative wall-clock, not about matching any particular biological
/// distribution, so a minimal generator is preferable to pulling in `rand` as a dependency for a
/// single ignored test).
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

const BASES: [u8; 4] = *b"ACGT";

fn random_base(state: &mut u64) -> u8 {
    BASES[(xorshift64(state) % 4) as usize]
}

/// A random reference sequence of `len` bases.
fn random_reference(len: usize, state: &mut u64) -> Vec<u8> {
    (0..len).map(|_| random_base(state)).collect()
}

/// `n_reads` substitution-only mutants of `reference` (~`mutation_rate` per-base probability). No
/// indels, so every read stays the same length as `reference` and the resulting graph's width
/// stays close to `reference.len()` throughout the align-and-add loop, keeping the benchmark's
/// per-alignment cost predictable across both engines.
fn mutated_reads(
    reference: &[u8],
    n_reads: usize,
    mutation_rate: f64,
    state: &mut u64,
) -> Vec<Vec<u8>> {
    (0..n_reads)
        .map(|_| {
            reference
                .iter()
                .map(|&base| {
                    let roll = (xorshift64(state) % 1_000_000) as f64 / 1_000_000.0;
                    if roll < mutation_rate {
                        random_base(state)
                    } else {
                        base
                    }
                })
                .collect()
        })
        .collect()
}

/// Runs the align-and-add loop (mirroring the CLI's own loop in `src/bin/spoars.rs`) for `engine`
/// over `reads`, returning the built [`Graph`] and the loop's total wall-clock duration.
fn timed_align_add_loop<E: AlignmentEngine>(
    mut engine: E,
    reads: &[Vec<u8>],
) -> (Graph, std::time::Duration) {
    let mut graph = Graph::new();
    let start = std::time::Instant::now();
    for read in reads {
        let (alignment, _score) = engine.align(read, &graph);
        graph
            .add_alignment_weight(&alignment, read, 1)
            .expect("add_alignment_weight must succeed for a well-formed alignment");
    }
    (graph, start.elapsed())
}

#[test]
#[ignore = "sanity wall-clock benchmark, not a correctness check; run with \
            `cargo test --release --test simd_bench -- --ignored --nocapture`"]
fn simd_engine_is_plausibly_faster_than_sisd_on_a_larger_synthetic_corpus() {
    // Fixed seed: deterministic input across runs/hosts, matching this repo's convention
    // (`tests/support/generators.rs`'s `FIXED_RNG_SEED`) for reproducible test/benchmark input.
    let mut state = 0x5b0a_2510_c0de_5eed_u64;
    let reference = random_reference(1_000, &mut state);
    let reads = mutated_reads(&reference, 80, 0.03, &mut state);

    let alignment_type = AlignmentType::Local;
    let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();

    let simd_engine = SimdEngine::new(alignment_type, scoring);
    let (mut simd_graph, simd_elapsed) = timed_align_add_loop(simd_engine, &reads);

    let sisd_engine = SisdEngine::new(alignment_type, scoring);
    let (mut sisd_graph, sisd_elapsed) = timed_align_add_loop(sisd_engine, &reads);

    // Cheap end-to-end cross-check on this corpus (the bit-exact "SimdEngine == SisdEngine"
    // contract is already proven per-alignment at the unit-test level in `src/align/simd/mod.rs`;
    // this just confirms the built-up graphs still agree after 80 alignments).
    assert_eq!(
        simd_graph.generate_consensus(),
        sisd_graph.generate_consensus(),
        "SimdEngine and SisdEngine must build equivalent graphs over the synthetic corpus"
    );

    let speedup = sisd_elapsed.as_secs_f64() / simd_elapsed.as_secs_f64().max(1e-9);
    eprintln!(
        "simd_bench: {} reads x {}bp (local/SW, default scores): SimdEngine={simd_elapsed:?} \
         SisdEngine={sisd_elapsed:?} (SimdEngine {speedup:.2}x SisdEngine)",
        reads.len(),
        reference.len(),
    );
}
