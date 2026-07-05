//! Profiling harness: a long, single-engine align-and-add loop for `perf record`/`perf annotate`.
//!
//! Unlike `tests/simd_bench.rs` (which runs *both* engines and cross-checks their consensus), this
//! runs exactly ONE engine's align loop, repeated enough to give a sampling profiler a clean, long
//! hotspot. It exists only for the AVX2-vs-SSE4.1 profiling investigation and is not a correctness
//! check (parity is proven by the test suite).
//!
//! Usage:
//!   cargo build --release --example profile_align   # add debuginfo via RUSTFLAGS="-C debuginfo=2"
//!   SPOARS_FORCE_ISA=sse41 ./target/release/examples/profile_align [repeats] [reads] [len]
//!   SPOARS_FORCE_SISD=1    ./target/release/examples/profile_align [repeats] [reads] [len]
//!
//! Defaults: repeats=4, reads=200, len=1000 (~10-15s of AVX2 fill, ample for perf sampling). The
//! selected engine is whatever `SimdEngine` dispatches to on this host, downgraded per
//! `SPOARS_FORCE_ISA` (or forced scalar via `SPOARS_FORCE_SISD`, handled here to match the CLI).

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SimdEngine, SisdEngine};
use spoars::graph::Graph;

/// Dependency-free xorshift64 PRNG (fixed seed → reproducible corpus), mirroring `simd_bench.rs`.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

fn random_base(state: &mut u64) -> u8 {
    BASES[(xorshift64(state) % 4) as usize]
}

fn random_reference(len: usize, state: &mut u64) -> Vec<u8> {
    (0..len).map(|_| random_base(state)).collect()
}

/// `n_reads` substitution-only mutants (~`mutation_rate` per base), keeping every read the same
/// length as the reference so the graph width stays predictable across the loop.
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

/// One align-and-add pass over `reads`; returns a score checksum so the loop can't be optimized out.
fn align_add_pass<E: AlignmentEngine>(engine: &mut E, reads: &[Vec<u8>]) -> i64 {
    let mut graph = Graph::new();
    let mut checksum: i64 = 0;
    for read in reads {
        let (alignment, score) = engine.align(read, &graph);
        checksum = checksum.wrapping_add(i64::from(score));
        graph
            .add_alignment_weight(&alignment, read, 1)
            .expect("add_alignment_weight must succeed for a well-formed alignment");
    }
    checksum
}

fn parse_arg<T: std::str::FromStr>(args: &[String], idx: usize, default: T) -> T {
    args.get(idx)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repeats: usize = parse_arg(&args, 1, 4);
    let n_reads: usize = parse_arg(&args, 2, 200);
    let len: usize = parse_arg(&args, 3, 1000);

    let mut state = 0x5b0a_2510_c0de_5eed_u64;
    let reference = random_reference(len, &mut state);
    let reads = mutated_reads(&reference, n_reads, 0.03, &mut state);

    let alignment_type = AlignmentType::Local;
    let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();

    let force_sisd = std::env::var("SPOARS_FORCE_SISD")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);

    let start = std::time::Instant::now();
    let mut checksum: i64 = 0;
    if force_sisd {
        let mut engine = SisdEngine::new(alignment_type, scoring);
        for _ in 0..repeats {
            checksum = checksum.wrapping_add(align_add_pass(&mut engine, &reads));
        }
    } else {
        let mut engine = SimdEngine::new(alignment_type, scoring);
        for _ in 0..repeats {
            checksum = checksum.wrapping_add(align_add_pass(&mut engine, &reads));
        }
    }
    let elapsed = start.elapsed();

    let engine_label = if force_sisd {
        "SisdEngine".to_string()
    } else {
        format!(
            "SimdEngine (SPOARS_FORCE_ISA={})",
            std::env::var("SPOARS_FORCE_ISA").unwrap_or_else(|_| "<default>".to_string())
        )
    };
    eprintln!(
        "profile_align: {engine_label} {repeats}x({n_reads} reads x {len}bp local/SW) \
         elapsed={elapsed:?} checksum={checksum}",
    );
}
