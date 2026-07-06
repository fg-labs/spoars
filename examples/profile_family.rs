//! Profiling harness for the fgumi small-family regime.
//!
//! Companion to `profile_align.rs`, but faithful to how `pairhmm-consensus-proto` drives
//! spoars: **Global (NW) + spoa's default convex gaps**, a small family (default 6 reads)
//! of short (~235 bp) molecules, rebuilt from scratch each pass — so a sampling profiler
//! sees the real per-family hotpath (align fill / backtrack plus the per-add
//! `topological_sort`), not the 200x1kbp assembly corpus.
//!
//! Build with debuginfo so the profiler can symbolicate:
//!   RUSTFLAGS="-C debuginfo=2" cargo build --release --example profile_family
//!
//! Profile (macOS, no sudo):
//!   samply record -- ./target/release/examples/profile_family 3000 6 235
//!   tricorder --trace ./target/release/examples/profile_family 3000 6 235
//!
//! Compare the scalar path:
//!   SPOARS_FORCE_SISD=1 ./target/release/examples/profile_family 3000 6 235
//!
//! Args: [repeats=3000] [reads=6] [len=235].

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SimdEngine, SisdEngine};
use spoars::graph::Graph;

// Reuse the deterministic corpus generator the Criterion benches drive, so the profiling
// harness and the benches sample the exact same family model (fixed-seed xorshift64, 1%
// substitution + 0.5% indel). `#[path]` includes the shared module without extracting a
// separate crate — the standard Cargo pattern for sharing helpers into an example.
#[path = "../benches/common/mod.rs"]
mod common;
use common::{mutant, random_molecule, Rng};

fn build_pass<E: AlignmentEngine>(engine: &mut E, reads: &[Vec<u8>]) -> i64 {
    let mut graph = Graph::new();
    let mut checksum = 0i64;
    for read in reads {
        let (alignment, score) = engine.align(read, &graph);
        checksum = checksum.wrapping_add(i64::from(score));
        graph
            .add_alignment_weight(&alignment, read, 1)
            .expect("well-formed alignment");
    }
    checksum
}

fn parse<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repeats: usize = parse(&args, 1, 3000);
    let n_reads: usize = parse(&args, 2, 6);
    let len: usize = parse(&args, 3, 235);

    let mut rng = Rng::new(0x5f0a_5b0a_d0be_1234);
    let truth = random_molecule(len, &mut rng);
    let reads: Vec<Vec<u8>> = (0..n_reads)
        .map(|_| mutant(&truth, 0.01, 0.005, &mut rng))
        .collect();

    let force_sisd = std::env::var("SPOARS_FORCE_SISD")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);

    let start = std::time::Instant::now();
    let mut checksum = 0i64;
    if force_sisd {
        let mut engine = SisdEngine::new(AlignmentType::Global, Scoring::spoa_default());
        for _ in 0..repeats {
            checksum = checksum.wrapping_add(build_pass(&mut engine, &reads));
        }
    } else {
        let mut engine = SimdEngine::new(AlignmentType::Global, Scoring::spoa_default());
        for _ in 0..repeats {
            checksum = checksum.wrapping_add(build_pass(&mut engine, &reads));
        }
    }
    let elapsed = start.elapsed();

    let engine_label = if force_sisd {
        "SisdEngine"
    } else {
        "SimdEngine"
    };
    let families = repeats;
    let per_family = elapsed.as_secs_f64() * 1e6 / families as f64;
    eprintln!(
        "{engine_label}: {families} families of {n_reads}x{len}bp (Global/convex) in {:.3}s \
         = {per_family:.1} us/family  (checksum {checksum})",
        elapsed.as_secs_f64()
    );
}
