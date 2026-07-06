//! Exploratory POA microbenchmarks.
//!
//! These target the regime the fgumi `pairhmm-consensus-proto` consumer actually drives:
//! **Global (NW) alignment + spoa's default convex gaps**, small families (3-10 reads) of
//! short (~235 bp) molecules, consumed via the graph accessors (`column_members` /
//! `msa_columns` / `sequence_path_iter`) rather than `generate_consensus`. A larger
//! `50 x 1000` point is included only as a scaling reference.
//!
//! Run:            cargo bench
//! One group:      cargo bench -- build_family
//! Quick pass:     cargo bench -- --warm-up-time 1 --measurement-time 2 --sample-size 20
//!
//! They exercise only the public API, so per-function *internal* attribution (fill vs
//! destripe vs backtrack vs topological_sort) comes from a sampling profiler — see
//! `benches/README.md`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

use spoars::align::{
    align_and_add, AlignmentEngine, AlignmentType, BandConfig, Scoring, SimdEngine, SisdEngine,
};
use spoars::graph::Graph;

mod common;
use common::{family, REGIME};

/// Fixed corpus seed — one value so every group aligns the same molecules.
const SEED: u64 = 0x5F0A_5B0A_D0BE_1234;

/// Build a POA graph from `reads` with a fresh `SimdEngine`, mirroring the consumer's
/// per-family setup (`SimdEngine::new` + `align_and_add` per read, Global/convex).
fn build_simd(reads: &[Vec<u8>]) -> Graph {
    let mut engine = SimdEngine::new(AlignmentType::Global, Scoring::spoa_default());
    let mut graph = Graph::new();
    for r in reads {
        align_and_add(&mut graph, &mut engine, r, 1).expect("valid alignment");
    }
    graph
}

/// Same, with the scalar `SisdEngine` (for the SIMD-vs-scalar crossover group).
fn build_sisd(reads: &[Vec<u8>]) -> Graph {
    let mut engine = SisdEngine::new(AlignmentType::Global, Scoring::spoa_default());
    let mut graph = Graph::new();
    for r in reads {
        align_and_add(&mut graph, &mut engine, r, 1).expect("valid alignment");
    }
    graph
}

/// Same as `build_simd`, but with the opt-in **banded** engine (abPOA-style adaptive band,
/// `BandConfig::default()`), mirroring the consumer's per-family setup so the two groups are
/// apples-to-apples: same corpus, same Global/convex scoring, same `align_and_add` loop.
fn build_simd_banded(reads: &[Vec<u8>]) -> Graph {
    let mut engine = SimdEngine::banded(
        AlignmentType::Global,
        Scoring::spoa_default(),
        BandConfig::default(),
    );
    let mut graph = Graph::new();
    for r in reads {
        align_and_add(&mut graph, &mut engine, r, 1).expect("valid alignment");
    }
    graph
}

fn label(n: usize, len: usize) -> String {
    format!("{n}x{len}")
}

/// **Analytical** (not measured) cells-computed ratio for one family point: how much less DP-cell
/// area the banded fill scans versus the exact fill, estimated purely from the public band
/// geometry (`BandConfig::width`) and the built graph's node count — no per-cell instrumentation
/// of the hot fill path. Criterion only times wall-clock, so this is the separate, explicit
/// "why" behind any wall-clock speedup (or lack of one).
///
/// The estimate treats every DP row (one per graph node) as `2*w+1` query columns wide under
/// banding versus `query_len+1` columns exact, where `w = BandConfig::width(query_len)`. This is
/// coarse: it ignores per-row anchor drift (rows near the query's start/end have a narrower
/// clamped window than `2*w+1`) and vector-lane quantization (`segment_range` rounds the window
/// out to whole SIMD lanes). Both effects make the true banded fraction slightly *higher* than
/// this estimate, so treat it as a lower bound on cells scanned, not an exact figure.
fn analytical_cells_ratio(graph: &Graph, reads: &[Vec<u8>], band: BandConfig) -> f64 {
    let rows = graph.num_nodes() as f64;
    let mut exact_cells = 0.0f64;
    let mut banded_cells = 0.0f64;
    for r in reads {
        let full_width = (r.len() + 1) as f64;
        let w = band.width(r.len()) as f64;
        let banded_width = (2.0 * w + 1.0).min(full_width);
        exact_cells += rows * full_width;
        banded_cells += rows * banded_width;
    }
    banded_cells / exact_cells
}

/// 1. End-to-end family build — what the consumer feels per UMI family.
fn bench_build_family(c: &mut Criterion) {
    let mut g = c.benchmark_group("build_family");
    for &(n, len) in REGIME {
        let (_truth, reads) = family(len, n, SEED);
        let total: u64 = reads.iter().map(|r| r.len() as u64).sum();
        g.throughput(Throughput::Bytes(total));
        g.bench_with_input(
            BenchmarkId::from_parameter(label(n, len)),
            &reads,
            |b, reads| {
                b.iter(|| black_box(build_simd(black_box(reads))));
            },
        );
    }
    g.finish();
}

/// 2. A single align into an already-built (n-1)-read graph — isolates align cost from
///    graph-mutation cost.
fn bench_align_one(c: &mut Criterion) {
    let mut g = c.benchmark_group("align_one");
    for &(n, len) in REGIME {
        if n < 2 {
            continue;
        }
        let (_truth, reads) = family(len, n, SEED);
        let (init, last) = reads.split_at(n - 1);
        let graph = build_simd(init);
        let last = last[0].clone();
        let mut engine = SimdEngine::new(AlignmentType::Global, Scoring::spoa_default());
        g.throughput(Throughput::Bytes(last.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label(n, len)), &(), |b, _| {
            b.iter(|| black_box(engine.align(black_box(&last), black_box(&graph))));
        });
    }
    g.finish();
}

/// 2b. Banded counterpart of `bench_build_family` — same corpus, same Global/convex scoring,
///     `SimdEngine::banded(.., BandConfig::default())` instead of the exact engine. Registered in
///     the same `criterion_group!` run as `bench_build_family` so `cargo bench -- build_family`
///     produces both exact and banded numbers in one pass for direct comparison. Also prints the
///     analytical cells-computed ratio per point (see `analytical_cells_ratio`) — labeled
///     "analytical" because it is derived from band geometry, not from measuring cells at runtime.
fn bench_build_family_banded(c: &mut Criterion) {
    let mut g = c.benchmark_group("build_family_banded");
    for &(n, len) in REGIME {
        let (_truth, reads) = family(len, n, SEED);
        let total: u64 = reads.iter().map(|r| r.len() as u64).sum();
        g.throughput(Throughput::Bytes(total));

        // One-off build (outside criterion's timed loop) purely to report the analytical
        // cells-computed ratio for this point; the timed loop below rebuilds independently.
        let graph = build_simd_banded(&reads);
        let ratio = analytical_cells_ratio(&graph, &reads, BandConfig::default());
        println!(
            "[cells-ratio, analytical] build_family_banded/{}: banded/exact = {ratio:.4} \
             ({:.1}x fewer cells)",
            label(n, len),
            1.0 / ratio,
        );

        g.bench_with_input(
            BenchmarkId::from_parameter(label(n, len)),
            &reads,
            |b, reads| {
                b.iter(|| black_box(build_simd_banded(black_box(reads))));
            },
        );
    }
    g.finish();
}

/// 2c. Banded counterpart of `bench_align_one` — isolates the banded align cost alone (no graph
///     mutation), mirroring `bench_align_one`'s (n-1)-read setup so the two are directly
///     comparable.
fn bench_align_one_banded(c: &mut Criterion) {
    let mut g = c.benchmark_group("align_one_banded");
    for &(n, len) in REGIME {
        if n < 2 {
            continue;
        }
        let (_truth, reads) = family(len, n, SEED);
        let (init, last) = reads.split_at(n - 1);
        let graph = build_simd_banded(init);
        let last = last[0].clone();
        let mut engine = SimdEngine::banded(
            AlignmentType::Global,
            Scoring::spoa_default(),
            BandConfig::default(),
        );
        g.throughput(Throughput::Bytes(last.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label(n, len)), &(), |b, _| {
            b.iter(|| black_box(engine.align(black_box(&last), black_box(&graph))));
        });
    }
    g.finish();
}

/// 3. Graph mutation alone (merge + the `topological_sort()` that `add_alignment` runs on
///    every add) given a precomputed alignment — tests whether topo-sort-per-add is hot at
///    small N. `iter_batched` clones a fresh template graph per iteration.
fn bench_add_alignment(c: &mut Criterion) {
    let mut g = c.benchmark_group("add_alignment");
    for &(n, len) in REGIME {
        if n < 2 {
            continue;
        }
        let (_truth, reads) = family(len, n, SEED);
        let (init, last) = reads.split_at(n - 1);
        let last = last[0].clone();
        let template = build_simd(init);
        // Precompute the Nth read's alignment against the (n-1)-read graph once; the
        // template is identical every iteration, so the alignment stays valid.
        let mut engine = SimdEngine::new(AlignmentType::Global, Scoring::spoa_default());
        let (alignment, _score) = engine.align(&last, &template);
        g.throughput(Throughput::Bytes(last.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label(n, len)), &(), |b, _| {
            b.iter_batched(
                || template.clone(),
                |mut graph| {
                    graph
                        .add_alignment_weight(black_box(&alignment), black_box(&last), 1)
                        .expect("valid alignment");
                    black_box(graph)
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

/// 4. The consumer's read-out path: accessor traversals over a built graph.
fn bench_accessors(c: &mut Criterion) {
    let mut g = c.benchmark_group("accessors");
    let (_truth, reads) = family(235, 10, SEED); // largest fgumi-regime family
    let graph = build_simd(&reads);
    let n = reads.len();

    g.bench_function("column_members", |b| {
        b.iter(|| black_box(graph.column_members()));
    });
    g.bench_function("msa_columns", |b| {
        b.iter(|| black_box(graph.msa_columns()));
    });
    g.bench_function("sequence_path_iter_all", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for seq in 0..n {
                for node in graph.sequence_path_iter(seq) {
                    acc = acc.wrapping_add(u64::from(node.0));
                }
            }
            black_box(acc)
        });
    });
    g.finish();
}

/// 5. Does SIMD pay off at 235 bp, or does per-align setup eat the win? Same family build
///    under both engines.
fn bench_sisd_vs_simd(c: &mut Criterion) {
    let mut g = c.benchmark_group("sisd_vs_simd");
    for &(n, len) in &[(6usize, 235usize), (10, 235), (50, 1000)] {
        let (_truth, reads) = family(len, n, SEED);
        let total: u64 = reads.iter().map(|r| r.len() as u64).sum();
        g.throughput(Throughput::Bytes(total));
        g.bench_with_input(
            BenchmarkId::new("simd", label(n, len)),
            &reads,
            |b, reads| {
                b.iter(|| black_box(build_simd(black_box(reads))));
            },
        );
        g.bench_with_input(
            BenchmarkId::new("sisd", label(n, len)),
            &reads,
            |b, reads| {
                b.iter(|| black_box(build_sisd(black_box(reads))));
            },
        );
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_build_family,
    bench_build_family_banded,
    bench_align_one,
    bench_align_one_banded,
    bench_add_alignment,
    bench_accessors,
    bench_sisd_vs_simd,
);
criterion_main!(benches);
