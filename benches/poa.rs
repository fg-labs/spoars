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
    align_and_add, AlignmentEngine, AlignmentType, Scoring, SimdEngine, SisdEngine,
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

fn label(n: usize, len: usize) -> String {
    format!("{n}x{len}")
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
    bench_align_one,
    bench_add_alignment,
    bench_accessors,
    bench_sisd_vs_simd,
);
criterion_main!(benches);
