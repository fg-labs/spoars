# spoars

[![CI](https://github.com/fg-labs/spoars/actions/workflows/check.yml/badge.svg)](https://github.com/fg-labs/spoars/actions/workflows/check.yml)
[![crates.io](https://img.shields.io/crates/v/spoars.svg)](https://crates.io/crates/spoars)
[![docs.rs](https://docs.rs/spoars/badge.svg)](https://docs.rs/spoars)
[![codecov](https://codecov.io/gh/fg-labs/spoars/branch/main/graph/badge.svg)](https://codecov.io/gh/fg-labs/spoars)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A faithful, memory-safe, SIMD-accelerated native-Rust reimplementation of [spoa](https://github.com/rvaser/spoa) — the C++ partial order alignment (POA) library for consensus generation and multiple sequence alignment.

## What it is

Partial order alignment builds a directed acyclic graph from a set of related sequences: each sequence is aligned to the graph built so far and merged into it, so shared subsequences collapse onto shared paths. The resulting DAG yields a **consensus** sequence and a **multiple sequence alignment (MSA)**, and can be exported as [GFA](https://gfa-spec.github.io/GFA-spec/) or Graphviz DOT.

`spoars` reproduces spoa v4.1.5's output **bit-for-bit** — the same dynamic-programming tie-breaks, the same consensus and MSA, the same GFA/DOT — verified against the C++ library through a differential oracle. Where a C++ idiom and a natural Rust one would diverge in observable behavior, the C++ behavior wins.

## Features

- **Bit-exact with spoa** across all nine alignment modes: `{Local, Global, Overlap}` × `{linear, affine, convex}` gap penalties.
- **SIMD-accelerated**, with a portable scalar fallback that produces identical results. Runtime dispatch to SSE4.1 / AVX2 on x86-64 and NEON on aarch64.
- **`#![deny(unsafe_code)]`** everywhere except the isolated SIMD kernels module.
- Consensus generation (with a minimum-coverage variant), MSA, and GFA/DOT export.
- Read-only accessors for inspecting the built graph, and an `AlignmentEngine` trait you can implement yourself.

## Quick start

```rust
use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SimdEngine};
use spoars::graph::Graph;

// Match, mismatch, gap-open, gap-extend, and a second (convex) gap-open/extend pair.
let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
let mut engine = SimdEngine::new(AlignmentType::Global, scoring);

let reads: [&[u8]; 3] = [b"ACGTACGT", b"ACGTTCGT", b"ACGTACGT"];

let mut graph = Graph::new();
for read in reads {
    let (alignment, _score) = engine.align(read, &graph);
    graph.add_alignment_weight(&alignment, read, 1).unwrap();
}

assert_eq!(graph.generate_consensus(), "ACGTACGT");
let msa = graph.generate_msa(false); // one row per read
assert_eq!(msa.len(), 3);
```

## Engines: scalar and SIMD, same answer

Alignment goes through the `AlignmentEngine` trait. Two implementations ship:

- **`SisdEngine`** — a portable scalar engine, correct on every target.
- **`SimdEngine`** — runtime-dispatched SSE4.1 / AVX2 (x86-64) or NEON (aarch64) with a scalar fallback, **bit-identical** to `SisdEngine`. It vectorizes the DP fill and reuses the scalar backtrack, so the accelerated path never changes the result — only the speed.

On one representative consensus workload (200 reads × ~1 kbp, convex gaps; `--release` build), `SimdEngine` ran roughly **4–5× faster** than `SisdEngine` on the machines measured below. These are observed results for that specific setup, not a guarantee — the speedup varies with CPU, toolchain, and input shape (read count, length, gap mode), so benchmark your own workload before relying on a number.

| ISA | machine measured | speedup vs scalar |
|---|---|---|
| NEON | Apple M-series / AWS Graviton | ~4.9× |
| AVX2 | x86-64 | ~4.0× |
| SSE4.1 | x86-64 | ~4.0× |

x86 SIMD is memory-bound on the striped→row-major de-stripe rather than on the vectorized fill, which is why AVX2 and SSE4.1 land close together; a faithful (non-Farrar, non-int8) port keeps the row-wise recurrence that makes them comparable.

You can also implement `AlignmentEngine` yourself — for a different scoring model, a banded fill, or a test mock — and feed the result straight into `Graph::add_alignment`. See the trait's documentation for the alignment format and a worked example.

## Inspecting the graph

Beyond building and summarizing the graph, `Graph` exposes read-only accessors so downstream code can walk the DAG directly: `nodes()` / `edges()` and their `node()` / `edge()` id lookups, `num_nodes()` / `num_edges()`, `encode()` / `decode()` to convert between raw bytes and internal symbol codes, and `rank_order()` / `sequence_starts()` / `consensus_nodes()` for traversal. `Node` carries `coverage()`, `successor()`, and `base()` helpers.

## Faithfulness and testing

Correctness is verified two ways:

- A **C++ differential oracle** (under `oracle/`) links the pinned spoa submodule (built without `-march` flags, forcing its scalar path) and is compared against `spoars` on generated inputs via property-based tests.
- The scalar `SisdEngine` then serves as an **in-process oracle** for the SIMD kernels: every `SimdEngine` result is checked bit-for-bit against `SisdEngine` across all nine modes and both the int16 and int32 lane widths.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build/test setup, the git hooks, and the bit-exact faithfulness contract.

## License

MIT — see [LICENSE](LICENSE). Third-party attribution (spoa) is in [THIRD-PARTY.md](THIRD-PARTY.md).
