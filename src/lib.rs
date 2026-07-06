// `deny`, not `forbid`: a crate-level `forbid(unsafe_code)` cannot be relaxed by an inner
// `#[allow(unsafe_code)]` (rustc E0453), but `deny` can. The `simd` module needs
// `#![allow(unsafe_code)]` for hand-tuned intrinsics; everywhere else in the crate `deny` still
// denies `unsafe` by default, so this is the standard idiom for a crate with exactly one
// unsafe-code module.
#![deny(unsafe_code)]
//! `spoars` is a faithful, memory-safe, SIMD-accelerated native-Rust reimplementation of
//! [spoa](https://github.com/rvaser/spoa) — the C++ partial order alignment (POA) library used for
//! consensus generation and multiple sequence alignment.
//!
//! Partial order alignment builds a directed acyclic graph ([`graph::Graph`]) from a set of related
//! sequences: each sequence is aligned to the graph so far and merged into it, so shared subsequences
//! collapse onto shared paths. The resulting DAG yields a consensus sequence and a multiple sequence
//! alignment (MSA), and can be exported as [GFA](https://gfa-spec.github.io/GFA-spec/) or Graphviz
//! DOT.
//!
//! # What "faithful" means
//!
//! `spoars` reproduces spoa v4.1.5's output **bit-for-bit** — the same DP tie-breaks, the same
//! consensus/MSA, the same GFA/DOT — validated against the C++ library through a differential oracle.
//! Where a C++ idiom and a natural Rust one diverge in observable behavior, the C++ behavior wins.
//!
//! # Engines: scalar and SIMD, same answer
//!
//! Alignment goes through the [`align::AlignmentEngine`] trait. Two implementations ship:
//!
//! - [`align::SisdEngine`] — a portable scalar engine, correct on every target.
//! - [`align::SimdEngine`] — runtime-dispatched SSE4.1 / AVX2 (x86-64) or NEON (aarch64) with a
//!   scalar fallback, **bit-identical** to `SisdEngine`. It vectorizes the DP fill and reuses the
//!   scalar backtrack, so the accelerated path never changes the result — only the speed (observed
//!   at roughly 4–5× the scalar engine on one consensus workload; see the README for the measured
//!   setup — actual speedup varies with CPU, toolchain, and input).
//!
//! Downstream crates can also implement [`align::AlignmentEngine`] themselves; see that trait for
//! the alignment format and a worked example.
//!
//! # Quick start
//!
//! Build a graph from a few reads and generate a consensus and MSA:
//!
//! ```
//! use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SimdEngine};
//! use spoars::graph::Graph;
//!
//! // Match, mismatch, gap-open, gap-extend, and a second (convex) gap-open/extend pair.
//! let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
//! let mut engine = SimdEngine::new(AlignmentType::Global, scoring);
//!
//! let reads: [&[u8]; 3] = [b"ACGTACGT", b"ACGTTCGT", b"ACGTACGT"];
//!
//! let mut graph = Graph::new();
//! for read in reads {
//!     // Align each read against the graph built so far, then merge it in.
//!     let (alignment, _score) = engine.align(read, &graph);
//!     graph.add_alignment_weight(&alignment, read, 1).unwrap();
//! }
//!
//! // The majority base at the divergent position wins the consensus.
//! assert_eq!(graph.generate_consensus(), "ACGTACGT");
//!
//! // One MSA row per read (optionally plus a consensus row).
//! let msa = graph.generate_msa(false);
//! assert_eq!(msa.len(), 3);
//! ```
//!
//! # Module map
//!
//! - [`graph`] — the arena-based POA graph: construction ([`graph::Graph::add_alignment`] and
//!   friends), consensus/MSA generation, GFA/DOT export, and read-only accessors for inspecting the
//!   built DAG.
//! - [`align`] — [`align::AlignmentType`], [`align::GapMode`], validated [`align::Scoring`], the
//!   [`align::AlignmentEngine`] trait, and the scalar/SIMD engines.

pub mod align;
pub mod graph;
