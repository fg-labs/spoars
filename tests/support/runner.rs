//! End-to-end driver that replays an [`OracleCase`] through the *Rust* engine
//! and graph, mirroring the exact call sequence upstream spoa's CLI performs
//! (`third_party/spoa/src/main.cpp:280-290` + consensus) and the oracle
//! reproduces (`oracle/spoa_oracle.cpp:544-564`): create a `Graph`, build a
//! [`SisdEngine`] for the case's alignment type and scoring, then for each
//! sequence align it against the graph-so-far and fold the returned alignment
//! back into the graph. Finally, generate the consensus.
//!
//! This is the Rust-native counterpart to [`crate::support::oracle::run_oracle`],
//! letting the engine-parity tests assert the Rust aligner's per-sequence
//! [`Alignment`]s and the resulting consensus against the C++ oracle
//! byte-for-byte.

use spoars::align::{Alignment, AlignmentEngine, AlignmentType, Scoring, SisdEngine};
use spoars::graph::Graph;

use super::oracle::{AlignType, OracleCase};

/// Maps the oracle's [`AlignType`] JSONL enum onto the Rust engine's
/// [`AlignmentType`].
fn alignment_type_from(ty: AlignType) -> AlignmentType {
    match ty {
        AlignType::Sw => AlignmentType::Local,
        AlignType::Nw => AlignmentType::Global,
        AlignType::Ov => AlignmentType::Overlap,
    }
}

/// Aligns each of `case`'s sequences against a growing [`Graph`] with a
/// [`SisdEngine`], folding each alignment back into the graph, then generates
/// the consensus. Returns the per-sequence [`Alignment`]s (in input order) and
/// the final consensus string.
///
/// Mirrors `oracle/spoa_oracle.cpp:544-564`: same scoring, same alignment type,
/// same per-sequence `Align` -> `AddAlignment` loop, same `GenerateConsensus`
/// (honoring `case.min_coverage`, and routing through the Phred-weighted
/// `add_alignment_quality` whenever `case.quals` is present).
pub fn run_spoars(case: &OracleCase) -> (Vec<Alignment>, String) {
    let scoring = Scoring::new(case.m, case.n, case.g, case.e, case.q, case.c)
        .expect("run_spoars: Scoring::new rejected the oracle case's penalties");
    let mut engine = SisdEngine::new(alignment_type_from(case.ty), scoring);

    let mut graph = Graph::new();
    let mut alignments = Vec::with_capacity(case.seqs.len());
    for (i, seq) in case.seqs.iter().enumerate() {
        let bytes = seq.as_bytes();
        let (alignment, _score) = engine.align(bytes, &graph);
        alignments.push(alignment.clone());

        if let Some(quals) = &case.quals {
            graph
                .add_alignment_quality(&alignment, bytes, quals[i].as_bytes())
                .unwrap_or_else(|e| {
                    panic!("run_spoars: add_alignment_quality failed (seq {i}): {e}")
                });
        } else {
            graph
                .add_alignment_weight(&alignment, bytes, 1)
                .unwrap_or_else(|e| {
                    panic!("run_spoars: add_alignment_weight failed (seq {i}): {e}")
                });
        }
    }

    let consensus = graph.generate_consensus_min_coverage(case.min_coverage);
    (alignments, consensus)
}
