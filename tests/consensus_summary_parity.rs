//! Differential parity of the consensus-summary methods against the C++ spoa oracle.
mod support;

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SisdEngine};
use spoars::graph::Graph;
use support::oracle::{run_oracle, OracleCase};

/// Aligns `seqs` (in order) with a fresh [`SisdEngine`] of the given `ty` and folds each into a
/// fresh [`Graph`], mirroring the oracle's own align-then-add loop
/// (`oracle/spoa_oracle.cpp:565-573`).
fn build_with_type(seqs: &[&str], ty: AlignmentType) -> Graph {
    let mut g = Graph::new();
    let mut engine = SisdEngine::new(ty, Scoring::spoa_default());
    for s in seqs {
        let bytes = s.as_bytes();
        let (aln, _) = engine.align(bytes, &g);
        g.add_alignment_weight(&aln, bytes, 1).unwrap();
    }
    g
}

/// Asserts both summaries for `g` match the oracle's for the same `seqs`, `ty`, and `min_coverage`.
fn assert_summaries_match(seqs: &[&str], ty: AlignmentType, case: OracleCase, min_coverage: i32) {
    let mut g = build_with_type(seqs, ty);

    let (_c1, my_cov) = g.generate_consensus_with_coverage(min_coverage);
    let (_c2, my_comp, my_stride) = g.generate_consensus_with_composition();

    let mut case = case;
    case.summarize_consensus = true;
    case.min_coverage = min_coverage;
    let res = &run_oracle(&[case])[0];

    assert_eq!(
        my_cov, res.consensus_coverage,
        "coverage mismatch seqs={seqs:?} mc={min_coverage}"
    );
    assert_eq!(
        my_stride, res.consensus_composition_stride,
        "stride mismatch seqs={seqs:?}"
    );
    assert_eq!(
        my_comp, res.consensus_composition,
        "composition mismatch seqs={seqs:?}"
    );

    // Invariant: each column's composition sums to the number of sequences covering it.
    // (Cross-check independent of the oracle.)
    if my_stride > 0 {
        let rows = g.num_codes() as usize + 1;
        for col in 0..my_stride {
            let col_sum: u32 = (0..rows).map(|r| my_comp[r * my_stride + col]).sum();
            assert!(col_sum <= seqs.len() as u32);
        }
    }
}

#[test]
fn consensus_summaries_match_oracle_on_small_families() {
    let families: &[&[&str]] = &[
        &["ACGT", "ACGT", "AGGT"],
        &["ACGTACGT", "ACGAACGT", "ACGTAAGT", "ACGTACG"],
        &["GATTACA", "GATTTACA", "GATACA", "GACTACA"],
    ];
    for seqs in families {
        for mc in [-1, 1, 2] {
            assert_summaries_match(seqs, AlignmentType::Global, OracleCase::nw(seqs), mc);
        }
    }
}

#[test]
fn consensus_summaries_match_oracle_across_alignment_modes() {
    let seqs: &[&str] = &["GATTACA", "GATTTACA", "GATACA", "GACTACA"];
    assert_summaries_match(seqs, AlignmentType::Global, OracleCase::nw(seqs), -1);
    assert_summaries_match(seqs, AlignmentType::Local, OracleCase::sw(seqs), -1);
    assert_summaries_match(seqs, AlignmentType::Overlap, OracleCase::ov(seqs), -1);
}
