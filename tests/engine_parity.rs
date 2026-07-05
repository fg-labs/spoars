//! Differential parity tests for the SISD (scalar) alignment engine against
//! the C++ oracle.
//!
//! These assert that the Rust [`spoars::align::SisdEngine`]'s per-sequence
//! [`Alignment`](spoars::align::Alignment)s and the resulting consensus match
//! upstream spoa *byte-for-byte*, across all three alignment types
//! (SW/NW/OV). This is the single most correctness-critical parity surface in
//! the project: the DP backtrack's tie-breaking must match spoa exactly, or
//! consensus and MSA diverge on any score-tie. See
//! `third_party/spoa/src/sisd_alignment_engine.cpp:295-463` (`Linear`) for the
//! ported recurrence and backtrack.

mod support;

use proptest::prelude::*;

use support::generators::{deterministic_config, small_dna};
use support::oracle::{run_oracle, AlignType, OracleCase};
use support::runner::run_spoars;

/// Two identical sequences under linear NW: the second aligns to the first on
/// the pure diagonal (every base a match, no gap ambiguity), so the returned
/// alignment is `(node_id, seq_index)` = `(0,0),(1,1),(2,2),(3,3)` and the
/// consensus is the sequence itself. Hand-computed so a regression is
/// diagnosable without proptest shrinking.
#[test]
fn linear_nw_identical_sequences_align_on_the_diagonal() {
    let seqs = vec!["ACGT".to_string(), "ACGT".to_string()];
    let case = OracleCase::linear(AlignType::Nw, &seqs);
    let (aligns, cons) = run_spoars(&case);

    assert_eq!(aligns.len(), 2);
    // First sequence aligns against the empty graph -> empty alignment.
    assert_eq!(aligns[0], Vec::<(i32, i32)>::new());
    // Second sequence maps 1:1 onto the first's four nodes (ids 0..=3).
    assert_eq!(aligns[1], vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    assert_eq!(cons, "ACGT");
}

/// A single sequence under linear NW: aligning against the empty graph yields
/// an empty alignment, and the whole sequence is added as a new path, so the
/// consensus equals the sequence verbatim.
#[test]
fn linear_nw_single_sequence_is_added_verbatim() {
    let seqs = vec!["ACGTACGT".to_string()];
    let case = OracleCase::linear(AlignType::Nw, &seqs);
    let (aligns, cons) = run_spoars(&case);

    assert_eq!(aligns, vec![Vec::<(i32, i32)>::new()]);
    assert_eq!(cons, "ACGTACGT");
}

/// A one-mismatch pair under linear NW, hand-checked against the oracle: even
/// with a single mismatch, a linear global alignment of two equal-length
/// sequences stays on the diagonal (one mismatch costs `n = -4`, cheaper than
/// two gaps at `2 * -8`), so the second sequence still maps 1:1 onto the
/// first's nodes.
#[test]
fn linear_nw_one_mismatch_stays_on_the_diagonal() {
    let seqs = vec!["ACGT".to_string(), "ACTT".to_string()];
    let case = OracleCase::linear(AlignType::Nw, &seqs);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let (aligns, cons) = run_spoars(&case);

    assert_eq!(aligns[1], vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    assert_eq!(&aligns, &exp.alignments);
    assert_eq!(cons, exp.consensus);
}

/// An affine NW insertion run: aligning `AACCGG` against a graph built from
/// `AAGG` places the two `C` bases as a single length-2 insertion between the
/// matched `AA` and `GG` (an affine gap run of `-8 + -6 = -14` beats two
/// separate opens at `-16`), exercising the `extend_left` E-run unwind. The
/// expected trace is hand-computed and cross-checked against the oracle.
#[test]
fn affine_nw_insertion_run_walks_the_e_matrix() {
    let seqs = vec!["AAGG".to_string(), "AACCGG".to_string()];
    let case = OracleCase::affine(AlignType::Nw, &seqs);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let (aligns, cons) = run_spoars(&case);

    // node ids 0=A,1=A,2=G,3=G; the two inserted C's are seq indices 2,3.
    assert_eq!(
        aligns[1],
        vec![(0, 0), (1, 1), (-1, 2), (-1, 3), (2, 4), (3, 5)]
    );
    assert_eq!(&aligns, &exp.alignments);
    assert_eq!(cons, exp.consensus);
}

/// An affine NW deletion run: aligning `AAGG` against a graph built from
/// `AACCGG` deletes the two `C` graph nodes as a single length-2 deletion run,
/// exercising the `extend_up` F-run unwind. The expected trace is
/// hand-computed and cross-checked against the oracle.
#[test]
fn affine_nw_deletion_run_walks_the_f_matrix() {
    let seqs = vec!["AACCGG".to_string(), "AAGG".to_string()];
    let case = OracleCase::affine(AlignType::Nw, &seqs);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let (aligns, cons) = run_spoars(&case);

    // graph node ids 0=A,1=A,2=C,3=C,4=G,5=G; nodes 2,3 are deleted.
    assert_eq!(
        aligns[1],
        vec![(0, 0), (1, 1), (2, -1), (3, -1), (4, 2), (5, 3)]
    );
    assert_eq!(&aligns, &exp.alignments);
    assert_eq!(cons, exp.consensus);
}

/// A convex NW deletion run whose length (3) makes the *second* affine function
/// (`q = -10, c = -4`) cheaper than the first (`g = -8, e = -6`): a length-3 gap
/// costs `min(8 + 2*6, 10 + 2*4) = min(20, 18) = 18`, so the run is scored via
/// `O`/`Q` and its backtrack walks the two-phase `extend_up` unwind through the
/// `O` matrix. Aligning `AAGG` against a graph built from `AACCCGG` deletes the
/// three `C` graph nodes as one run. Hand-computed and cross-checked vs. the
/// oracle.
#[test]
fn convex_nw_long_deletion_run_uses_the_second_gap_function() {
    let seqs = vec!["AACCCGG".to_string(), "AAGG".to_string()];
    let case = OracleCase::convex(AlignType::Nw, &seqs);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let (aligns, cons) = run_spoars(&case);

    // graph node ids 0=A,1=A,2=C,3=C,4=C,5=G,6=G; nodes 2,3,4 are deleted.
    assert_eq!(
        aligns[1],
        vec![(0, 0), (1, 1), (2, -1), (3, -1), (4, -1), (5, 2), (6, 3)]
    );
    assert_eq!(&aligns, &exp.alignments);
    assert_eq!(cons, exp.consensus);
}

/// A convex NW insertion run, the sequence-axis mirror of
/// `convex_nw_long_deletion_run_uses_the_second_gap_function`: aligning
/// `AACCCGG` against a graph built from `AAGG` inserts the three `C` bases as one
/// length-3 run, which the convex model again scores via the cheaper second gap
/// function (`Q` matrix) and whose backtrack walks the `extend_left` `E`/`Q` run
/// unwind. Hand-computed and cross-checked vs. the oracle.
#[test]
fn convex_nw_long_insertion_run_uses_the_second_gap_function() {
    let seqs = vec!["AAGG".to_string(), "AACCCGG".to_string()];
    let case = OracleCase::convex(AlignType::Nw, &seqs);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let (aligns, cons) = run_spoars(&case);

    // graph node ids 0=A,1=A,2=G,3=G; the three inserted C's are seq indices 2,3,4.
    assert_eq!(
        aligns[1],
        vec![(0, 0), (1, 1), (-1, 2), (-1, 3), (-1, 4), (2, 5), (3, 6)]
    );
    assert_eq!(&aligns, &exp.alignments);
    assert_eq!(cons, exp.consensus);
}

/// An even longer (length-5) convex deletion run, locking the `O`/`Q` second
/// gap function on a run long enough that the first function is decisively worse
/// (`min(8 + 4*6, 10 + 4*4) = min(32, 26) = 26`). Cross-checked vs. the oracle.
#[test]
fn convex_nw_very_long_deletion_run_walks_the_o_matrix() {
    let seqs = vec!["AACCCCCGG".to_string(), "AAGG".to_string()];
    let case = OracleCase::convex(AlignType::Nw, &seqs);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let (aligns, cons) = run_spoars(&case);

    // graph node ids 0=A,1=A,2..=6=C,7=G,8=G; nodes 2..=6 are deleted.
    assert_eq!(
        aligns[1],
        vec![
            (0, 0),
            (1, 1),
            (2, -1),
            (3, -1),
            (4, -1),
            (5, -1),
            (6, -1),
            (7, 2),
            (8, 3)
        ]
    );
    assert_eq!(&aligns, &exp.alignments);
    assert_eq!(cons, exp.consensus);
}

proptest! {
    // Each generated input runs three oracle round-trips (SW/NW/OV), so cap the
    // case count to keep the suite's wall time reasonable while still fuzzing
    // the tie-break paths broadly.
    #![proptest_config(ProptestConfig { cases: 48, ..deterministic_config() })]

    /// The core differential fuzzer: for one randomly generated small-DNA input,
    /// aligns it through the Rust engine (per-sequence `Align` -> `AddAlignment`)
    /// and asserts both the per-sequence alignment traces and the final consensus
    /// match the C++ oracle exactly, across all three alignment types. A tie-break
    /// divergence in the backtrack is precisely what this is designed to catch;
    /// `prop_assert_eq!` shrinks any failure to a minimal reproducing input.
    #[test]
    fn linear_alignment_matches_oracle(seqs in small_dna(40, 6)) {
        for ty in [AlignType::Sw, AlignType::Nw, AlignType::Ov] {
            let case = OracleCase::linear(ty, &seqs);
            let exp = &run_oracle(std::slice::from_ref(&case))[0];
            let (aligns, cons) = run_spoars(&case);
            prop_assert_eq!(
                &aligns, &exp.alignments,
                "alignment mismatch ty={:?} seqs={:?}", ty, seqs
            );
            prop_assert_eq!(
                &cons, &exp.consensus,
                "consensus mismatch ty={:?} seqs={:?}", ty, seqs
            );
        }
    }

    /// The affine-gap counterpart of `linear_alignment_matches_oracle`: same
    /// per-sequence alignment + consensus parity assertions, but with
    /// `OracleCase::affine` (`g = -8, e = -6`, classified `kAffine`), which
    /// exercises the separate `E`/`F` matrices and the `extend_left`/`extend_up`
    /// gap-run unwinding in the backtrack.
    #[test]
    fn affine_alignment_matches_oracle(seqs in small_dna(40, 6)) {
        for ty in [AlignType::Sw, AlignType::Nw, AlignType::Ov] {
            let case = OracleCase::affine(ty, &seqs);
            let exp = &run_oracle(std::slice::from_ref(&case))[0];
            let (aligns, cons) = run_spoars(&case);
            prop_assert_eq!(
                &aligns, &exp.alignments,
                "alignment mismatch ty={:?} seqs={:?}", ty, seqs
            );
            prop_assert_eq!(
                &cons, &exp.consensus,
                "consensus mismatch ty={:?} seqs={:?}", ty, seqs
            );
        }
    }

    /// The convex-gap counterpart of `linear_alignment_matches_oracle` /
    /// `affine_alignment_matches_oracle`: same per-sequence alignment + consensus
    /// parity assertions, but with `OracleCase::convex` (spoa's CLI defaults
    /// `g = -8, e = -6, q = -10, c = -4`, classified `kConvex`), which exercises
    /// the second affine function's `O`/`Q` matrices and the compound `extend_*`
    /// conditions plus the two-phase `extend_up` unwind in the backtrack.
    #[test]
    fn convex_alignment_matches_oracle(seqs in small_dna(40, 6)) {
        for ty in [AlignType::Sw, AlignType::Nw, AlignType::Ov] {
            let case = OracleCase::convex(ty, &seqs);
            let exp = &run_oracle(std::slice::from_ref(&case))[0];
            let (aligns, cons) = run_spoars(&case);
            prop_assert_eq!(
                &aligns, &exp.alignments,
                "alignment mismatch ty={:?} seqs={:?}", ty, seqs
            );
            prop_assert_eq!(
                &cons, &exp.consensus,
                "consensus mismatch ty={:?} seqs={:?}", ty, seqs
            );
        }
    }
}

proptest! {
    // The capstone full-parity sweep: for one generated input, exercise EVERY
    // scalar-engine combination — {linear, affine, convex} x {SW, NW, OV} — in a
    // single case, asserting alignment + consensus parity against the oracle for
    // all nine. This is 9 oracle round-trips per case, so keep the case count
    // modest; it is the whole-engine parity lock, complementing the per-mode
    // fuzzers above.
    #![proptest_config(ProptestConfig { cases: 32, ..deterministic_config() })]

    #[test]
    fn all_gap_modes_match_oracle(seqs in small_dna(40, 6)) {
        // Each gap mode paired with its OracleCase constructor, so the sweep
        // stays exhaustive over the {mode} x {type} product.
        type CaseBuilder = fn(AlignType, &[String]) -> OracleCase;
        let builders: [(&str, CaseBuilder); 3] = [
            ("linear", OracleCase::linear),
            ("affine", OracleCase::affine),
            ("convex", OracleCase::convex),
        ];
        for (mode, build) in builders {
            for ty in [AlignType::Sw, AlignType::Nw, AlignType::Ov] {
                let case = build(ty, &seqs);
                let exp = &run_oracle(std::slice::from_ref(&case))[0];
                let (aligns, cons) = run_spoars(&case);
                prop_assert_eq!(
                    &aligns, &exp.alignments,
                    "alignment mismatch mode={} ty={:?} seqs={:?}", mode, ty, seqs
                );
                prop_assert_eq!(
                    &cons, &exp.consensus,
                    "consensus mismatch mode={} ty={:?} seqs={:?}", mode, ty, seqs
                );
            }
        }
    }
}
