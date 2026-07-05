//! Differential parity tests for the SIMD (vectorized) alignment engine against the scalar
//! [`SisdEngine`] oracle — the int16 + int32 **linear-gap** fill (SIMD kernels plan Tasks 7-8, 11),
//! **affine-gap** fill (Tasks 9a-9b), and **convex-gap** fill (Tasks 10a-10b), all across all
//! three [`AlignmentType`]s (Global/NW, Local/SW, Overlap/OV) — the full engine (all 9 gap-mode x
//! alignment-type combinations) on whichever ISA `SimdEngine`'s runtime dispatch selects: SSE4.1
//! on x86_64, and **NEON on aarch64** (SIMD kernels plan Task 12).
//!
//! Per the SIMD kernels plan's Global Constraints, bit-exactness against `SisdEngine` is the
//! acceptance test: for every `(sequence, graph)` a SIMD kernel must return the identical
//! per-sequence [`Alignment`] AND `score`, and (folded into a growing POA graph) yield the
//! identical consensus. `SisdEngine` was itself certified against the C++ oracle in the earlier
//! milestones, so it is a sound in-process oracle and no C++ is needed in this loop.
//!
//! # ISA gating (process constraint)
//!
//! The SIMD-path assertions are gated on [`simd_kernel_active`], which is true whenever the running
//! CPU exposes an ISA `SimdEngine` actually vectorizes on: SSE4.1 on x86_64 (including Rosetta 2 on
//! Apple Silicon, via `cargo test --target x86_64-apple-darwin`) and NEON on **native aarch64**.
//! So on this Apple-Silicon / Graviton hardware these tests EXECUTE the real NEON kernels natively
//! (no Rosetta) — the first time the parity sweep runs on real target silicon rather than emulated
//! x86. On a target with no vectorized ISA the gate is `false` and the assertions no-op (green).

mod support;

use proptest::collection::vec;
use proptest::prelude::*;

use spoars::align::{Alignment, AlignmentEngine, AlignmentType, Scoring, SimdEngine, SisdEngine};
use spoars::graph::Graph;

use support::generators::{deterministic_config, small_dna};

/// Whether the running CPU exposes an ISA that [`SimdEngine`] actually vectorizes on — SSE4.1 on
/// x86_64 (true under Rosetta 2 on Apple Silicon) or NEON on aarch64 (native Apple Silicon /
/// Graviton). On any other target the relevant `is_*_feature_detected!` macro does not exist, so
/// this compiles to a constant `false` and the parity assertions no-op.
fn simd_kernel_active() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse4.1")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// spoa's CLI-default match/mismatch scores with a **linear** gap penalty (`g == e`, so
/// [`Scoring::gap_mode`] classifies it `Linear`) — the exact `OracleCase::linear` parameters.
fn linear_scoring() -> Scoring {
    Scoring::new(5, -4, -8, -8, -8, -8).unwrap()
}

/// An **affine** gap penalty (`g < e` and `g <= q`, so [`Scoring::gap_mode`] classifies it
/// `Affine`) — the exact `OracleCase::affine` parameters (`g = -8`, `e = -6`, `q = -8`, `c = -6`).
fn affine_scoring() -> Scoring {
    Scoring::new(5, -4, -8, -6, -8, -6).unwrap()
}

/// A **convex** gap penalty with two distinct affine functions — the exact `OracleCase::convex`
/// parameters (`g = -8`, `e = -6`, `q = -10`, `c = -4`), classified [`Scoring::gap_mode`]
/// `Convex`. The first function (`g`/`e`) is the cheaper OPEN with a steep extend; the second
/// (`q`/`c`) is a pricier open with a shallow extend, so it wins for long gaps: at length `L` the
/// per-gap cost is `max(-8 - 6*(L-1), -10 - 4*(L-1))`, and the second term overtakes the first for
/// `L >= 3`.
fn convex_scoring() -> Scoring {
    Scoring::new(5, -4, -8, -6, -10, -4).unwrap()
}

// ---- large-penalty scorings + generator (force the SSE4.1 int32 kernel) ------------------------
//
// `SimdEngine::align`'s private `escalate` (`simd/mod.rs`) selects `Escalation::Int32` whenever
// `Scoring::worst_case_alignment_score(seq_len as i64 + 8, node_count as i64) < i16::MIN + 1024`.
// The `_large` scorings below (near-`i8::MIN` gap penalties, one per gap mode) paired with
// [`large_dna`]'s length range are chosen so that bound is crossed for EVERY length combination
// the generator can draw, across all three gap modes — see
// `simd_int32_escalation_is_actually_forced_at_the_generators_tightest_margin` for the check that
// proves it (using the exact same public formula `escalate` uses internally, since `escalate`
// itself is private to `simd/mod.rs` and unreachable from this integration test).

/// Large-penalty **linear** scoring (`g == e`, both near `i8::MIN`), paired with [`large_dna`].
fn linear_scoring_large() -> Scoring {
    Scoring::new(127, -128, -128, -128, -128, -128).unwrap()
}

/// Large-penalty **affine** scoring (`g < e`; [`Scoring::new`]'s normalization then sets `q = g`,
/// `c = e`), paired with [`large_dna`].
fn affine_scoring_large() -> Scoring {
    Scoring::new(127, -128, -128, -100, -128, -100).unwrap()
}

/// Large-penalty **convex** scoring (two distinct affine functions: `g > q` and `e < c`, so
/// neither of `Scoring::classify`'s `Affine` conditions holds), paired with [`large_dna`].
fn convex_scoring_large() -> Scoring {
    Scoring::new(127, -128, -110, -90, -128, -80).unwrap()
}

/// The length range [`large_dna`] draws from for the int32-forcing tests below: chosen (see
/// `simd_int32_escalation_is_actually_forced_at_the_generators_tightest_margin`) so that even the
/// tightest-margin combination (`LARGE_MIN_LEN` on both sides) crosses the int16->int32 escalation
/// boundary for all three `_large` scorings above.
const LARGE_MIN_LEN: usize = 190;
/// See [`LARGE_MIN_LEN`].
const LARGE_MAX_LEN: usize = 230;

/// A proptest strategy generating exactly `n_seqs` plain-ACGT sequences, each of length
/// `min_len..=max_len`. Deliberately NOT [`small_dna`] (whose 1-based minimum length can generate
/// sequences far too short to force the int32 escalation tier): every length in
/// `min_len..=max_len` must provably force `Escalation::Int32` when paired with a `_large` scoring,
/// which requires a floor well above 1.
fn large_dna(min_len: usize, max_len: usize, n_seqs: usize) -> impl Strategy<Value = Vec<String>> {
    let seq = vec(
        prop_oneof![Just('A'), Just('C'), Just('G'), Just('T')],
        min_len..=max_len,
    )
    .prop_map(|chars| chars.into_iter().collect::<String>());
    vec(seq, n_seqs..=n_seqs)
}

/// Asserts, via the SAME public formula `SimdEngine`'s private `escalate` uses internally
/// (`Scoring::worst_case_alignment_score(seq_len as i64 + 8, node_count as i64)`, per that
/// function's doc in `simd/mod.rs`), that aligning a `seq_len`-long sequence against a
/// `node_count`-node graph under `scoring` provably selects `Escalation::Int32`: the worst case
/// must be strictly below `i16::MIN + 1024` (the int16->int32 boundary) and at or above
/// `i32::MIN + 1024` (the int32->fallback boundary, so this doesn't accidentally exercise the
/// fallback tier instead of int32). `escalate` itself is private to `simd/mod.rs`, so this
/// integration test cannot call it directly; replaying its exact, documented formula through the
/// public `Scoring::worst_case_alignment_score` is the closest available proof, from outside the
/// crate, that the int32 kernel (not a delegation) is what actually runs.
fn assert_forces_int32_escalation(scoring: Scoring, seq_len: usize, node_count: usize) {
    let worst_case = scoring.worst_case_alignment_score(seq_len as i64 + 8, node_count as i64);
    assert!(
        worst_case < i64::from(i16::MIN) + 1024,
        "expected the int16->int32 escalation boundary to be crossed: worst_case={worst_case} \
         seq_len={seq_len} node_count={node_count}"
    );
    assert!(
        worst_case >= i64::from(i32::MIN) + 1024,
        "expected NOT to cross into the fallback tier: worst_case={worst_case} \
         seq_len={seq_len} node_count={node_count}"
    );
}

/// Proves [`large_dna`]'s TIGHTEST margin (`LARGE_MIN_LEN` on both the sequence and the graph's
/// node count — the smallest worst-case magnitude the generator can produce) still crosses the
/// int16->int32 escalation boundary, for all three `_large` scorings. Since
/// `Scoring::worst_case_alignment_score`'s gap term only grows more negative as lengths grow, this
/// is the binding case: every other length combination [`large_dna`] can draw is strictly further
/// into the int32 tier. This is a plain `#[test]` (not a proptest) because it checks a fixed
/// boundary value, not a randomized input.
#[test]
fn simd_int32_escalation_is_actually_forced_at_the_generators_tightest_margin() {
    for (scoring, mode) in [
        (linear_scoring_large(), "linear"),
        (affine_scoring_large(), "affine"),
        (convex_scoring_large(), "convex"),
    ] {
        assert_forces_int32_escalation(scoring, LARGE_MIN_LEN, LARGE_MIN_LEN);
        eprintln!("simd_parity: {mode} large-penalty scoring provably forces Escalation::Int32");
    }
}

/// Replays the `run_spoars` alignment loop through an arbitrary [`AlignmentEngine`]: align each
/// sequence against the graph-so-far, fold the returned alignment back in, and finally generate
/// the consensus. Returns the per-sequence alignments, their scores, and the consensus.
fn drive<E: AlignmentEngine>(
    engine: &mut E,
    seqs: &[String],
) -> (Vec<Alignment>, Vec<i32>, String) {
    let mut graph = Graph::new();
    let mut alignments = Vec::with_capacity(seqs.len());
    let mut scores = Vec::with_capacity(seqs.len());
    for seq in seqs {
        let bytes = seq.as_bytes();
        let (alignment, score) = engine.align(bytes, &graph);
        alignments.push(alignment.clone());
        scores.push(score);
        graph
            .add_alignment_weight(&alignment, bytes, 1)
            .expect("add_alignment_weight failed");
    }
    let consensus = graph.generate_consensus_min_coverage(-1);
    (alignments, scores, consensus)
}

/// Drives `seqs` through both a [`SimdEngine`] and a [`SisdEngine`] built with identical params
/// and returns `(simd_result, sisd_result)`, each `(alignments, scores, consensus)`.
#[allow(clippy::type_complexity)]
fn run_both(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seqs: &[String],
) -> (
    (Vec<Alignment>, Vec<i32>, String),
    (Vec<Alignment>, Vec<i32>, String),
) {
    let mut simd = SimdEngine::new(alignment_type, scoring);
    let mut sisd = SisdEngine::new(alignment_type, scoring);
    (drive(&mut simd, seqs), drive(&mut sisd, seqs))
}

// ---- hand cases (exact expected alignment) ------------------------------------------------------

/// Two identical sequences under linear NW: the second aligns on the pure diagonal, so the SIMD
/// engine must return the exact `(0,0),(1,1),(2,2),(3,3)` trace, matching both the hand-computed
/// expectation and the scalar engine.
#[test]
fn simd_linear_nw_identical_sequences_align_on_the_diagonal() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGT".to_string(), "ACGT".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, linear_scoring(), &seqs);

    assert_eq!(simd.0[0], Vec::<(i32, i32)>::new());
    assert_eq!(simd.0[1], vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    assert_eq!(simd.2, "ACGT");
    assert_eq!(
        simd, sisd,
        "SIMD must match SISD (alignments, scores, consensus)"
    );
}

/// A one-mismatch pair under linear NW: a single mismatch (`n = -4`) is cheaper than two gaps
/// (`2 * -8`), so the alignment stays on the diagonal. The SIMD trace must match the hand
/// expectation and the scalar engine exactly.
#[test]
fn simd_linear_nw_one_mismatch_stays_on_the_diagonal() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGT".to_string(), "ACTT".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, linear_scoring(), &seqs);

    assert_eq!(simd.0[1], vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    assert_eq!(
        simd, sisd,
        "SIMD must match SISD (alignments, scores, consensus)"
    );
}

/// A single, longer sequence spanning more than one 8-lane int16 segment (length 10 > 8),
/// exercising the inter-segment `x` carry across a segment boundary. Aligned against the empty
/// graph it yields an empty alignment and is added verbatim, so the consensus is the sequence.
#[test]
fn simd_linear_nw_multi_segment_single_sequence_added_verbatim() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGTACGTAC".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, linear_scoring(), &seqs);

    assert_eq!(simd.0, vec![Vec::<(i32, i32)>::new()]);
    assert_eq!(simd.2, "ACGTACGTAC");
    assert_eq!(
        simd, sisd,
        "SIMD must match SISD (alignments, scores, consensus)"
    );
}

/// Logs the ISA gate's decision once, so a run on a SIMD-capable host (native aarch64 NEON, or
/// x86_64/Rosetta SSE4.1) visibly reports that the vectorized path executed, and a SIMD-less host
/// visibly reports the no-op.
#[test]
fn simd_parity_reports_simd_availability() {
    if simd_kernel_active() {
        eprintln!("simd_parity: vectorized SIMD ISA active — parity assertions ACTIVE (native)");
    } else {
        eprintln!("simd_parity: no vectorized SIMD ISA — parity assertions SKIPPED");
    }
}

// ---- affine hand cases (gap runs) ---------------------------------------------------------------

/// A multi-base DELETION run under affine NW: the graph is built from a sequence carrying an
/// internal `TTTTT` block the second sequence lacks, so aligning the second forces a gap RUN along
/// the sequence axis (exercising the shared backtrack's `extend_left`/`extend_up` unwinds, which
/// only fire on runs of length >= 2). The SIMD affine fill must destripe H/E/F so the run
/// backtracks identically to the scalar engine.
#[test]
fn simd_affine_nw_deletion_run_matches_sisd() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGTTTTTACGT".to_string(), "ACGTACGT".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, affine_scoring(), &seqs);
    assert_eq!(
        simd, sisd,
        "SIMD affine NW must match SISD (alignments, scores, consensus)"
    );
}

/// A multi-base INSERTION run under affine NW: the mirror of the deletion case, the second
/// sequence carrying an internal block the graph lacks, forcing a gap RUN the other direction.
#[test]
fn simd_affine_nw_insertion_run_matches_sisd() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGTACGT".to_string(), "ACGTTTTTACGT".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, affine_scoring(), &seqs);
    assert_eq!(
        simd, sisd,
        "SIMD affine NW must match SISD (alignments, scores, consensus)"
    );
}

/// A multi-segment (length > 8, i.e. more than one 8-lane int16 stripe) affine NW gap-run case,
/// exercising the inter-segment `x` carry through both the F/diagonal fold and the E prefix-max.
#[test]
fn simd_affine_nw_multi_segment_gap_run_matches_sisd() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec![
        "ACGTACGTAAAAACGTACGT".to_string(),
        "ACGTACGTACGTACGT".to_string(),
    ];
    let (simd, sisd) = run_both(AlignmentType::Global, affine_scoring(), &seqs);
    assert_eq!(
        simd, sisd,
        "SIMD affine NW must match SISD (alignments, scores, consensus)"
    );
}

proptest! {
    // Affine NW is a single engine run per case (no C++ oracle round-trip), so the case count can
    // stay modest while broadly fuzzing the F (vertical), E (prefix-max horizontal) and H=max(H,E,F)
    // paths plus the inter-segment carry and gap-run backtracks.
    #![proptest_config(ProptestConfig { cases: 48, ..deterministic_config() })]

    /// The affine parity fuzzer: for one randomly generated small-DNA input, align it through both
    /// the SIMD engine (forced onto the SSE4.1 int16 affine path by the affine scoring + small
    /// size) and the scalar engine, for EACH of the three [`AlignmentType`]s (Global/NW — SIMD
    /// kernels plan Task 9a; Local/SW + Overlap/OV — Task 9b), asserting the per-sequence
    /// alignments, scores, and consensus are all identical. Gated on [`simd_kernel_active`] so native
    /// arm64 is a no-op and Rosetta/x86 is the real test.
    #[test]
    fn simd_affine_matches_sisd(seqs in small_dna(40, 6)) {
        if simd_kernel_active() {
            for alignment_type in [
                AlignmentType::Global,
                AlignmentType::Local,
                AlignmentType::Overlap,
            ] {
                let (simd, sisd) = run_both(alignment_type, affine_scoring(), &seqs);
                prop_assert_eq!(
                    &simd.0, &sisd.0,
                    "alignment mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
                prop_assert_eq!(
                    &simd.1, &sisd.1,
                    "score mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
                prop_assert_eq!(
                    &simd.2, &sisd.2,
                    "consensus mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
            }
        }
    }
}

// ---- convex hand cases (long gaps forcing the 2nd affine function) ------------------------------

/// A LONG deletion run under convex NW: the graph carries an internal `TTTTTTTT` block (length 8)
/// the second sequence lacks, forcing an 8-long gap along the sequence axis. At that length the
/// second convex function (`q = -10`, `c = -4`) is strictly cheaper than the first (`g = -8`,
/// `e = -6`) — `-10 - 4*7 = -38` vs `-8 - 6*7 = -50` — so the `O`/`Q` (second-affine) ladder MUST
/// win the gap for the SIMD fill to match the scalar engine.
#[test]
fn simd_convex_nw_long_deletion_run_uses_second_function() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGTTTTTTTTACGT".to_string(), "ACGTACGT".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, convex_scoring(), &seqs);
    assert_eq!(
        simd, sisd,
        "SIMD convex NW must match SISD (alignments, scores, consensus)"
    );
}

/// The mirror LONG insertion run under convex NW: the second sequence carries an internal
/// `TTTTTTTT` block the graph lacks, forcing a long gap the other direction (the `E`/`Q` horizontal
/// ladder, whose second `Q` prefix-max wins for long runs).
#[test]
fn simd_convex_nw_long_insertion_run_uses_second_function() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec!["ACGTACGT".to_string(), "ACGTTTTTTTTACGT".to_string()];
    let (simd, sisd) = run_both(AlignmentType::Global, convex_scoring(), &seqs);
    assert_eq!(
        simd, sisd,
        "SIMD convex NW must match SISD (alignments, scores, consensus)"
    );
}

/// A multi-segment (length > 8, i.e. more than one 8-lane int16 stripe) convex NW long-gap case,
/// exercising BOTH inter-segment carries (`x` for the `E` ladder and `y` for the `Q` ladder) across
/// a segment boundary while the second affine function is active over the long gap run.
#[test]
fn simd_convex_nw_multi_segment_long_gap_matches_sisd() {
    if !simd_kernel_active() {
        eprintln!("simd_parity: skipping (no vectorized SIMD ISA active on this target/host)");
        return;
    }
    let seqs = vec![
        "ACGTACGTAAAAAAAACGTACGT".to_string(),
        "ACGTACGTACGTACGT".to_string(),
    ];
    let (simd, sisd) = run_both(AlignmentType::Global, convex_scoring(), &seqs);
    assert_eq!(
        simd, sisd,
        "SIMD convex NW must match SISD (alignments, scores, consensus)"
    );
}

proptest! {
    // Convex NW is a single engine run per case (no C++ oracle round-trip), so the case count can
    // stay modest while broadly fuzzing the DUAL affine pairs (F/E first function + O/Q second
    // function), the two prefix-max ladders, the dual x/y carries, and the 4-way H max.
    #![proptest_config(ProptestConfig { cases: 48, ..deterministic_config() })]

    /// The convex parity fuzzer: for one randomly generated small-DNA input, align it through both
    /// the SIMD engine (forced onto the SSE4.1 int16 convex path by the convex scoring + small
    /// size) and the scalar engine, for EACH of the three [`AlignmentType`]s (Global/NW — SIMD
    /// kernels plan Task 10a; Local/SW + Overlap/OV — Task 10b), asserting the per-sequence
    /// alignments, scores, and consensus are all identical. Gated on [`simd_kernel_active`] so native
    /// arm64 is a no-op and Rosetta/x86 is the real test.
    #[test]
    fn simd_convex_matches_sisd(seqs in small_dna(40, 6)) {
        if simd_kernel_active() {
            for alignment_type in [
                AlignmentType::Global,
                AlignmentType::Local,
                AlignmentType::Overlap,
            ] {
                let (simd, sisd) = run_both(alignment_type, convex_scoring(), &seqs);
                prop_assert_eq!(
                    &simd.0, &sisd.0,
                    "alignment mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
                prop_assert_eq!(
                    &simd.1, &sisd.1,
                    "score mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
                prop_assert_eq!(
                    &simd.2, &sisd.2,
                    "consensus mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
            }
        }
    }
}

proptest! {
    // Linear is a single engine run per case per type (no C++ oracle round-trip), so the case
    // count can stay modest while still fuzzing the fill's diagonal/vertical/prefix-max/carry
    // paths (and, for SW/OV, the clamp-to-0 / sink-node max-tracking) broadly.
    #![proptest_config(ProptestConfig { cases: 48, ..deterministic_config() })]

    /// The core parity fuzzer: for one randomly generated small-DNA input, align it through both
    /// the SIMD engine (forced onto the SSE4.1 int16 linear path by the linear scoring + small
    /// size selecting int16) and the scalar engine, for EACH of the three [`AlignmentType`]s
    /// (Global/NW, Local/SW, Overlap/OV), and assert the per-sequence alignments, their scores,
    /// and the final consensus are all identical. Gated on [`simd_kernel_active`]: on a host without
    /// SSE4.1 the case body does nothing (vacuously passes), making native arm64 a no-op and
    /// Rosetta/x86 the real test.
    #[test]
    fn simd_linear_matches_sisd(seqs in small_dna(40, 6)) {
        if simd_kernel_active() {
            for alignment_type in [
                AlignmentType::Global,
                AlignmentType::Local,
                AlignmentType::Overlap,
            ] {
                let (simd, sisd) = run_both(alignment_type, linear_scoring(), &seqs);
                prop_assert_eq!(
                    &simd.0, &sisd.0,
                    "alignment mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
                prop_assert_eq!(
                    &simd.1, &sisd.1,
                    "score mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
                prop_assert_eq!(
                    &simd.2, &sisd.2,
                    "consensus mismatch type={:?} seqs={:?}", alignment_type, seqs
                );
            }
        }
    }
}

// ---- combined 9-combo capstone sweep (mirrors `engine_parity.rs`'s scalar
// `all_gap_modes_match_oracle`) — completes the full SSE4.1 engine (int16 + int32, all 9
// gap-mode x alignment-type combinations) ---------------------------------------------------------

/// A `Scoring`-builder function pointer, so the capstone sweeps below stay exhaustive over the
/// `{gap mode} x {alignment type}` product with the mode name available for failure messages.
type ScoringBuilder = fn() -> Scoring;

proptest! {
    // A single generated input already drives all 9 (mode x type) combos per case, so the case
    // count can stay modest while still covering the full product broadly.
    #![proptest_config(ProptestConfig { cases: 24, ..deterministic_config() })]

    /// The SSE4.1 capstone sweep, **int16** tier: for ONE randomly generated small-DNA input,
    /// sweeps ALL 9 (gap mode x alignment type) combinations — linear/affine/convex x
    /// Global/Local/Overlap — asserting `SimdEngine` (SSE4.1, ordinary scoring, so the int16
    /// kernel) matches `SisdEngine` on every combo's alignments, scores, and consensus. Mirrors
    /// `engine_parity.rs`'s scalar `all_gap_modes_match_oracle` capstone, one level up (SIMD vs
    /// SISD rather than SISD vs the C++ oracle). Gated on [`simd_kernel_active`].
    #[test]
    fn simd_capstone_all_gap_modes_and_types_match_sisd_int16(seqs in small_dna(40, 6)) {
        if simd_kernel_active() {
            let builders: [(&str, ScoringBuilder); 3] = [
                ("linear", linear_scoring),
                ("affine", affine_scoring),
                ("convex", convex_scoring),
            ];
            for (mode, build) in builders {
                let scoring = build();
                for alignment_type in [
                    AlignmentType::Global,
                    AlignmentType::Local,
                    AlignmentType::Overlap,
                ] {
                    let (simd, sisd) = run_both(alignment_type, scoring, &seqs);
                    prop_assert_eq!(
                        &simd.0, &sisd.0,
                        "alignment mismatch mode={} type={:?} seqs={:?}", mode, alignment_type, seqs
                    );
                    prop_assert_eq!(
                        &simd.1, &sisd.1,
                        "score mismatch mode={} type={:?} seqs={:?}", mode, alignment_type, seqs
                    );
                    prop_assert_eq!(
                        &simd.2, &sisd.2,
                        "consensus mismatch mode={} type={:?} seqs={:?}", mode, alignment_type, seqs
                    );
                }
            }
        }
    }
}

proptest! {
    // Two ~200-base sequences per case is already a substantial DP fill x 9 combos; keep the case
    // count modest to bound wall-clock time under Rosetta emulation.
    #![proptest_config(ProptestConfig { cases: 12, ..deterministic_config() })]

    /// The SSE4.1 capstone sweep, **int32** tier — and simultaneously the "int32-forcing" parity
    /// test the SIMD kernels plan's Task 11 calls for: the same 9-combo sweep as
    /// `simd_capstone_all_gap_modes_and_types_match_sisd_int16`, but over [`large_dna`]-generated
    /// sequences and the `_large` scorings, which together provably force `Escalation::Int32` for
    /// every case (see `simd_int32_escalation_is_actually_forced_at_the_generators_tightest_margin`
    /// for the proof) — so this exercises `SimdEngine`'s SSE4.1 int32 kernel, not a delegation,
    /// across all three gap modes and all three alignment types. Gated on [`simd_kernel_active`].
    #[test]
    fn simd_capstone_all_gap_modes_and_types_match_sisd_int32(
        seqs in large_dna(LARGE_MIN_LEN, LARGE_MAX_LEN, 2)
    ) {
        if simd_kernel_active() {
            let builders: [(&str, ScoringBuilder); 3] = [
                ("linear", linear_scoring_large),
                ("affine", affine_scoring_large),
                ("convex", convex_scoring_large),
            ];
            for (mode, build) in builders {
                let scoring = build();
                for alignment_type in [
                    AlignmentType::Global,
                    AlignmentType::Local,
                    AlignmentType::Overlap,
                ] {
                    let (simd, sisd) = run_both(alignment_type, scoring, &seqs);
                    prop_assert_eq!(
                        &simd.0, &sisd.0,
                        "alignment mismatch mode={} type={:?} seqs={:?}", mode, alignment_type, seqs
                    );
                    prop_assert_eq!(
                        &simd.1, &sisd.1,
                        "score mismatch mode={} type={:?} seqs={:?}", mode, alignment_type, seqs
                    );
                    prop_assert_eq!(
                        &simd.2, &sisd.2,
                        "consensus mismatch mode={} type={:?} seqs={:?}", mode, alignment_type, seqs
                    );
                }
            }
        }
    }
}
