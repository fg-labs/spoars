//! Parity tests for [`spoars::graph::Graph`]'s heaviest-bundle consensus
//! against the C++ oracle.
//!
//! Graph is tested here using the oracle's *own* alignments rather than any
//! Rust-produced alignment (no Rust aligner exists yet as of this task) — the
//! Graph/engine decoupling the task brief calls out. `build_graph_from_oracle_alignments`
//! replays `exp.alignments[i]` + `case.seqs[i]` (and `case.quals[i]`, when
//! present) through `add_alignment_weight` / `add_alignment_quality`, in
//! order, exactly as the oracle itself does when it builds its graph (see
//! `oracle/spoa_oracle.cpp:536-548`).

mod support;

use proptest::prelude::*;

use spoars::graph::Graph;
use support::generators::{
    deterministic_config, small_dna, upstream_fastq, upstream_fastq_with_names,
};
use support::oracle::{run_oracle, AlignType, OracleCase, OracleResult};

/// Replays one oracle case's own alignments through a fresh [`Graph`], in
/// sequence order, so the resulting graph is exactly what the oracle built
/// internally before it called `GenerateConsensus`.
///
/// `exp.alignments[i]` is `Vec<(i32, i32)>` = `(node_id, seq_index)` pairs,
/// matching `add_alignment`'s `(i32, i32)` tuple order exactly (confirmed
/// against `oracle/spoa_oracle.cpp:574`, which emits
/// `alignment[j].first, alignment[j].second` where `first` is the graph node
/// id and `second` is the query/sequence index — and against
/// `third_party/spoa/src/graph.cpp:162-247`, which reads
/// `alignment[i]->first` as a node id and `alignment[i]->second` as a
/// sequence index).
fn build_graph_from_oracle_alignments(case: &OracleCase, exp: &OracleResult) -> Graph {
    let mut g = Graph::new();
    for i in 0..case.seqs.len() {
        let seq = case.seqs[i].as_bytes();
        let alignment = &exp.alignments[i];
        if let Some(quals) = &case.quals {
            g.add_alignment_quality(alignment, seq, quals[i].as_bytes())
                .unwrap_or_else(|e| panic!("add_alignment_quality failed for seq {i}: {e}"));
        } else {
            g.add_alignment_weight(alignment, seq, 1)
                .unwrap_or_else(|e| panic!("add_alignment_weight failed for seq {i}: {e}"));
        }
    }
    g
}

#[test]
fn consensus_matches_oracle_on_upstream_fastq() {
    let (seqs, _quals) = upstream_fastq();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let cases = vec![OracleCase::nw(&seq_refs)];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        assert_eq!(
            g.generate_consensus(),
            exp.consensus,
            "case id={} consensus mismatch",
            case.id
        );
    }
}

#[test]
fn msa_matches_oracle_on_upstream_fastq() {
    // The oracle emits `GenerateMultipleSequenceAlignment(false)` (see
    // `oracle/spoa_oracle.cpp:552-553`), so parity is asserted against
    // `generate_msa(false)` (no consensus row appended).
    let (seqs, _quals) = upstream_fastq();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let cases = vec![OracleCase::nw(&seq_refs)];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        assert_eq!(
            g.generate_msa(false),
            exp.msa,
            "case id={} msa mismatch",
            case.id
        );
    }
}

#[test]
fn gfa_matches_oracle_on_upstream_fastq() {
    // The oracle computes consensus BEFORE emitting GFA (spoa_oracle.cpp: GenerateConsensus,
    // then PrintGfaToString), so this test must call `generate_consensus()` before `to_gfa`
    // too, or the `ic:Z:true` consensus tags won't match.
    let (seqs, _quals, names) = upstream_fastq_with_names();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let mut case = OracleCase::nw(&seq_refs);
    case.names = Some(names.clone());
    let cases = vec![case];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        g.generate_consensus();
        assert_eq!(
            g.to_gfa(&names, &[], false),
            exp.gfa,
            "case id={} gfa mismatch",
            case.id
        );
    }
}

#[test]
fn gfa_uses_supplied_non_index_names_end_to_end() {
    // The upstream corpus's FASTQ headers happen to be "0".."54" — identical to the oracle's
    // `std::to_string(i)` header fallback — so `gfa_matches_oracle_on_upstream_fastq` can't tell
    // whether the names pipeline works or is silently ignored on both sides. This test closes
    // that gap with names that are DELIBERATELY NOT their indices, so a broken pipeline (oracle
    // ignoring "names" and falling back to "0"/"1"/"2", or the Rust `to_gfa` mis-emitting the
    // `headers[i]` P-line column) produces a visible mismatch.
    let seqs = ["ACGT", "ACGT", "AGT"];
    let names = vec![
        "read_alpha".to_string(),
        "contig_7".to_string(),
        "sampleXYZ".to_string(),
    ];
    let mut case = OracleCase::nw(&seqs);
    case.names = Some(names.clone());
    let cases = vec![case];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        g.generate_consensus();
        assert_eq!(
            g.to_gfa(&names, &[], false),
            exp.gfa,
            "case id={} gfa mismatch",
            case.id
        );
        // The observable assertion: the P-line header column comes straight from `headers[i]`,
        // so these exact non-index strings must appear. If the oracle's `has_names` branch were
        // broken (falling back to "0"/"1"/"2"), or the Rust emitter used the wrong column, these
        // substrings would be absent and this fails.
        assert!(
            exp.gfa.contains("P\tread_alpha\t"),
            "oracle GFA missing 'read_alpha' P-line; names not flowing end-to-end:\n{}",
            exp.gfa
        );
        assert!(
            exp.gfa.contains("P\tcontig_7\t"),
            "oracle GFA missing 'contig_7' P-line; names not flowing end-to-end:\n{}",
            exp.gfa
        );
        assert!(
            exp.gfa.contains("P\tsampleXYZ\t"),
            "oracle GFA missing 'sampleXYZ' P-line; names not flowing end-to-end:\n{}",
            exp.gfa
        );
    }
}

#[test]
fn dot_matches_oracle_on_upstream_fastq() {
    // Same consensus-before-emit ordering requirement as the GFA test above.
    let (seqs, _quals) = upstream_fastq();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let cases = vec![OracleCase::nw(&seq_refs)];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        g.generate_consensus();
        assert_eq!(g.to_dot(), exp.dot, "case id={} dot mismatch", case.id);
    }
}

#[test]
fn consensus_min_coverage_matches_oracle_on_upstream_fastq() {
    let (seqs, _quals) = upstream_fastq();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let min_coverage = 5i32;
    let mut case = OracleCase::nw(&seq_refs);
    case.min_coverage = min_coverage;
    let cases = vec![case];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        assert_eq!(
            g.generate_consensus_min_coverage(min_coverage),
            exp.consensus,
            "case id={} min_coverage consensus mismatch",
            case.id
        );
    }
}

#[test]
fn consensus_and_msa_match_oracle_with_qualities() {
    // Every other parity test in this file discards the FASTQ qualities (`let (seqs, _quals) =
    // upstream_fastq()`), so the Phred `q-33` weighting path (`add_alignment_quality`, which
    // changes edge weights and therefore can change heaviest-bundle consensus tie-breaks) has had
    // zero differential coverage until now. This is the closest thing to a real quality-bearing
    // corpus available (the upstream sample.fastq.gz fixture), so it drives `with_quals` end to
    // end: oracle case carries `quals`, and `build_graph_from_oracle_alignments` (see its own doc
    // comment above) routes to `add_alignment_quality` whenever `case.quals` is `Some`.
    let (seqs, quals) = upstream_fastq();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let qual_refs: Vec<&str> = quals.iter().map(|q| q.as_str()).collect();

    // A second, synthetic case designed so quality weighting actually FLIPS the consensus versus
    // uniform (unweighted) voting: two sequences vote "T" at the last base but carry low quality
    // (`'#'` = Phred 2), one sequence votes "A" but carries high quality (`'I'` = Phred 40). Under
    // uniform weight-1 voting this is a 2-vs-1 majority for "T" (confirmed separately against the
    // oracle: `OracleCase::nw` on this input yields consensus "ACGT"), but the Phred-weighted sum
    // (2+2=4 for "T" vs 40 for "A") should make heaviest-bundle traversal pick "A" instead
    // (confirmed against the oracle: consensus "ACGA"). Without this case, quality-weighting
    // parity here would be non-discriminating: the real FASTQ corpus's Phred qualities happen to
    // be uniform enough that its quality-weighted consensus is byte-identical to the unweighted
    // one, so a broken `add_alignment_quality` (e.g. one that silently used weight 1 throughout)
    // would still pass a quality-weighted-consensus check against that corpus alone.
    let tie_break_seqs = ["ACGT", "ACGT", "ACGA"];
    let tie_break_quals = ["####", "####", "IIII"];

    let cases = vec![
        OracleCase::with_quals(AlignType::Nw, &seq_refs, &qual_refs),
        OracleCase::with_quals(AlignType::Nw, &tie_break_seqs, &tie_break_quals),
    ];
    let expected = run_oracle(&cases);

    for (case, exp) in cases.iter().zip(&expected) {
        let mut g = build_graph_from_oracle_alignments(case, exp);
        assert_eq!(
            g.generate_consensus(),
            exp.consensus,
            "case id={} consensus mismatch (quality-weighted); input {:?}",
            case.id,
            case.seqs
        );
        assert_eq!(
            g.generate_msa(false),
            exp.msa,
            "case id={} msa mismatch (quality-weighted); input {:?}",
            case.id,
            case.seqs
        );
    }

    // Belt-and-suspenders: confirm the synthetic case's oracle consensus really is "ACGA" (the
    // quality-weighted answer), not "ACGT" (the naive majority-vote answer) — otherwise the
    // discriminating power this case is designed to add would silently regress if the oracle's
    // scoring defaults ever changed.
    assert_eq!(
        expected[1].consensus, "ACGA",
        "tie-break fixture no longer discriminates quality weighting from uniform \
         weighting (expected the Phred-weighted consensus \"ACGA\", not the naive-majority \
         \"ACGT\"); the fixture's constants need revisiting"
    );
}

#[test]
fn consensus_and_msa_match_oracle_on_degenerate_inputs() {
    // Explicit, deterministic corner cases the random proptest below hits only probabilistically:
    // a single sequence, length-1 sequences, lowercase bases, and an ambiguous/non-ACGT base.
    // spoa's coder is 256-wide and round-trips arbitrary bytes, so these are legitimate inputs, not
    // error cases.
    let corpora: Vec<Vec<&str>> = vec![
        vec!["ACGTACGT"],     // single sequence: consensus must equal the sequence itself
        vec!["A", "A", "C"],  // length-1 sequences
        vec!["acgt", "acgt"], // lowercase
        vec!["ACGN", "ACGT"], // N / ambiguous base
    ];

    for seqs in corpora {
        let cases = vec![OracleCase::nw(&seqs)];
        let expected = run_oracle(&cases);

        for (case, exp) in cases.iter().zip(&expected) {
            if seqs.len() == 1 {
                assert_eq!(
                    exp.consensus, seqs[0],
                    "sanity check: oracle consensus for a single sequence should equal it \
                     verbatim; corpus {seqs:?}"
                );
            }

            let mut g = build_graph_from_oracle_alignments(case, exp);
            assert_eq!(
                g.generate_consensus(),
                exp.consensus,
                "corpus {seqs:?}: consensus mismatch"
            );
            assert_eq!(
                g.generate_msa(false),
                exp.msa,
                "corpus {seqs:?}: msa mismatch"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..deterministic_config() })]

    /// Differential fuzzer: for one randomly generated small-DNA input, builds NW/SW/OV oracle
    /// cases (batched into a single oracle round-trip), replays each oracle alignment through a
    /// fresh [`Graph`], and asserts `generate_consensus`/`generate_msa`/`to_gfa`/`to_dot` all
    /// match the oracle exactly. This is the first differential coverage of SW (local) and OV
    /// (overlap) alignment shapes, and of graphs built from randomized (not just the fixed
    /// 55-sequence corpus) input — a real faithfulness bug in the graph builder's handling of
    /// partial/local alignments is squarely what this is designed to catch.
    /// `prop_assert_eq!` (not `assert_eq!`) is used throughout so a failure shrinks to a minimal
    /// reproducing input, and every assertion message includes the align type and the exact
    /// generated input so a failure is directly diagnosable without re-running under a debugger.
    #[test]
    fn consensus_msa_gfa_dot_match_oracle_across_align_types(seqs in small_dna(12, 5)) {
        let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
        let cases = vec![
            OracleCase::nw(&seq_refs),
            OracleCase::sw(&seq_refs),
            OracleCase::ov(&seq_refs),
        ];
        let expected = run_oracle(&cases);
        // With `case.names` left `None`, the oracle falls back to `std::to_string(i)` GFA P-line
        // headers, so supplying the same index strings to `to_gfa` here keeps both sides
        // consistent without asserting anything about the (separately-tested) names pipeline.
        let names: Vec<String> = (0..seq_refs.len()).map(|i| i.to_string()).collect();

        for (case, exp) in cases.iter().zip(&expected) {
            let mut g = build_graph_from_oracle_alignments(case, exp);

            let consensus = g.generate_consensus();
            prop_assert_eq!(
                &consensus, &exp.consensus,
                "consensus mismatch for align type {:?} on input {:?}", case.ty, seqs
            );

            let msa = g.generate_msa(false);
            prop_assert_eq!(
                &msa, &exp.msa,
                "msa mismatch for align type {:?} on input {:?}", case.ty, seqs
            );

            let gfa = g.to_gfa(&names, &[], false);
            prop_assert_eq!(
                &gfa, &exp.gfa,
                "gfa mismatch for align type {:?} on input {:?}", case.ty, seqs
            );

            let dot = g.to_dot();
            prop_assert_eq!(
                &dot, &exp.dot,
                "dot mismatch for align type {:?} on input {:?}", case.ty, seqs
            );
        }
    }
}
