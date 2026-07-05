//! End-to-end parity tests for the `spoars` CLI binary against the offline
//! C++ oracle.
//!
//! The reference is the **offline oracle** (`support::oracle::run_oracle`), not the real `spoa`
//! executable (which would need a network `FetchContent` of `bioparser`/`biosoup`/zlib to build).
//! The oracle reconstructs consensus/MSA/GFA graph state exactly as upstream spoa would, so each
//! test here drives it with the same corpus (sequences + qualities + names) and defaults the CLI
//! itself would use on that corpus, then diffs the CLI's stdout against the oracle's output.
//!
//! `CARGO_BIN_EXE_spoars` (set automatically by cargo for integration tests) points at the built
//! `spoars` binary, so no extra process-execution dependency (e.g. `assert_cmd`) is needed.

mod support;

use std::process::{Command, Output};

use spoars::graph::Graph;
use support::generators::upstream_fastq_with_names;
use support::oracle::{run_oracle, AlignType, OracleCase, OracleResult};

/// Path (relative to the crate root) to the upstream FASTQ fixture the CLI parity tests drive.
const DATA: &str = "third_party/spoa/test/data/sample.fastq.gz";

/// Runs the built `spoars` binary with `args`, from the crate root (so relative paths like
/// [`DATA`] resolve the same way they do for `cargo test`).
fn run_bin(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_spoars"))
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to execute the spoars binary")
}

/// Runs the built `spoars` binary and asserts it exited successfully, returning its stdout as a
/// `String`.
fn run_bin_ok(args: &[&str]) -> String {
    let output = run_bin(args);
    assert!(
        output.status.success(),
        "spoars {args:?} exited with {}: stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("spoars stdout was not valid UTF-8")
}

/// Builds an [`OracleCase`] from the upstream FASTQ corpus (sequences, qualities, and names),
/// carrying `ty`'s alignment type and spoa's CLI-default scores (`m/n/g/e/q/c`), so it matches
/// exactly what the `spoars` CLI builds by default when pointed at [`DATA`] (default `-l 0` =
/// `AlignType::Sw`, per the CLI's `algorithm = 0` default).
fn base_case(ty: AlignType) -> (OracleCase, Vec<String>) {
    let (seqs, quals, names) = upstream_fastq_with_names();
    let seq_refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
    let qual_refs: Vec<&str> = quals.iter().map(|q| q.as_str()).collect();
    let mut case = OracleCase::with_quals(ty, &seq_refs, &qual_refs);
    case.names = Some(names.clone());
    (case, names)
}

/// Replays one oracle case's own alignments through a fresh [`Graph`], in sequence order, so the
/// resulting graph is exactly what the oracle (and, if faithful, the CLI) built internally before
/// generating consensus/MSA/GFA. Mirrors `tests/graph_parity.rs`'s helper of the same shape.
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
fn cli_r0_consensus_matches_oracle_defaults() {
    // Defaults: -l 0 (SW), -r 0, scores m=5 n=-4 g=-8 e=-6 q=-10 c=-4, min-coverage=-1.
    let (case, _names) = base_case(AlignType::Sw);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];

    let stdout = run_bin_ok(&["-r", "0", DATA]);
    let expected = format!(
        ">Consensus LN:i:{}\n{}\n",
        exp.consensus.len(),
        exp.consensus
    );
    assert_eq!(stdout, expected);
}

#[test]
fn cli_r1_msa_matches_oracle() {
    let (case, names) = base_case(AlignType::Sw);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];

    let stdout = run_bin_ok(&["-r", "1", DATA]);

    let mut expected = String::new();
    for (i, row) in exp.msa.iter().enumerate() {
        expected.push_str(&format!(">{}\n{row}\n", names[i]));
    }
    assert_eq!(stdout, expected);
}

#[test]
fn cli_r2_msa_and_consensus_matches_library_replay() {
    // The oracle only ever calls `GenerateMultipleSequenceAlignment(false)` (see
    // `oracle/spoa_oracle.cpp:565-566`), so there is no oracle-native `-r 2` (MSA + trailing
    // consensus row) reference. Instead, replay the oracle's own alignments through the (already
    // graph-parity-tested, see `tests/graph_parity.rs`) Rust `Graph::generate_msa(true)` on the
    // same input, and compare the CLI's stdout against THAT — an honest reference built from the
    // real library, not a re-derivation of the CLI's own logic.
    let (case, names) = base_case(AlignType::Sw);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];
    let mut g = build_graph_from_oracle_alignments(&case, exp);
    let msa_with_consensus = g.generate_msa(true);

    let stdout = run_bin_ok(&["-r", "2", DATA]);

    let mut expected = String::new();
    for (i, row) in msa_with_consensus.iter().enumerate() {
        let name = if i < names.len() {
            names[i].as_str()
        } else {
            "Consensus"
        };
        expected.push_str(&format!(">{name}\n{row}\n"));
    }
    assert_eq!(stdout, expected);
}

#[test]
fn cli_r3_gfa_matches_oracle() {
    // The oracle's GFA is always `include_consensus = false` with an empty `is_reversed`
    // (`oracle/spoa_oracle.cpp` comment above `PrintGfaToString`), exactly matching the CLI's
    // `-r 3` (mode == 4 is the only include_consensus=true case) with `--strand-ambiguous` unset.
    let (case, _names) = base_case(AlignType::Sw);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];

    let stdout = run_bin_ok(&["-r", "3", DATA]);
    assert_eq!(stdout, exp.gfa);
}

#[test]
fn cli_r0_consensus_matches_oracle_with_non_default_scores() {
    let (mut case, _names) = base_case(AlignType::Sw);
    case.m = 3;
    case.n = -2;
    let exp = &run_oracle(std::slice::from_ref(&case))[0];

    let stdout = run_bin_ok(&["-m", "3", "-n", "-2", "-r", "0", DATA]);
    let expected = format!(
        ">Consensus LN:i:{}\n{}\n",
        exp.consensus.len(),
        exp.consensus
    );
    assert_eq!(stdout, expected);
}

#[test]
fn cli_l1_nw_consensus_matches_oracle() {
    let (case, _names) = base_case(AlignType::Nw);
    let exp = &run_oracle(std::slice::from_ref(&case))[0];

    let stdout = run_bin_ok(&["-l", "1", "-r", "0", DATA]);
    let expected = format!(
        ">Consensus LN:i:{}\n{}\n",
        exp.consensus.len(),
        exp.consensus
    );
    assert_eq!(stdout, expected);
}

#[test]
fn cli_version_prints_crate_version() {
    let stdout = run_bin_ok(&["--version"]);
    assert_eq!(stdout, format!("{}\n", env!("CARGO_PKG_VERSION")));
}

#[test]
fn cli_invalid_algorithm_exits_non_zero() {
    let output = run_bin(&["-l", "5", DATA]);
    assert!(
        !output.status.success(),
        "spoars -l 5 should fail (algorithm >= 3 is invalid)"
    );
}

#[test]
fn cli_score_i8_wrap_does_not_error() {
    // spoa's `atoi(optarg)` -> `int8_t m` narrowing silently wraps (200 -> -56 for a signed
    // 8-bit type) rather than erroring; a plain `str::parse::<i8>()` would instead fail to
    // parse "200" at all and diverge from upstream. `m` is otherwise unconstrained (only the
    // gap-open/gap-extend penalties are sign-validated by `Scoring::new`), so this must still run
    // to completion successfully with the wrapped value.
    let output = run_bin(&["-m", "200", "-r", "0", DATA]);
    assert!(
        output.status.success(),
        "spoars -m 200 should succeed via i32->i8 wraparound (200 -> -56), not fail to parse; \
         stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}
