//! Integration test binary for the C++ oracle subprocess harness.
//!
//! Each file directly under `tests/` compiles to its own test binary, and
//! test binaries cannot import one another. The harness and generator code
//! that later parity-test binaries (graph_parity, engine_parity, cli_parity)
//! also need therefore lives in `tests/support/` (a subdirectory, which is
//! NOT compiled as its own binary) and is pulled in here via `mod support;`.

mod support;

use support::generators::upstream_fastq;
use support::oracle::{run_oracle, AlignType, OracleCase};

#[test]
fn oracle_roundtrips_a_trivial_case() {
    let cases = vec![OracleCase::nw(&["ACGT", "AGT"])];
    let out = run_oracle(&cases);
    assert_eq!(out.len(), 1);
    assert!(!out[0].consensus.is_empty());
    assert_eq!(out[0].alignments.len(), 2);
}

/// SISD tripwire (regression, not a differential-parity test): freezes the
/// oracle's output for a small case chosen to hit a score tie (three
/// sequences of equal length, one base apart, so several equally-scoring
/// alignments/backtrack paths exist) as an inline expected value.
///
/// `oracle/CMakeLists.txt` forces spoa's SISD (scalar) alignment engine by
/// withholding every SIMD-enabling define; on score ties, spoa's SIMD
/// backtrack can diverge from SISD's. If a future change to that CMake
/// config (or a CI host/toolchain change) ever let the SIMD engine compile
/// in silently, this test's frozen consensus/alignment would very likely
/// stop matching — catching a class of bug that a purely-behavioral
/// "non-empty consensus" smoke test cannot.
#[test]
fn oracle_sisd_tripwire_matches_frozen_output() {
    let cases = vec![OracleCase {
        id: 0,
        ty: AlignType::Nw,
        m: 5,
        n: -4,
        g: -8,
        e: -6,
        q: -10,
        c: -4,
        seqs: vec![
            "ACGTACGT".to_string(),
            "ACGAACGT".to_string(),
            "ACGTACGA".to_string(),
        ],
        quals: None,
        min_coverage: -1,
        names: None,
        subgraph: None,
        summarize_consensus: false,
    }];
    let out = run_oracle(&cases);
    assert_eq!(out.len(), 1);

    // Frozen against this arm64 (SISD-forced) build; see doc comment above.
    // The first sequence's alignment to an empty graph is trivially empty
    // (nothing to align against yet); the other two align 1:1 to the graph
    // built so far since they're each one substitution away from "ACGTACGT".
    assert_eq!(out[0].consensus, "ACGTACGT");
    assert_eq!(out[0].alignments[0], Vec::<(i32, i32)>::new());
    assert_eq!(
        out[0].alignments[1],
        vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7)
        ]
    );
    assert_eq!(
        out[0].alignments[2],
        vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7)
        ]
    );
}

#[test]
fn upstream_fastq_matches_upstream_record_count_and_seq_qual_lengths() {
    let (seqs, quals) = upstream_fastq();

    // third_party/spoa/test/spoa_test.cpp asserts this same fixture parses
    // to 55 records; mirroring that count here confirms this loader reads
    // the identical fixture upstream's own test suite validates against.
    assert_eq!(seqs.len(), 55);
    assert_eq!(quals.len(), 55);

    for (i, (seq, qual)) in seqs.iter().zip(quals.iter()).enumerate() {
        assert_eq!(
            seq.len(),
            qual.len(),
            "record {i}: sequence and quality lengths differ"
        );
    }
}

/// Regression test for two harness defects: (1) the pipe deadlock that hit
/// once a batch's combined stdin/stdout exceeded the OS pipe buffer, and
/// (2) constructors hardcoding `id: 0`, which broke `run_oracle`'s id-keyed
/// correlation for multi-case batches. Builds a batch large enough that its
/// combined stdout comfortably exceeds a single pipe buffer (six cases, each
/// six ~30-base sequences one substitution apart, so each result carries a
/// non-trivial msa/gfa/dot), relying entirely on the constructors'
/// auto-assigned sequential ids. Under the old single-threaded write-then-read
/// code this would hang forever; under the old `id: 0` constructors the id
/// assertions would fail.
#[test]
fn run_oracle_handles_a_multi_case_batch_without_deadlock() {
    // A ~30-base template plus five single-substitution variants of it, so
    // each case builds a real (branchy) POA graph and emits a sizeable
    // msa/gfa/dot rather than a trivial one.
    let template = "ACGTACGTACGTACGTACGTACGTACGTAC";
    let variants: Vec<String> = (0..6)
        .map(|i| {
            let mut bytes = template.as_bytes().to_vec();
            let pos = (i * 5) % bytes.len();
            bytes[pos] = if bytes[pos] == b'A' { b'C' } else { b'A' };
            String::from_utf8(bytes).unwrap()
        })
        .collect();
    let seqs: Vec<&str> = variants.iter().map(|s| s.as_str()).collect();

    // Auto-assigned ids: we do NOT set ids manually. Interleave the three
    // alignment modes to prove the counter is shared across constructors.
    let cases = vec![
        OracleCase::nw(&seqs),
        OracleCase::sw(&seqs),
        OracleCase::ov(&seqs),
        OracleCase::nw(&seqs),
        OracleCase::sw(&seqs),
        OracleCase::ov(&seqs),
    ];
    let mut expected_ids: Vec<u32> = cases.iter().map(|c| c.id).collect();

    // Constructors must have handed out six distinct ids (the old `id: 0`
    // constructors would collide). We assert distinctness rather than strict
    // 0,1,2,... contiguity because the id counter is process-wide and cargo
    // runs test functions in parallel, so a constructor in another test may
    // interleave — but every id is still unique.
    let mut distinct = expected_ids.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        cases.len(),
        "constructors handed out duplicate ids"
    );
    expected_ids.sort_unstable();

    let out = run_oracle(&cases);

    // (a) one result per case.
    assert_eq!(out.len(), cases.len());
    // (b) results carry exactly the distinct ids the constructors assigned
    //     (run_oracle returns them sorted by id).
    let got_ids: Vec<u32> = out.iter().map(|r| r.id).collect();
    assert_eq!(got_ids, expected_ids);
    // (c) every consensus is non-empty.
    for result in &out {
        assert!(
            !result.consensus.is_empty(),
            "case id={} produced an empty consensus",
            result.id
        );
    }
}
