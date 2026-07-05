//! Programmatic input generators for the oracle differential tests.
//!
//! `small_dna` is a proptest strategy for random-ish small alignment inputs;
//! `upstream_fastq` loads the fixed FASTQ fixture spoa's own upstream test
//! suite uses, giving parity tests a real quality-bearing corpus in addition
//! to synthetic cases.

use std::path::PathBuf;

use needletail::parse_fastx_file;
use proptest::collection::vec;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, RngSeed};

/// Characters spoa's alignment coder round-trips through decoder / MSA /
/// consensus / GFA: standard DNA bases in both cases, `N`/`n`, and the IUPAC
/// ambiguity codes. spoa's coder is 256-wide and accepts any byte, but this
/// generator sticks to the biologically meaningful subset that differential
/// tests actually care about rather than arbitrary bytes.
const DNA_ALPHABET: &[char] = &[
    'A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'N', 'n', 'R', 'Y', 'S', 'W', 'K', 'M',
];

/// A fixed proptest RNG seed so failing cases reproduce deterministically
/// both locally and in CI, instead of proptest's default per-run random
/// seed. Parity test binaries should pass [`deterministic_config`] to
/// `TestRunner::new` (or override `proptest!`'s config) rather than relying
/// on `PROPTEST_RNG_SEED` being set in the environment.
const FIXED_RNG_SEED: u64 = 0x5b0a_2510_c0de_5eed;

/// A `proptest::test_runner::Config` pinned to [`FIXED_RNG_SEED`], for the
/// parity test binaries to reuse so every run explores the same sequence of
/// generated cases.
pub fn deterministic_config() -> Config {
    Config {
        rng_seed: RngSeed::Fixed(FIXED_RNG_SEED),
        ..Config::default()
    }
}

fn dna_char() -> impl Strategy<Value = char> {
    (0..DNA_ALPHABET.len()).prop_map(|i| DNA_ALPHABET[i])
}

fn dna_seq(max_len: usize) -> impl Strategy<Value = String> {
    vec(dna_char(), 1..=max_len).prop_map(|chars| chars.into_iter().collect())
}

/// A proptest strategy generating 1..=`n_seqs` sequences, each of length
/// 1..=`max_len`, drawn from [`DNA_ALPHABET`] (mixed-case DNA plus `N` and
/// IUPAC ambiguity codes). At `max_len == 1`, every generated sequence is
/// length-1; at `n_seqs == 1`, every generated batch is a single sequence —
/// covering both degenerate cases via the caller's choice of bounds rather
/// than requiring a separate zero-sequence case (spoa's `AlignmentEngine`
/// requires at least one sequence per case).
pub fn small_dna(max_len: usize, n_seqs: usize) -> impl Strategy<Value = Vec<String>> {
    debug_assert!(
        n_seqs >= 1,
        "small_dna: n_seqs must be >= 1 (spoa needs at least one sequence); \
         n_seqs == 0 builds an inverted 1..=0 range that silently yields an \
         empty strategy"
    );
    debug_assert!(max_len >= 1, "small_dna: max_len must be >= 1");
    vec(dna_seq(max_len), 1..=n_seqs)
}

/// Locates the crate root regardless of the calling test binary's working
/// directory.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Loads sequences and per-base Phred quality strings from the upstream spoa
/// test fixture `third_party/spoa/test/data/sample.fastq.gz` (the same file
/// `third_party/spoa/test/spoa_test.cpp` parses, which asserts it holds 55
/// records). Returned as `(sequences, qualities)`, index-aligned, so this
/// corpus can drive the Phred `q-33` weighting path in
/// `Graph::AddAlignment` — a real quality-bearing input, not a synthetic
/// one, and the only source of that in this module.
///
/// A thin wrapper over [`upstream_fastq_with_names`] that drops the record
/// names, for the (majority of) callers that don't need GFA `P`-line names.
pub fn upstream_fastq() -> (Vec<String>, Vec<String>) {
    let (seqs, quals, _names) = upstream_fastq_with_names();
    (seqs, quals)
}

/// Like [`upstream_fastq`], but also returns each record's name: the first
/// whitespace-delimited token of its FASTQ header line (biosoup's `Sequence`
/// semantics — the same convention spoa's own CLI uses to build its GFA
/// `P`-line names), not the full header line, which may carry additional
/// whitespace-separated description text. Returned as
/// `(sequences, qualities, names)`, all index-aligned.
pub fn upstream_fastq_with_names() -> (Vec<String>, Vec<String>, Vec<String>) {
    let path = crate_root().join("third_party/spoa/test/data/sample.fastq.gz");
    let mut reader = parse_fastx_file(&path).unwrap_or_else(|e| {
        panic!(
            "failed to open upstream FASTQ fixture {}: {e}",
            path.display()
        )
    });

    let mut seqs = Vec::new();
    let mut quals = Vec::new();
    let mut names = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.unwrap_or_else(|e| {
            panic!(
                "failed to parse a record from upstream FASTQ fixture {}: {e}",
                path.display()
            )
        });
        let seq = String::from_utf8(record.seq().into_owned()).unwrap_or_else(|e| {
            panic!(
                "non-UTF8 sequence bytes in upstream FASTQ fixture {}: {e}",
                path.display()
            )
        });
        let qual = record.qual().unwrap_or_else(|| {
            panic!(
                "record in upstream FASTQ fixture {} is missing qualities",
                path.display()
            )
        });
        let qual = String::from_utf8(qual.to_vec()).unwrap_or_else(|e| {
            panic!(
                "non-UTF8 quality bytes in upstream FASTQ fixture {}: {e}",
                path.display()
            )
        });
        let id = record.id();
        let name_bytes = id.split(|&b| b == b' ' || b == b'\t').next().unwrap_or(id);
        let name = String::from_utf8(name_bytes.to_vec()).unwrap_or_else(|e| {
            panic!(
                "non-UTF8 record name bytes in upstream FASTQ fixture {}: {e}",
                path.display()
            )
        });
        seqs.push(seq);
        quals.push(qual);
        names.push(name);
    }

    (seqs, quals, names)
}
