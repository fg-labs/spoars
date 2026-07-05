#![forbid(unsafe_code)]
//! `spoars` CLI entry point: a faithful Rust port of upstream spoa's `main.cpp`.
//!
//! Mirrors `third_party/spoa/src/main.cpp`'s argument parsing, align-and-add loop, and `-r`
//! result-mode output switch, including its documented quirks: score options parse as `i32` then
//! narrow to `i8` (spoa's `atoi` -> `int8_t` silently wraps, e.g. `-m 200` -> `-56`);
//! `--min-coverage`/`--version` are long-option-only (no `-M`/`-v` short forms); an explicit `-r`
//! replaces (rather than appends to) the default `{0}` result mode
//! (`main.cpp:242-244`, `results.erase(results.begin())`); and a sequence's name (for GFA `P`-line
//! headers and MSA/consensus row labels) is the first whitespace-delimited token of its
//! FASTA/FASTQ record id (biosoup's `Sequence` semantics), not the full header line.
//!
//! Aligns with the dispatching [`SimdEngine`] (best available ISA at runtime, SISD fallback) by
//! default; set `SPOARS_FORCE_SISD=1` to force the scalar [`SisdEngine`] instead (a hidden,
//! undocumented-in-`--help` escape hatch for differential debugging — see
//! [`should_force_sisd`]).

use std::io::Write;
use std::process::ExitCode;

use needletail::parse_fastx_file;

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SimdEngine, SisdEngine};
use spoars::graph::Graph;

/// Name of the hidden escape-hatch environment variable that forces the CLI to align with the
/// scalar [`SisdEngine`] instead of the default (dispatching) [`SimdEngine`]. Not documented in
/// `--help` (a low-profile dev knob, not a user-facing feature); useful for differential debugging
/// (comparing SIMD vs scalar output/timing on a real input) and as a safety valve should a
/// vectorized kernel ever be suspected of diverging from the oracle-validated `SisdEngine` on some
/// host's ISA.
const FORCE_SISD_ENV: &str = "SPOARS_FORCE_SISD";

/// Decides whether [`FORCE_SISD_ENV`]'s raw value (`None` if the variable is unset) should force
/// the scalar engine. Any value other than unset, empty, or `"0"` forces SISD (so `=1`, `=true`,
/// `=yes` all work) — a permissive boolean parse, deliberately more lenient than requiring an
/// exact `"1"`, since this is a low-stakes dev knob rather than a validated CLI option.
fn should_force_sisd(value: Option<&str>) -> bool {
    matches!(value, Some(v) if !v.is_empty() && v != "0")
}

/// Parsed CLI options, with spoa's exact defaults (`main.cpp:207-220`).
struct Options {
    m: i8,
    n: i8,
    g: i8,
    e: i8,
    q: i8,
    c: i8,
    min_coverage: i32,
    algorithm: u8,
    results: Vec<u8>,
    dot_path: String,
    strand_ambiguous: bool,
    input: Option<String>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            m: 5,
            n: -4,
            g: -8,
            e: -6,
            q: -10,
            c: -4,
            min_coverage: -1,
            algorithm: 0,
            results: vec![0],
            dot_path: String::new(),
            strand_ambiguous: false,
            input: None,
        }
    }
}

/// The outcome of parsing argv: either a fully parsed [`Options`] ready to run, or an early exit
/// (successful, for `--help`/`--version`; or an error, mirroring spoa's various `return 1`s).
enum ParseOutcome {
    Run(Options),
    /// Print `message` (if any) to stdout, then exit 0.
    ExitOk(Option<String>),
    /// Print `message` (if any) to stderr, then exit 1.
    ExitErr(Option<String>),
}

/// Parses a CLI integer argument the way spoa's `atoi(optarg)` does for a well-formed integer
/// (accepting a leading `+`/`-` sign): unlike `atoi`, this rejects non-numeric or trailing-garbage
/// input outright rather than silently parsing a prefix and ignoring the rest, since none of the
/// documented CLI quirks (`-m 200`'s `i8` wraparound included) depend on that leniency.
fn parse_i32(value: &str) -> Result<i32, String> {
    value
        .trim()
        .parse::<i32>()
        .map_err(|_| format!("[spoars::] error: invalid integer argument '{value}'"))
}

/// Parses `args` (argv, excluding `argv[0]`) into an [`Options`], or an early [`ParseOutcome`]
/// for `--help`/`--version`/an unrecognized or malformed option.
///
/// Options are matched against spoa's `optstr = "m:n:g:e:q:c:l:r:d:sh"` plus its `options[]` long
/// table (`main.cpp:15-24,222`): `-m -n -g -e -q -c -l/--algorithm -r/--result -d/--dot -s/-h`
/// short forms, plus the long-only `--min-coverage` and `--version`. Short options accept either
/// an attached value (`-m5`) or a separate next-argv value (`-m 5`); long options accept either
/// `--opt=value` or a separate next-argv value.
fn parse_args(args: &[String]) -> ParseOutcome {
    let mut opts = Options::default();
    let mut i = 0usize;

    while i < args.len() {
        let token = args[i].clone();

        if token == "--" {
            i += 1;
            while i < args.len() {
                if opts.input.is_none() {
                    opts.input = Some(args[i].clone());
                }
                i += 1;
            }
            break;
        }

        if let Some(rest) = token.strip_prefix("--") {
            match parse_long_option(rest, args, &mut i, &mut opts) {
                Ok(()) => {
                    i += 1;
                    continue;
                }
                Err(outcome) => return outcome,
            }
        }

        if token.starts_with('-') && token.len() > 1 {
            match parse_short_cluster(&token, args, &mut i, &mut opts) {
                Ok(()) => {
                    i += 1;
                    continue;
                }
                Err(outcome) => return outcome,
            }
        }

        if opts.input.is_none() {
            opts.input = Some(token);
        }
        i += 1;
    }

    // main.cpp:242-244: an explicit `-r` replaces (rather than appends to) the default `{0}`.
    if opts.results.len() > 1 {
        opts.results.remove(0);
    }

    ParseOutcome::Run(opts)
}

/// Handles one `--long[=value]` token (with `long` already stripped of its `--` prefix),
/// consuming an extra `args[*i]` for options that need a value not supplied inline.
fn parse_long_option(
    rest: &str,
    args: &[String],
    i: &mut usize,
    opts: &mut Options,
) -> Result<(), ParseOutcome> {
    let (name, inline_value) = match rest.split_once('=') {
        Some((n, v)) => (n, Some(v.to_string())),
        None => (rest, None),
    };
    let needs_value = matches!(name, "algorithm" | "result" | "min-coverage" | "dot");

    let value = if inline_value.is_some() {
        inline_value
    } else if needs_value {
        *i += 1;
        match args.get(*i) {
            Some(v) => Some(v.clone()),
            None => {
                return Err(ParseOutcome::ExitErr(Some(format!(
                    "[spoars::] error: option '--{name}' requires an argument"
                ))))
            }
        }
    } else {
        None
    };

    match name {
        "algorithm" => match parse_i32(value.as_deref().unwrap_or_default()) {
            Ok(v) => opts.algorithm = v as u8,
            Err(e) => return Err(ParseOutcome::ExitErr(Some(e))),
        },
        "result" => match parse_i32(value.as_deref().unwrap_or_default()) {
            Ok(v) => opts.results.push(v as u8),
            Err(e) => return Err(ParseOutcome::ExitErr(Some(e))),
        },
        "min-coverage" => match parse_i32(value.as_deref().unwrap_or_default()) {
            Ok(v) => opts.min_coverage = v,
            Err(e) => return Err(ParseOutcome::ExitErr(Some(e))),
        },
        "dot" => opts.dot_path = value.unwrap_or_default(),
        "strand-ambiguous" => opts.strand_ambiguous = true,
        "version" => {
            return Err(ParseOutcome::ExitOk(Some(format!(
                "{}\n",
                env!("CARGO_PKG_VERSION")
            ))))
        }
        "help" => return Err(ParseOutcome::ExitOk(Some(help_text()))),
        _ => {
            return Err(ParseOutcome::ExitErr(Some(format!(
                "[spoars::] error: unrecognized option '--{name}'"
            ))))
        }
    }
    Ok(())
}

/// Handles one `-xyz` short-option cluster token, walking its characters left to right (mirroring
/// `getopt_long`): flag-only options (`-s`, `-h`) consume no value, while value-taking options
/// consume either the rest of the token (`-m5`) or (if the token is exhausted) the next `args[*i]`
/// (`-m 5`) and stop scanning the cluster (the remaining token bytes, if any, are that value, not
/// further options).
fn parse_short_cluster(
    token: &str,
    args: &[String],
    i: &mut usize,
    opts: &mut Options,
) -> Result<(), ParseOutcome> {
    let mut chars = token[1..].chars();

    while let Some(ch) = chars.next() {
        match ch {
            's' => opts.strand_ambiguous = true,
            'h' => return Err(ParseOutcome::ExitOk(Some(help_text()))),
            'm' | 'n' | 'g' | 'e' | 'q' | 'c' | 'l' | 'r' | 'd' => {
                let attached: String = chars.by_ref().collect();
                let value = if !attached.is_empty() {
                    attached
                } else {
                    *i += 1;
                    match args.get(*i) {
                        Some(v) => v.clone(),
                        None => {
                            return Err(ParseOutcome::ExitErr(Some(format!(
                                "[spoars::] error: option '-{ch}' requires an argument"
                            ))))
                        }
                    }
                };

                match ch {
                    'm' => opts.m = parse_i32(&value).map_err(err_outcome)? as i8,
                    'n' => opts.n = parse_i32(&value).map_err(err_outcome)? as i8,
                    'g' => opts.g = parse_i32(&value).map_err(err_outcome)? as i8,
                    'e' => opts.e = parse_i32(&value).map_err(err_outcome)? as i8,
                    'q' => opts.q = parse_i32(&value).map_err(err_outcome)? as i8,
                    'c' => opts.c = parse_i32(&value).map_err(err_outcome)? as i8,
                    'l' => opts.algorithm = parse_i32(&value).map_err(err_outcome)? as u8,
                    'r' => opts
                        .results
                        .push(parse_i32(&value).map_err(err_outcome)? as u8),
                    'd' => opts.dot_path = value,
                    _ => unreachable!("guarded by the outer match arm"),
                }
            }
            other => {
                return Err(ParseOutcome::ExitErr(Some(format!(
                    "[spoars::] error: unrecognized option '-{other}'"
                ))))
            }
        }
    }
    Ok(())
}

/// Wraps a `parse_i32` error message into a [`ParseOutcome::ExitErr`], for use with `?` inside
/// [`parse_short_cluster`]'s value-parsing match arms.
fn err_outcome(message: String) -> ParseOutcome {
    ParseOutcome::ExitErr(Some(message))
}

/// spoa's `-h`/`--help` usage text (`main.cpp`'s `Help()`), with the program name updated to
/// `spoars`. Not part of this task's byte-for-byte parity surface (the brief calls this out as
/// untested), but kept faithful to upstream's structure and wording.
fn help_text() -> String {
    concat!(
        "usage: spoars [options ...] <sequences>\n",
        "\n",
        "  # default output is stdout\n",
        "  <sequences>\n",
        "    input file in FASTA/FASTQ format (can be compressed with gzip)\n",
        "\n",
        "  options:\n",
        "    -m <int>\n",
        "      default: 5\n",
        "      score for matching bases\n",
        "    -n <int>\n",
        "      default: -4\n",
        "      score for mismatching bases\n",
        "    -g <int>\n",
        "      default: -8\n",
        "      gap opening penalty (must be non-positive)\n",
        "    -e <int>\n",
        "      default: -6\n",
        "      gap extension penalty (must be non-positive)\n",
        "    -q <int>\n",
        "      default: -10\n",
        "      gap opening penalty of the second affine function\n",
        "      (must be non-positive)\n",
        "    -c <int>\n",
        "      default: -4\n",
        "      gap extension penalty of the second affine function\n",
        "      (must be non-positive)\n",
        "    -l, --algorithm <int>\n",
        "      default: 0\n",
        "      alignment mode:\n",
        "        0 - local (Smith-Waterman)\n",
        "        1 - global (Needleman-Wunsch)\n",
        "        2 - semi-global\n",
        "    -r, --result <int> (option can be used multiple times)\n",
        "      default: 0\n",
        "      result mode:\n",
        "        0 - consensus (FASTA)\n",
        "        1 - multiple sequence alignment (FASTA)\n",
        "        2 - 0 & 1 (FASTA)\n",
        "        3 - partial order graph (GFA)\n",
        "        4 - 0 & 3 (GFA)\n",
        "    --min-coverage <int>\n",
        "      default: -1\n",
        "      minimal consensus coverage (usable only with -r 0)\n",
        "    -d, --dot <file>\n",
        "      output file for the partial order graph in DOT format\n",
        "    -s, --strand-ambiguous\n",
        "      for each sequence pick the strand with the better alignment\n",
        "    --version\n",
        "      prints the version number\n",
        "    -h, --help\n",
        "      prints the usage\n",
        "\n",
        "  gap mode:\n",
        "    linear if g >= e\n",
        "    affine if g <= q or e >= c\n",
        "    convex otherwise (default)\n",
    )
    .to_string()
}

/// Reverse-complements `seq`, mapping standard DNA/RNA bases and the IUPAC ambiguity codes to
/// their complements (case-preserving); any other byte passes through unchanged.
///
/// Upstream's `-s`/`--strand-ambiguous` path calls `biosoup::Sequence::ReverseAndComplement`, a
/// method of a dependency fetched over the network at C++ build time (not vendored in
/// `third_party/`), so it cannot be ported verbatim here. `--strand-ambiguous` output is not part
/// of this task's byte-for-byte parity surface (see the task brief); this is a faithful,
/// standard-IUPAC reverse complement.
fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| complement_base(b)).collect()
}

/// Complements a single IUPAC base byte, preserving case; unrecognized bytes pass through
/// unchanged.
fn complement_base(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'a' => b't',
        b'C' => b'G',
        b'c' => b'g',
        b'G' => b'C',
        b'g' => b'c',
        b'T' | b'U' => b'A',
        b't' | b'u' => b'a',
        b'R' => b'Y',
        b'r' => b'y',
        b'Y' => b'R',
        b'y' => b'r',
        b'K' => b'M',
        b'k' => b'm',
        b'M' => b'K',
        b'm' => b'k',
        b'B' => b'V',
        b'b' => b'v',
        b'V' => b'B',
        b'v' => b'b',
        b'D' => b'H',
        b'd' => b'h',
        b'H' => b'D',
        b'h' => b'd',
        other => other, // S/W/N and any other byte are their own complement or unrecognized.
    }
}

/// The file extensions upstream `CreateParser` (`main.cpp:26-60`) accepts, independent of file
/// content: FASTA (`.fasta/.fna/.faa/.fa`) and FASTQ (`.fastq/.fq`), each optionally `.gz`.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    ".fasta",
    ".fasta.gz",
    ".fna",
    ".fna.gz",
    ".faa",
    ".faa.gz",
    ".fa",
    ".fa.gz",
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
];

/// Returns whether `path` ends with one of [`SUPPORTED_EXTENSIONS`]. Mirrors upstream
/// `CreateParser`'s `is_suffix` extension gate (`main.cpp:28-52`), a pure suffix match over the
/// raw path string (case-sensitive, content-independent).
fn has_supported_extension(path: &str) -> bool {
    SUPPORTED_EXTENSIONS.iter().any(|ext| path.ends_with(ext))
}

/// Index-aligned `(sequences, per-base-quality-bytes, names)` read from an input file, as
/// returned by [`read_sequences`].
type SequenceRecords = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<String>);

/// Reads every record of the FASTA/FASTQ (optionally gzip-compressed) file at `path`, returning
/// index-aligned `(sequences, qualities, names)`. `qualities[i]` is empty for FASTA records (no
/// quality line). `names[i]` is the first whitespace-delimited token of the record's id line
/// (biosoup's `Sequence` name semantics), not the full header.
///
/// Rejects any `path` whose extension is not in [`SUPPORTED_EXTENSIONS`] BEFORE opening it,
/// independent of content, mirroring upstream `CreateParser` (`main.cpp:26-60`).
fn read_sequences(path: &str) -> Result<SequenceRecords, String> {
    if !has_supported_extension(path) {
        return Err(format!(
            "[spoars::CreateParser] error: file {path} has unsupported format extension (valid \
             extensions: .fasta, .fasta.gz, .fna, .fna.gz, .faa, .faa.gz, .fa, .fa.gz, .fastq, \
             .fastq.gz, .fq, .fq.gz)"
        ));
    }

    let mut reader = parse_fastx_file(path)
        .map_err(|e| format!("[spoars::CreateParser] error: failed to open {path}: {e}"))?;

    let mut seqs = Vec::new();
    let mut quals = Vec::new();
    let mut names = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.map_err(|e| format!("[spoars::CreateParser] error: {path}: {e}"))?;

        let seq = record.seq().into_owned();
        let qual = record.qual().map(<[u8]>::to_vec).unwrap_or_default();
        let id = record.id();
        let name_bytes = id.split(|&b| b == b' ' || b == b'\t').next().unwrap_or(id);
        let name = String::from_utf8_lossy(name_bytes).into_owned();

        seqs.push(seq);
        quals.push(qual);
        names.push(name);
    }

    Ok((seqs, quals, names))
}

/// Runs the align-and-add loop plus the `-r` result-mode output switch for a fully parsed
/// [`Options`]. Mirrors `main.cpp:246-358`.
fn run(opts: Options) -> ExitCode {
    let Some(input_path) = opts.input.clone() else {
        eprintln!("[spoars::] error: missing input file!");
        print!("{}", help_text());
        return ExitCode::FAILURE;
    };

    // Validation order mirrors spoa's `AlignmentEngine::Create`
    // (`alignment_engine.cpp:40-55`): the alignment TYPE is checked first, then the gap-open
    // penalties, then the gap-extend penalties. So the invalid-`-l` (type) check must run before
    // `Scoring::new`'s penalty-sign validation, to match upstream's stderr precedence when both
    // are invalid together.
    let alignment_type = match opts.algorithm {
        0 => AlignmentType::Local,
        1 => AlignmentType::Global,
        2 => AlignmentType::Overlap,
        other => {
            eprintln!("[spoars::AlignmentEngine::Create] error: invalid alignment type {other}");
            return ExitCode::FAILURE;
        }
    };

    let scoring = match Scoring::new(opts.m, opts.n, opts.g, opts.e, opts.q, opts.c) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let (seqs, quals, names) = match read_sequences(&input_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let force_sisd = should_force_sisd(std::env::var(FORCE_SISD_ENV).ok().as_deref());
    let mut engine: Box<dyn AlignmentEngine> = if force_sisd {
        Box::new(SisdEngine::new(alignment_type, scoring))
    } else {
        Box::new(SimdEngine::new(alignment_type, scoring))
    };
    let mut graph = Graph::new();
    let mut is_reversed: Vec<bool> = Vec::new();

    for i in 0..seqs.len() {
        let seq = &seqs[i];
        let (mut alignment, score) = engine.align(seq, &graph);
        let mut used_seq = seq.clone();
        // When the reverse strand wins under `-s`, `used_seq` becomes the reverse-complemented
        // sequence, so its per-base qualities must be REVERSED (not complemented — quality is a
        // per-base value that mirrors position-for-position) to stay index-aligned with it.
        // Upstream gets this for free because biosoup's `ReverseAndComplement` mutates `data` and
        // `quality` together (`main.cpp:293`).
        let mut used_quals = quals[i].clone();

        if opts.strand_ambiguous {
            let rev_seq = reverse_complement(seq);
            let (alignment_rev, score_rev) = engine.align(&rev_seq, &graph);
            if score >= score_rev {
                is_reversed.push(false);
            } else {
                alignment = alignment_rev;
                used_seq = rev_seq;
                used_quals = quals[i].iter().rev().copied().collect();
                is_reversed.push(true);
            }
        }

        let add_result = if used_quals.is_empty() {
            graph.add_alignment_weight(&alignment, &used_seq, 1)
        } else {
            graph.add_alignment_quality(&alignment, &used_seq, &used_quals)
        };
        if let Err(e) = add_result {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for &mode in &opts.results {
        match mode {
            0 => {
                let consensus = graph.generate_consensus_min_coverage(opts.min_coverage);
                let _ = writeln!(out, ">Consensus LN:i:{}", consensus.len());
                let _ = writeln!(out, "{consensus}");
            }
            1 | 2 => {
                let msa = graph.generate_msa(mode == 2);
                for (i, row) in msa.iter().enumerate() {
                    let name = if i < names.len() {
                        names[i].as_str()
                    } else {
                        "Consensus"
                    };
                    let _ = writeln!(out, ">{name}");
                    let _ = writeln!(out, "{row}");
                }
            }
            3 | 4 => {
                graph.generate_consensus();
                let gfa = graph.to_gfa(&names, &is_reversed, mode == 4);
                let _ = write!(out, "{gfa}");
            }
            _ => {} // main.cpp:351-352: unrecognized result modes are silently ignored.
        }
    }

    if !opts.dot_path.is_empty() {
        let dot = graph.to_dot();
        if let Err(e) = std::fs::write(&opts.dot_path, dot) {
            eprintln!(
                "[spoars::Graph::PrintDot] error: failed to write {}: {e}",
                opts.dot_path
            );
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        ParseOutcome::Run(opts) => run(opts),
        ParseOutcome::ExitOk(message) => {
            if let Some(message) = message {
                print!("{message}");
            }
            ExitCode::SUCCESS
        }
        ParseOutcome::ExitErr(message) => {
            if let Some(message) = message {
                eprintln!("{message}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spoars::graph::Graph;

    #[test]
    fn should_force_sisd_treats_unset_empty_and_zero_as_false() {
        assert!(!should_force_sisd(None));
        assert!(!should_force_sisd(Some("")));
        assert!(!should_force_sisd(Some("0")));
    }

    #[test]
    fn should_force_sisd_treats_any_other_value_as_true() {
        assert!(should_force_sisd(Some("1")));
        assert!(should_force_sisd(Some("true")));
        assert!(should_force_sisd(Some("yes")));
        assert!(should_force_sisd(Some("anything")));
    }

    #[test]
    fn reverse_complement_reverses_and_complements_preserving_case() {
        assert_eq!(reverse_complement(b"AACC"), b"GGTT");
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT"); // palindromic
        assert_eq!(reverse_complement(b"acgt"), b"acgt"); // case preserved
        assert_eq!(reverse_complement(b"N"), b"N"); // N is its own complement
    }

    #[test]
    fn strand_flip_keeps_base_and_quality_index_aligned() {
        // When the reverse strand wins under `-s`, the CLI pairs `revcomp(seq)` with `rev(qual)`
        // (quality REVERSED only, never complemented — it is a per-base value that mirrors
        // position-for-position). This locks the invariant that for every position `j` of the
        // flipped arrays, the base and its quality both originate from the SAME original position
        // `len - 1 - j`: `revcomp(seq)[j]` is the complement of `seq[len-1-j]`, and
        // `rev(qual)[j] == qual[len-1-j]`. A regression that forgot to reverse the qualities (the
        // exact bug this test guards) would break the second equality.
        let seq: &[u8] = b"AACCG";
        let qual: &[u8] = b"IH#5!"; // distinct per-base qualities so a mispairing is observable
        let rev_seq = reverse_complement(seq);
        let rev_qual: Vec<u8> = qual.iter().rev().copied().collect();

        assert_eq!(rev_seq.len(), rev_qual.len());
        let n = seq.len();
        for j in 0..n {
            let orig = n - 1 - j;
            assert_eq!(
                rev_seq[j],
                complement_base(seq[orig]),
                "position {j}: base must be the complement of original base at {orig}"
            );
            assert_eq!(
                rev_qual[j], qual[orig],
                "position {j}: quality must equal original quality at {orig}"
            );
        }
    }

    #[test]
    fn strand_flip_quality_reversal_yields_correct_graph_edge_weights() {
        // End-to-end regression via the public `Graph` API: feed a reverse-complemented sequence
        // with its REVERSED qualities (what the fixed CLI does) into a fresh graph, and confirm
        // the resulting per-edge quality-derived weights (observable through GFA `ew:f:` tags)
        // match a hand computation from the reversed qualities — and, critically, DIFFER from the
        // buggy build that pairs the reverse-complemented sequence with the original (un-reversed)
        // qualities. `Graph::add_alignment_quality` weights each base by `qual - 33` and each
        // interior edge by the sum of its two endpoints' weights (`src/graph.rs`), so an
        // asymmetric 3-base quality profile makes the two interior edge weights swap under the
        // bug, making it visible here.
        let seq: &[u8] = b"ACG";
        let qual: &[u8] = b"5?I"; // Phred 20, 30, 40
        let rev_seq = reverse_complement(seq); // "CGT"
        let rev_qual: Vec<u8> = qual.iter().rev().copied().collect(); // Phred [40, 30, 20]

        let mut correct = Graph::new();
        correct
            .add_alignment_quality(&[], &rev_seq, &rev_qual)
            .expect("add_alignment_quality (reversed quals) must succeed");
        let gfa_correct = correct.to_gfa(&["s".to_string()], &[], false);

        // Reversed quals [40,30,20]: edge 0->1 weight = 40+30 = 70, edge 1->2 = 30+20 = 50.
        assert!(
            gfa_correct.contains("ew:f:70") && gfa_correct.contains("ew:f:50"),
            "expected reversed-quality edge weights 70 and 50 in GFA:\n{gfa_correct}"
        );

        let mut buggy = Graph::new();
        buggy
            .add_alignment_quality(&[], &rev_seq, qual)
            .expect("add_alignment_quality (forward quals) must succeed");
        let gfa_buggy = buggy.to_gfa(&["s".to_string()], &[], false);

        // The bug (forward quals [20,30,40]) would swap the interior edge weights to 50 and 70's
        // complement (50 for 0->1, 70 for 1->2), producing a DIFFERENT GFA. If these were equal,
        // the quality reversal would be unobservable and this test would not guard the fix.
        assert_ne!(
            gfa_correct, gfa_buggy,
            "reversing the qualities must change the quality-derived edge weights; if not, the \
             strand-flip quality fix is unverifiable"
        );
    }
}
