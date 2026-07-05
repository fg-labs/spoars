//! Subprocess wrapper around the C++ differential-testing oracle
//! (`oracle/spoa_oracle.cpp`, built by `oracle/CMakeLists.txt`).
//!
//! The oracle is a dev/CI-only helper that links the pinned upstream `spoa`
//! submodule forced to its SISD (scalar) alignment engine and reproduces the
//! exact sequence of `spoa::AlignmentEngine` / `spoa::Graph` calls that
//! upstream's own CLI performs. Feeding it JSONL cases on stdin and reading
//! JSONL results back on stdout lets the Rust reimplementation's test suite
//! assert parity against upstream spoa without linking spoa into this crate.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// spoa's alignment mode, selecting which of `spoa::AlignmentType`'s three
/// dynamic-programming recurrences the oracle's `AlignmentEngine` uses.
/// Serializes to/from the oracle's `"SW"` / `"NW"` / `"OV"` string contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignType {
    #[serde(rename = "SW")]
    Sw,
    #[serde(rename = "NW")]
    Nw,
    #[serde(rename = "OV")]
    Ov,
}

/// spoa's CLI-default alignment scores (see `third_party/spoa/src/main.cpp`):
/// match=5, mismatch=-4, gap-open=-8, gap-extend=-6, second gap-open=-10,
/// second gap-extend=-4. `OracleCase`'s constructors default to these so
/// callers only need to override what a given test actually varies.
const DEFAULT_M: i8 = 5;
const DEFAULT_N: i8 = -4;
const DEFAULT_G: i8 = -8;
const DEFAULT_E: i8 = -6;
const DEFAULT_Q: i8 = -10;
const DEFAULT_C: i8 = -4;
const DEFAULT_MIN_COVERAGE: i32 = -1;

/// One request line in the oracle's JSONL contract: an alignment type, a
/// full set of affine-gap scoring parameters, and the sequences (and
/// optional per-base Phred qualities) to align and fold into a POA graph.
#[derive(Debug, Clone, Serialize)]
pub struct OracleCase {
    pub id: u32,
    #[serde(rename = "type")]
    pub ty: AlignType,
    pub m: i8,
    pub n: i8,
    pub g: i8,
    pub e: i8,
    pub q: i8,
    pub c: i8,
    pub seqs: Vec<String>,
    pub quals: Option<Vec<String>>,
    pub min_coverage: i32,
    /// Optional GFA `P`-line sequence names, index-aligned with `seqs`. When present, the
    /// oracle uses these as `PrintGfa`'s `headers` argument instead of falling back to
    /// `0..seqs.len()` decimal placeholders; DOT output is unaffected (it labels nodes by id,
    /// not by sequence name).
    pub names: Option<Vec<String>>,
}

/// Monotonic source of default `OracleCase` ids. `defaulted` draws from this
/// so a batch built via the constructors (e.g.
/// `vec![OracleCase::nw(..), OracleCase::sw(..)]`) gets distinct, increasing
/// ids without the caller having to assign them — which is exactly how the
/// parity tests build batches, and what `run_oracle`'s id-keyed result
/// correlation relies on. Callers may still overwrite `id` explicitly.
static NEXT_CASE_ID: AtomicU32 = AtomicU32::new(0);

impl OracleCase {
    /// Builds a case with spoa's CLI-default scores, `min_coverage: -1`, and a
    /// distinct auto-assigned `id` drawn from a process-wide counter; callers
    /// typically use `nw`/`sw`/`ov`/`with_quals` instead of this directly, but
    /// it's exposed for constructing custom score sweeps.
    fn defaulted(ty: AlignType, seqs: &[&str]) -> Self {
        OracleCase {
            id: NEXT_CASE_ID.fetch_add(1, Ordering::Relaxed),
            ty,
            m: DEFAULT_M,
            n: DEFAULT_N,
            g: DEFAULT_G,
            e: DEFAULT_E,
            q: DEFAULT_Q,
            c: DEFAULT_C,
            seqs: seqs.iter().map(|s| s.to_string()).collect(),
            quals: None,
            min_coverage: DEFAULT_MIN_COVERAGE,
            names: None,
        }
    }

    /// A Needleman-Wunsch (global) alignment case with default scores.
    pub fn nw(seqs: &[&str]) -> Self {
        Self::defaulted(AlignType::Nw, seqs)
    }

    /// A Smith-Waterman (local) alignment case with default scores.
    pub fn sw(seqs: &[&str]) -> Self {
        Self::defaulted(AlignType::Sw, seqs)
    }

    /// An overlap alignment case with default scores.
    pub fn ov(seqs: &[&str]) -> Self {
        Self::defaulted(AlignType::Ov, seqs)
    }

    /// A case forcing spoa's **linear** gap mode (`g == e`, so
    /// `AlignmentEngine::Create` classifies it as `kLinear`), for the given
    /// alignment type. Keeps the default match/mismatch scores (`m = 5`,
    /// `n = -4`) and sets `g = e = -8`; `q`/`c` are irrelevant to a linear
    /// engine (they're never read once the subtype is `kLinear`) but are set
    /// to valid non-positive values (`-8`).
    pub fn linear(ty: AlignType, seqs: &[String]) -> Self {
        let refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
        let mut case = Self::defaulted(ty, &refs);
        case.g = -8;
        case.e = -8;
        case.q = -8;
        case.c = -8;
        case
    }

    /// A case forcing spoa's **affine** gap mode (`g < e` and `g <= q`, so
    /// `AlignmentEngine::Create` classifies it as `kAffine`), for the given
    /// alignment type: `g = -8`, `e = -6`, `q = -8`, `c = -6`. The DP fill for
    /// this mode lands in Task 12; the constructor is provided now so that
    /// task's parity test can reuse this harness.
    pub fn affine(ty: AlignType, seqs: &[String]) -> Self {
        let refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
        let mut case = Self::defaulted(ty, &refs);
        case.g = -8;
        case.e = -6;
        case.q = -8;
        case.c = -6;
        case
    }

    /// A case forcing spoa's **convex** gap mode (spoa's CLI defaults,
    /// `g = -8`, `e = -6`, `q = -10`, `c = -4`, classify as `kConvex`), for the
    /// given alignment type. The DP fill for this mode lands in Task 13; the
    /// constructor is provided now so that task's parity test can reuse this
    /// harness.
    pub fn convex(ty: AlignType, seqs: &[String]) -> Self {
        let refs: Vec<&str> = seqs.iter().map(|s| s.as_str()).collect();
        let mut case = Self::defaulted(ty, &refs);
        case.g = -8;
        case.e = -6;
        case.q = -10;
        case.c = -4;
        case
    }

    /// A case carrying per-base Phred quality strings alongside `seqs`,
    /// exercising spoa's `q-33` weighting path in `Graph::AddAlignment`.
    /// `quals` must be the same length as `seqs`, one quality string per
    /// sequence.
    pub fn with_quals(ty: AlignType, seqs: &[&str], quals: &[&str]) -> Self {
        assert_eq!(
            seqs.len(),
            quals.len(),
            "with_quals: seqs and quals must have the same length"
        );
        let mut case = Self::defaulted(ty, seqs);
        case.quals = Some(quals.iter().map(|q| q.to_string()).collect());
        case
    }
}

/// One JSONL result line from the oracle: per-sequence alignment traces
/// (`(graph_node_id, query_index)` pairs, `-1` marking a gap on either side),
/// the generated consensus, the multiple sequence alignment, and GFA/DOT
/// graph dumps.
#[derive(Debug, Clone, Deserialize)]
pub struct OracleResult {
    pub id: u32,
    pub alignments: Vec<Vec<(i32, i32)>>,
    pub consensus: String,
    pub msa: Vec<String>,
    pub gfa: String,
    pub dot: String,
}

/// Locates the crate root regardless of the test binary's working directory.
fn crate_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Builds `oracle/build/spoa_oracle` via CMake configure + build the first time
/// `run_oracle` is called, and reuses it thereafter.
///
/// This must be safe under `cargo nextest`, which runs **each test in its own
/// process** — so a per-process `Once` is not enough: many test processes would
/// otherwise `cmake` into the same `oracle/build/` concurrently and corrupt each
/// other. Cross-process coordination uses an atomic lock directory
/// (`create_dir` is atomic on every platform): exactly one process builds while
/// the others wait for it to finish, and once the binary exists every process
/// takes the fast path and skips CMake entirely.
fn ensure_oracle_built() {
    static BUILD_ONCE: Once = Once::new();
    BUILD_ONCE.call_once(|| {
        let root = crate_root();
        let build_dir = root.join("oracle/build");
        let binary = build_dir.join("spoa_oracle");
        let lock = build_dir.join(".spoars-oracle-build.lock");
        std::fs::create_dir_all(&build_dir).expect("failed to create oracle/build");

        // Generous cap for a cold build on a slow CI runner; if the lock is
        // still held past this, we assume the holder crashed and reclaim it.
        let deadline = Instant::now() + Duration::from_secs(600);
        loop {
            // Fast path: already built (by us earlier, or by another process).
            if binary.exists() {
                return;
            }
            match std::fs::create_dir(&lock) {
                Ok(()) => {
                    // We hold the build lock. Build, then release it even on
                    // failure so a broken build never wedges other processes.
                    let outcome = build_oracle(&root);
                    let _ = std::fs::remove_dir(&lock);
                    outcome.unwrap_or_else(|message| panic!("{message}"));
                    return;
                }
                Err(_) => {
                    // Another process is building; wait for it to release the
                    // lock, then loop (fast path returns once the binary exists).
                    while lock.exists() && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    if lock.exists() {
                        // Presumed-stale lock from a crashed builder: reclaim it.
                        let _ = std::fs::remove_dir(&lock);
                    }
                }
            }
        }
    });
}

/// Runs `cmake` configure + build for `oracle/`. Returns `Err(message)` on
/// failure so the caller can release the build lock before propagating.
fn build_oracle(root: &std::path::Path) -> Result<(), String> {
    let configure = Command::new("cmake")
        .args([
            "-S",
            "oracle",
            "-B",
            "oracle/build",
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .current_dir(root)
        .status()
        .map_err(|e| format!("failed to spawn cmake configure for oracle/: {e}"))?;
    if !configure.success() {
        return Err(format!(
            "cmake configure for oracle/ failed (exit status: {configure})"
        ));
    }

    let build = Command::new("cmake")
        .args(["--build", "oracle/build", "-j"])
        .current_dir(root)
        .status()
        .map_err(|e| format!("failed to spawn cmake build for oracle/: {e}"))?;
    if !build.success() {
        return Err(format!(
            "cmake build for oracle/ failed (exit status: {build})"
        ));
    }
    Ok(())
}

/// The oracle child process, plus a mutex serializing access to its stdio.
/// Held for the lifetime of the test process: spawning the oracle once and
/// reusing it across `run_oracle` calls avoids paying the process-spawn and
/// spoa-graph-teardown cost per call.
struct OracleProcess {
    child: Mutex<Child>,
}

static ORACLE_PROCESS: OnceLock<OracleProcess> = OnceLock::new();

fn oracle_process() -> &'static OracleProcess {
    ORACLE_PROCESS.get_or_init(|| {
        ensure_oracle_built();

        let binary = crate_root().join("oracle/build/spoa_oracle");
        assert!(
            binary.exists(),
            "expected oracle binary at {}, but it does not exist after building",
            binary.display()
        );

        let child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn oracle binary {}: {e}", binary.display()));

        OracleProcess {
            child: Mutex::new(child),
        }
    })
}

/// Runs `cases` through the C++ oracle and returns one [`OracleResult`] per
/// case, sorted by `id`.
///
/// Builds the oracle on first use (see [`ensure_oracle_built`]), then spawns
/// it once per test process and reuses that child for every call. All cases
/// are written as JSONL to the child's stdin in one batch, and results are
/// read back as JSONL from stdout.
///
/// The oracle is FAIL-FAST: on the first malformed/exception case it logs to
/// stderr and exits non-zero, and JSONL lines for prior cases in the batch
/// may already be flushed to stdout. Tests only ever feed this harness valid
/// generator output, so a short read or non-zero exit here is a tripwire —
/// not a recoverable condition — and this function panics with a clear
/// message rather than returning a partial/short result vector.
pub fn run_oracle(cases: &[OracleCase]) -> Vec<OracleResult> {
    let process = oracle_process();
    let mut child = process
        .child
        .lock()
        .unwrap_or_else(|e| panic!("oracle child process mutex poisoned: {e}"));

    // The oracle is a streaming server: it flushes one JSONL result per input
    // line. Writing the whole batch to stdin and only *then* reading stdout,
    // single-threaded, deadlocks once either pipe fills its OS buffer (~16-64
    // KiB) — the child blocks writing results the parent isn't draining while
    // the parent blocks writing cases the child isn't consuming. So write
    // stdin from a dedicated thread while the main thread concurrently drains
    // stdout. Disjoint-field borrows let stdin move into the writer thread and
    // stdout stay on the main thread without taking (and losing) either handle
    // from the reused child.
    let child = &mut *child;
    let stdin = child
        .stdin
        .as_mut()
        .expect("oracle child process stdin was not piped");
    let stdout = child
        .stdout
        .as_mut()
        .expect("oracle child process stdout was not piped");
    let mut reader = BufReader::new(stdout);

    let mut results = Vec::with_capacity(cases.len());
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for case in cases {
                let line = serde_json::to_string(case).unwrap_or_else(|e| {
                    panic!("failed to serialize oracle case id={}: {e}", case.id)
                });
                // A write error here means the child closed stdin early — i.e.
                // it hit its fail-fast path and exited. Stop writing and let
                // the main thread's count-match assertion below surface the
                // clean diagnostic, rather than panicking with a raw
                // BrokenPipe that would mask it.
                if writeln!(stdin, "{line}")
                    .and_then(|()| stdin.flush())
                    .is_err()
                {
                    break;
                }
            }
        });

        for _ in 0..cases.len() {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .expect("failed to read a result line from oracle child stdout");
            if n == 0 {
                break; // EOF: the oracle exited early (fail-fast); handled below.
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            let result: OracleResult = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                panic!("failed to parse oracle result line as JSON: {e}\nline: {trimmed}")
            });
            results.push(result);
        }
    });

    assert_eq!(
        results.len(),
        cases.len(),
        "oracle returned {} result(s) for {} case(s) submitted; the oracle is \
         fail-fast, so a short count means it errored out partway through \
         this batch (check stderr output above for the cause)",
        results.len(),
        cases.len()
    );

    // The oracle process is reused across calls, so an in-batch failure that
    // hasn't yet made the child exit wouldn't be caught by the count check
    // above alone; poll its exit status without blocking indefinitely.
    if let Ok(Some(status)) = child.try_wait() {
        assert!(
            status.success(),
            "oracle child process exited with {status} after processing this batch \
             (fail-fast: it logs the offending case to stderr before exiting)"
        );
    }

    results.sort_by_key(|r| r.id);
    results
}
