//! Python bindings for spoars (partial order alignment).
//!
//! Exposes a small, Pythonic surface over the Rust crate: a validated [`Scoring`]
//! wrapper and a [`Poa`] builder that aligns sequences with the SIMD engine and
//! produces a consensus, MSA, GFA, or DOT — plus a one-call [`poa`] convenience.
//! The heavy lifting (and bit-exact-with-spoa guarantee) lives in the `spoars`
//! crate; this is a thin, safe adapter.

// pyo3's #[pymethods]/#[pyfunction] macros emit an identity `PyErr -> PyErr`
// conversion on the return type of fallible functions, which clippy 1.95+ flags
// as `useless_conversion` pointing at macro-generated code we cannot edit. Allow
// it crate-wide; it is not triggered by any hand-written conversion here.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use spoars::align::{AlignmentEngine, AlignmentType, GapMode, Scoring as RsScoring, SimdEngine};
use spoars::graph::Graph;

/// Maps an alignment-type name (case-insensitive `"global"`/`"local"`/`"overlap"`,
/// or the spoa aliases `"nw"`/`"sw"`/`"ov"`) to the Rust enum.
fn parse_alignment_type(name: &str) -> PyResult<AlignmentType> {
    match name.to_ascii_lowercase().as_str() {
        "global" | "nw" => Ok(AlignmentType::Global),
        "local" | "sw" => Ok(AlignmentType::Local),
        "overlap" | "ov" => Ok(AlignmentType::Overlap),
        other => Err(PyValueError::new_err(format!(
            "unknown alignment_type {other:?}; expected 'global', 'local', or 'overlap'"
        ))),
    }
}

/// Validated match/mismatch/gap scoring, mirroring `spoars::align::Scoring`.
///
/// Positive gap penalties are rejected (spoa's sign convention). The gap model
/// (linear/affine/convex) is inferred from the penalties; see :meth:`gap_mode`.
//
// `from_py_object`: opt in to the `FromPyObject` derive so `Scoring` can be passed
// by value as a function argument (pyo3 0.29 made this opt-in for `Clone` pyclasses).
#[pyclass(module = "spoars._spoars", frozen, from_py_object)]
#[derive(Clone, Copy)]
struct Scoring {
    inner: RsScoring,
}

#[pymethods]
impl Scoring {
    /// `Scoring(match, mismatch, gap_open, gap_extend, gap_open2, gap_extend2)`.
    ///
    /// Raises `ValueError` if any gap-open or gap-extend penalty is positive.
    #[new]
    #[pyo3(signature = (r#match, mismatch, gap_open, gap_extend, gap_open2, gap_extend2))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        r#match: i8,
        mismatch: i8,
        gap_open: i8,
        gap_extend: i8,
        gap_open2: i8,
        gap_extend2: i8,
    ) -> PyResult<Self> {
        let inner = RsScoring::new(
            r#match,
            mismatch,
            gap_open,
            gap_extend,
            gap_open2,
            gap_extend2,
        )
        .map_err(|e| PyValueError::new_err(format!("invalid scoring: {e:?}")))?;
        Ok(Self { inner })
    }

    /// The spoa/CLI default convex scoring: `m=5, n=-4, g=-8, e=-6, q=-10, c=-4`.
    #[staticmethod]
    fn default() -> Self {
        // These constants are spoa's defaults and always satisfy `Scoring::new`.
        Self {
            inner: RsScoring::new(5, -4, -8, -6, -10, -4).expect("spoa defaults are valid"),
        }
    }

    /// The inferred gap model: `"linear"`, `"affine"`, or `"convex"`.
    fn gap_mode(&self) -> &'static str {
        match self.inner.gap_mode() {
            GapMode::Linear => "linear",
            GapMode::Affine => "affine",
            GapMode::Convex => "convex",
        }
    }

    fn __repr__(&self) -> String {
        format!("Scoring(gap_mode={:?})", self.gap_mode())
    }
}

/// A partial order alignment graph builder.
///
/// Construct with an alignment type and scoring, add sequences one at a time
/// (each is aligned against the graph so far with the SIMD engine and merged in),
/// then read off the consensus, MSA, GFA, or DOT.
///
/// `unsendable`: a `Poa` owns SIMD scratch buffers and is not shared across
/// threads; it is used under Python's GIL from a single thread.
#[pyclass(module = "spoars._spoars", unsendable)]
struct Poa {
    graph: Graph,
    engine: SimdEngine,
}

#[pymethods]
impl Poa {
    /// `Poa(alignment_type="global", scoring=None)`.
    ///
    /// `scoring` defaults to :meth:`Scoring.default`. `alignment_type` is one of
    /// `"global"`, `"local"`, `"overlap"` (case-insensitive).
    #[new]
    #[pyo3(signature = (alignment_type="global", scoring=None))]
    fn new(alignment_type: &str, scoring: Option<Scoring>) -> PyResult<Self> {
        let alignment_type = parse_alignment_type(alignment_type)?;
        let scoring = scoring.unwrap_or_else(Scoring::default);
        Ok(Self {
            graph: Graph::new(),
            engine: SimdEngine::new(alignment_type, scoring.inner),
        })
    }

    /// Align `sequence` against the current graph and merge it in with `weight`,
    /// returning the assigned sequence index (0-based, in insertion order).
    #[pyo3(signature = (sequence, weight=1))]
    fn add(&mut self, sequence: &str, weight: u32) -> PyResult<usize> {
        let seq = sequence.as_bytes();
        let index = self.graph.sequence_starts().len();
        let (alignment, _score) = self.engine.align(seq, &self.graph);
        self.graph
            .add_alignment_weight(&alignment, seq, weight)
            .map_err(|e| PyValueError::new_err(format!("add failed: {e:?}")))?;
        Ok(index)
    }

    /// The consensus sequence. With `min_coverage`, nodes below that coverage are
    /// pruned from the consensus path (`generate_consensus_min_coverage`).
    #[pyo3(signature = (min_coverage=None))]
    fn consensus(&mut self, min_coverage: Option<i32>) -> String {
        match min_coverage {
            Some(min_coverage) => self.graph.generate_consensus_min_coverage(min_coverage),
            None => self.graph.generate_consensus(),
        }
    }

    /// The multiple sequence alignment, one row per added sequence (optionally
    /// with a trailing consensus row).
    #[pyo3(signature = (include_consensus=false))]
    fn msa(&mut self, include_consensus: bool) -> Vec<String> {
        self.graph.generate_msa(include_consensus)
    }

    /// The graph in GFA v1 format. `headers` (one per sequence) and `is_reversed`
    /// default to `["0", "1", ...]` and all-`False`; if given, their lengths must
    /// equal the number of added sequences.
    #[pyo3(signature = (headers=None, is_reversed=None, include_consensus=false))]
    fn gfa(
        &self,
        headers: Option<Vec<String>>,
        is_reversed: Option<Vec<bool>>,
        include_consensus: bool,
    ) -> PyResult<String> {
        let n = self.graph.sequence_starts().len();
        let headers = headers.unwrap_or_else(|| (0..n).map(|i| i.to_string()).collect());
        let is_reversed = is_reversed.unwrap_or_else(|| vec![false; n]);
        if headers.len() != n || is_reversed.len() != n {
            return Err(PyValueError::new_err(format!(
                "headers ({}) and is_reversed ({}) must each have one entry per sequence ({n})",
                headers.len(),
                is_reversed.len(),
            )));
        }
        Ok(self.graph.to_gfa(&headers, &is_reversed, include_consensus))
    }

    /// The graph in Graphviz DOT format.
    fn dot(&self) -> String {
        self.graph.to_dot()
    }

    /// Number of nodes in the graph.
    fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Number of edges in the graph.
    fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    /// Number of sequences added to the graph.
    fn num_sequences(&self) -> usize {
        self.graph.sequence_starts().len()
    }

    fn __len__(&self) -> usize {
        self.graph.sequence_starts().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Poa(num_sequences={}, num_nodes={})",
            self.graph.sequence_starts().len(),
            self.graph.num_nodes()
        )
    }
}

/// Build a POA graph from `sequences` in one call and return the [`Poa`].
///
/// `weights`, if given, must have one entry per sequence (default weight 1).
#[pyfunction]
#[pyo3(signature = (sequences, alignment_type="global", scoring=None, weights=None))]
fn poa(
    sequences: Vec<String>,
    alignment_type: &str,
    scoring: Option<Scoring>,
    weights: Option<Vec<u32>>,
) -> PyResult<Poa> {
    if let Some(weights) = &weights {
        if weights.len() != sequences.len() {
            return Err(PyValueError::new_err(format!(
                "weights ({}) must have one entry per sequence ({})",
                weights.len(),
                sequences.len(),
            )));
        }
    }
    let mut graph = Poa::new(alignment_type, scoring)?;
    for (i, sequence) in sequences.iter().enumerate() {
        let weight = weights.as_ref().map_or(1, |w| w[i]);
        graph.add(sequence, weight)?;
    }
    Ok(graph)
}

/// The `_spoars` extension module.
#[pymodule]
fn _spoars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scoring>()?;
    m.add_class::<Poa>()?;
    m.add_function(wrap_pyfunction!(poa, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
