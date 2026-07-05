//! Alignment engine scaffolding: alignment/gap-mode enums, validated scoring, the alignment
//! sentinel type, and the [`AlignmentEngine`] trait.
//!
//! Mirrors `spoa::AlignmentEngine` (`third_party/spoa/include/spoa/alignment_engine.hpp` and
//! `third_party/spoa/src/alignment_engine.cpp`). This module holds the scaffolding shared by every
//! gap mode: [`AlignmentType`], [`GapMode`], [`Scoring`] (validated + normalized match/mismatch/gap
//! penalties), the [`Alignment`] sentinel type, and the [`AlignmentEngine`] trait itself. The DP
//! fill and backtrack for each gap mode (linear/affine/convex) live in the [`sisd`] (scalar) engine
//! and its bit-identical [`simd`] counterpart.

mod backtrack;
pub mod simd;
pub mod sisd;

pub use simd::SimdEngine;
pub use sisd::SisdEngine;

use crate::graph::{Graph, GraphError, NodeId};

/// The three alignment modes spoa supports.
///
/// Mirrors spoa's `AlignmentType` enum (`alignment_type.hpp:16-20`): `kSW` (Smith-Waterman,
/// local), `kNW` (Needleman-Wunsch, global), `kOV` (semi-global/overlap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentType {
    /// Smith-Waterman local alignment (`kSW`).
    Local,
    /// Needleman-Wunsch global alignment (`kNW`).
    Global,
    /// Semi-global (overlap) alignment (`kOV`).
    Overlap,
}

/// The three gap-penalty models spoa supports.
///
/// Mirrors spoa's `AlignmentSubtype` enum (`alignment_type.hpp:22-26`): linear (`g * i`), affine
/// (`g + (i - 1) * e`), and convex (piecewise minimum of two affine models,
/// `min(g + (i - 1) * e, q + (i - 1) * c)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapMode {
    /// Linear gap penalty: `g * i`.
    Linear,
    /// Affine gap penalty: `g + (i - 1) * e`.
    Affine,
    /// Convex (double-affine) gap penalty: `min(g + (i - 1) * e, q + (i - 1) * c)`.
    Convex,
}

/// Error returned by [`Scoring::new`] when the supplied penalties violate spoa's sign
/// invariants.
///
/// Mirrors the two `std::invalid_argument` throws in `spoa::AlignmentEngine::Create`
/// (`alignment_engine.cpp:46-55`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringError {
    /// `g > 0 || q > 0` (`alignment_engine.cpp:46-50`).
    GapOpenPositive,
    /// `e > 0 || c > 0` (`alignment_engine.cpp:51-55`).
    GapExtendPositive,
}

impl std::fmt::Display for ScoringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            ScoringError::GapOpenPositive => "gap opening penalty must be non-positive",
            ScoringError::GapExtendPositive => "gap extension penalty must be non-positive",
        };
        write!(f, "[spoars::Scoring::new] error: {message}")
    }
}

impl std::error::Error for ScoringError {}

/// Validated, normalized match/mismatch/gap scoring penalties.
///
/// Mirrors the `m_, n_, g_, e_, q_, c_` fields threaded through `spoa::AlignmentEngine::Create`
/// (`alignment_engine.cpp:32-72`): `m` = match score, `n` = mismatch score, `g`/`e` = first
/// gap-open/gap-extend penalty, `q`/`c` = second (convex) gap-open/gap-extend penalty.
///
/// [`Scoring::new`] validates the non-positive-penalty invariants
/// (`alignment_engine.cpp:46-55`) and then **normalizes** `e`/`q`/`c` per the gap mode
/// (`alignment_engine.cpp:61-66`), so that every downstream reader of a `Scoring` value (in
/// particular the DP fill in later tasks) can use `e`/`q`/`c` directly without re-deriving the
/// gap mode first:
/// - [`GapMode::Linear`]: `e` is overwritten with `g` (gap cost is `g` per base, extend == open).
/// - [`GapMode::Affine`]: `q` is overwritten with `g` and `c` with `e` (the second affine layer
///   collapses onto the first).
/// - [`GapMode::Convex`]: all four gap fields are left as supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scoring {
    /// Match score (spoa's `m_`).
    pub m: i8,
    /// Mismatch score (spoa's `n_`).
    pub n: i8,
    /// First gap-open penalty (spoa's `g_`).
    pub g: i8,
    /// First gap-extend penalty (spoa's `e_`); normalized to equal `g` under
    /// [`GapMode::Linear`].
    pub e: i8,
    /// Second (convex) gap-open penalty (spoa's `q_`); normalized to equal `g` under
    /// [`GapMode::Affine`].
    pub q: i8,
    /// Second (convex) gap-extend penalty (spoa's `c_`); normalized to equal `e` under
    /// [`GapMode::Affine`].
    pub c: i8,
}

impl Scoring {
    /// Validates and normalizes `(m, n, g, e, q, c)` into a [`Scoring`].
    ///
    /// Mirrors `spoa::AlignmentEngine::Create(type, m, n, g, e, q, c)`
    /// (`alignment_engine.cpp:32-72`), minus the `type`/`AlignmentType` validation (that lives on
    /// [`AlignmentType`] itself in this port, which is a closed Rust enum and so cannot hold an
    /// invalid discriminant).
    ///
    /// # Errors
    ///
    /// Returns [`ScoringError::GapOpenPositive`] if `g > 0 || q > 0`, or
    /// [`ScoringError::GapExtendPositive`] if `e > 0 || c > 0` (`alignment_engine.cpp:46-55`).
    pub fn new(m: i8, n: i8, g: i8, e: i8, q: i8, c: i8) -> Result<Scoring, ScoringError> {
        if g > 0 || q > 0 {
            return Err(ScoringError::GapOpenPositive);
        }
        if e > 0 || c > 0 {
            return Err(ScoringError::GapExtendPositive);
        }

        // Normalization (alignment_engine.cpp:61-66). Computed on the raw (pre-normalization)
        // inputs, exactly as upstream does before it overwrites e/q/c in place.
        let (g, e, q, c) = match Self::classify(g, e, q, c) {
            GapMode::Linear => (g, g, q, c),
            GapMode::Affine => (g, e, g, e),
            GapMode::Convex => (g, e, q, c),
        };

        Ok(Scoring { m, n, g, e, q, c })
    }

    /// The spoa/CLI default scoring: `m=5, n=-4, g=-8, e=-6, q=-10, c=-4` (a [`GapMode::Convex`]
    /// model). A named preset so callers need not hardcode the magic numbers.
    pub fn spoa_default() -> Scoring {
        // These constants satisfy `Scoring::new`'s sign invariants, so the unwrap cannot fail.
        Scoring::new(5, -4, -8, -6, -10, -4).expect("spoa default scoring is valid")
    }

    /// Classifies `(g, e, q, c)` into a [`GapMode`].
    ///
    /// Ports `alignment_engine.cpp:57-59` EXACTLY:
    /// `g >= e ? Linear : (g <= q || e >= c ? Affine : Convex)`.
    fn classify(g: i8, e: i8, q: i8, c: i8) -> GapMode {
        if g >= e {
            GapMode::Linear
        } else if g <= q || e >= c {
            GapMode::Affine
        } else {
            GapMode::Convex
        }
    }

    /// Returns this scoring's [`GapMode`].
    ///
    /// Re-derives the classification from `self`'s (already-normalized) fields via
    /// [`Scoring::classify`]. This is safe because normalization is a fixed point of
    /// classification: e.g. under [`GapMode::Linear`] normalization sets `e = g`, and
    /// `classify(g, g, q, c)` still evaluates `g >= e` (now `g >= g`) as `true`, so it still
    /// returns [`GapMode::Linear`]; the same holds for [`GapMode::Affine`]'s `q = g, c = e`
    /// normalization and [`GapMode::Convex`]'s no-op normalization.
    pub fn gap_mode(&self) -> GapMode {
        Self::classify(self.g, self.e, self.q, self.c)
    }

    /// The worst-case (most negative) score reachable when aligning a length-`i` sequence
    /// against a length-`j` graph path, used as an `i32` DP-cell overflow guard.
    ///
    /// Ports `spoa::AlignmentEngine::WorstCaseAlignmentScore` (`alignment_engine.cpp:101-110`)
    /// EXACTLY, including its local `gap_score` closure. Computed in `i64` (matching upstream's
    /// `std::int64_t`) to safely evaluate the products before the caller compares the result
    /// against `NEG_INF` cast up to `i64`.
    pub fn worst_case_alignment_score(&self, i: i64, j: i64) -> i64 {
        let m = i64::from(self.m);
        let g = i64::from(self.g);
        let e = i64::from(self.e);
        let q = i64::from(self.q);
        let c = i64::from(self.c);
        let gap_score = |len: i64| -> i64 {
            if len == 0 {
                0
            } else {
                (g + (len - 1) * e).min(q + (len - 1) * c)
            }
        };
        (-(m * i.min(j) + gap_score((i - j).abs()))).min(gap_score(i) + gap_score(j))
    }
}

/// A graph-to-sequence alignment: a sequence of `(graph_node_index, seq_index)` pairs.
///
/// `-1` in either slot is spoa's "no match" sentinel: `-1` for the node index marks an insertion
/// (a sequence base with no graph counterpart yet), and `-1` for the seq index marks a deletion
/// (an existing graph node with no counterpart in this sequence). This is the same sentinel form
/// accepted by [`crate::graph::Graph::add_alignment`]. Mirrors `spoa::Alignment`
/// (`alignment_engine.hpp:29`, `using Alignment = std::vector<std::pair<int32_t, int32_t>>`).
///
/// A more ergonomic `Option`-based wrapper is left to a later CLI/API task; this is the faithful,
/// zero-cost sentinel form the DP engines themselves produce and consume.
pub type Alignment = Vec<(i32, i32)>;
// NOTE: the `Alignment` element convention — `(node_index, sequence_index)` with `-1` sentinels —
// is the contract shared by every `AlignmentEngine` and by `Graph::add_alignment`; it is documented
// in full on the `AlignmentEngine` trait below (the extension point) so implementers see it there.

/// A pairwise sequence-to-graph alignment engine — the crate's extension point.
///
/// Mirrors `spoa::AlignmentEngine::Align` (`alignment_engine.hpp:61-69`): aligns `seq` against
/// `graph`, returning the alignment path and its score. The two engines this crate ships —
/// [`SisdEngine`] (portable scalar) and [`SimdEngine`] (SSE4.1/AVX2/NEON with a scalar fallback,
/// bit-identical to `SisdEngine`) — both implement it, and downstream crates can implement it too
/// (e.g. a different scoring model, a banded fill, or a mock for testing) and feed the result
/// straight into [`Graph::add_alignment`].
///
/// # The alignment format
///
/// The returned [`Alignment`] is a `Vec<(node_index, sequence_index)>` walked from the start of the
/// alignment to its end. Each pair encodes one column of the pairwise alignment between `graph` and
/// `seq`, using `-1` as a sentinel:
///
/// - `(n, s)` with `n >= 0, s >= 0` — sequence position `s` is **matched/substituted** onto graph
///   node [`NodeId`](crate::graph::NodeId)`(n as u32)`.
/// - `(-1, s)` with `s >= 0` — sequence position `s` is an **insertion** (no graph node); a fresh
///   node is created for it.
/// - `(n, -1)` with `n >= 0` — graph node `n` is a **deletion** (skipped by this sequence).
///
/// `(-1, -1)` never appears. `sequence_index` values must be strictly the positions of `seq`
/// (`0..seq.len()`); [`Graph::add_alignment`] validates this and returns
/// [`GraphError::InvalidAlignment`](crate::graph::GraphError) otherwise. This is exactly upstream
/// spoa's format, so the alignment an engine produces is consumed unchanged by graph construction.
///
/// # Score
///
/// The `i32` score is the optimal alignment score under the engine's model. It is informational for
/// graph construction (`add_alignment` ignores it); callers use it for filtering/reporting.
///
/// # Implementing a custom engine
///
/// A minimal engine that treats every sequence as entirely new (empty alignment ⇒
/// [`Graph::add_alignment`] appends the whole sequence as a fresh disjoint path):
///
/// ```
/// use spoars::align::{Alignment, AlignmentEngine};
/// use spoars::graph::Graph;
///
/// struct AppendAsNew;
///
/// impl AlignmentEngine for AppendAsNew {
///     fn align(&mut self, _seq: &[u8], _graph: &Graph) -> (Alignment, i32) {
///         (Vec::new(), 0) // empty alignment: nothing is matched onto the existing graph
///     }
/// }
///
/// let mut engine = AppendAsNew;
/// let mut graph = Graph::new();
/// for read in [b"ACGT".as_slice(), b"ACGT".as_slice()] {
///     let (alignment, _score) = engine.align(read, &graph);
///     graph.add_alignment_weight(&alignment, read, 1).unwrap();
/// }
/// // Two disjoint chains of 4 nodes each, since nothing was aligned together.
/// assert_eq!(graph.num_nodes(), 8);
/// ```
///
/// The align-then-add loop above is exactly how [`SisdEngine`]/[`SimdEngine`] are driven; swapping
/// in a real engine (one that returns non-empty alignments) is what makes sequences merge into a
/// consensus DAG.
pub trait AlignmentEngine {
    /// Aligns `seq` against `graph`, returning `(alignment, score)` in the format documented on the
    /// [`AlignmentEngine`] trait. `&mut self` lets an engine reuse internal scratch buffers across
    /// calls (as both shipped engines do); it must not mutate `graph`.
    fn align(&mut self, seq: &[u8], graph: &Graph) -> (Alignment, i32);
}

/// Aligns `seq` with `engine` against `graph`, merges it in weighted by `weight`, and returns the
/// assigned 0-based sequence index — collapsing the usual `engine.align` + `Graph::add_alignment_weight`
/// two-step so callers can correlate the result without counting calls (see the ordering guarantee
/// on [`Graph::add_alignment`]).
///
/// A free function rather than a `Graph` method so `graph` need not depend on the alignment trait.
///
/// # Errors
/// Propagates any [`GraphError`] from the underlying merge.
pub fn align_and_add<E: AlignmentEngine>(
    graph: &mut Graph,
    engine: &mut E,
    seq: &[u8],
    weight: u32,
) -> Result<u32, GraphError> {
    let index = graph.sequence_starts().len() as u32;
    let (alignment, _score) = engine.align(seq, graph);
    graph.add_alignment_weight(&alignment, seq, weight)?;
    Ok(index)
}

/// Like [`align_and_add`], but weights the merge by per-base `quality` (one entry per base of `seq`)
/// via [`Graph::add_alignment_quality`].
///
/// # Errors
/// Propagates any [`GraphError`] from the underlying merge (including a `quality`/`seq` length
/// mismatch).
pub fn align_and_add_quality<E: AlignmentEngine>(
    graph: &mut Graph,
    engine: &mut E,
    seq: &[u8],
    quality: &[u8],
) -> Result<u32, GraphError> {
    let index = graph.sequence_starts().len() as u32;
    let (alignment, _score) = engine.align(seq, graph);
    graph.add_alignment_quality(&alignment, seq, quality)?;
    Ok(index)
}

/// Converts a `-1`-sentinel [`Alignment`] into explicit `Option` pairs: each `(node_index,
/// sequence_index)` becomes `(Option<NodeId>, Option<u32>)`, with `-1` mapped to `None` (a graph
/// deletion or a sequence insertion, respectively). A one-way adaptor for callers that prefer the
/// optional form over the raw sentinel encoding.
pub fn alignment_to_optional(alignment: &Alignment) -> Vec<(Option<NodeId>, Option<u32>)> {
    alignment
        .iter()
        .map(|&(node_index, seq_index)| {
            let node = (node_index != -1).then_some(NodeId(node_index as u32));
            let seq = (seq_index != -1).then_some(seq_index as u32);
            (node, seq)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boundary table for [`Scoring::classify`] mirroring `alignment_engine.cpp:57-59`, exercised
    /// through [`Scoring::new`]/[`Scoring::gap_mode`] (i.e. against the *normalized* fields, per
    /// [`Scoring::gap_mode`]'s fixed-point argument).
    #[test]
    fn gap_mode_boundary_table() {
        struct Case {
            name: &'static str,
            g: i8,
            e: i8,
            q: i8,
            c: i8,
            expected: GapMode,
        }
        let cases = [
            Case {
                name: "g == e -> Linear",
                g: -2,
                e: -2,
                q: -5,
                c: -5,
                expected: GapMode::Linear,
            },
            Case {
                name: "g > e -> Linear",
                g: -2,
                e: -5,
                q: -1,
                c: -1,
                expected: GapMode::Linear,
            },
            Case {
                name: "g < e, g == q boundary -> Affine",
                g: -5,
                e: -2,
                q: -5,
                c: -1,
                expected: GapMode::Affine,
            },
            Case {
                name: "g < e, e == c boundary -> Affine",
                g: -5,
                e: -2,
                q: -9,
                c: -2,
                expected: GapMode::Affine,
            },
            Case {
                name: "g < e, g < q (strict) -> Affine",
                g: -5,
                e: -2,
                q: -3,
                c: -1,
                expected: GapMode::Affine,
            },
            Case {
                name: "g < e, e > c (strict, second disjunct) -> Affine",
                g: -5,
                e: -2,
                q: -9,
                c: -5,
                expected: GapMode::Affine,
            },
            Case {
                name: "q < g < e < c (strict) -> Convex",
                g: -5,
                e: -2,
                q: -8,
                c: -1,
                expected: GapMode::Convex,
            },
            Case {
                name: "spoa CLI default -8/-6/-10/-4 -> Convex",
                g: -8,
                e: -6,
                q: -10,
                c: -4,
                expected: GapMode::Convex,
            },
        ];

        for case in cases {
            let scoring = Scoring::new(5, -4, case.g, case.e, case.q, case.c)
                .unwrap_or_else(|e| panic!("{}: Scoring::new failed: {e}", case.name));
            assert_eq!(
                scoring.gap_mode(),
                case.expected,
                "{}: g={} e={} q={} c={}",
                case.name,
                case.g,
                case.e,
                case.q,
                case.c
            );
        }
    }

    #[test]
    fn normalization_linear_sets_extend_equal_to_open() {
        let scoring = Scoring::new(5, -4, -2, -2, -9, -9).unwrap();
        assert_eq!(scoring.gap_mode(), GapMode::Linear);
        assert_eq!(scoring.e, scoring.g, "linear normalization must set e == g");
        // Upstream leaves q/c untouched under Linear normalization.
        assert_eq!(scoring.q, -9);
        assert_eq!(scoring.c, -9);
    }

    #[test]
    fn normalization_affine_sets_second_pair_equal_to_first() {
        let scoring = Scoring::new(5, -4, -5, -2, -3, -1).unwrap();
        assert_eq!(scoring.gap_mode(), GapMode::Affine);
        assert_eq!(scoring.q, scoring.g, "affine normalization must set q == g");
        assert_eq!(scoring.c, scoring.e, "affine normalization must set c == e");
    }

    #[test]
    fn normalization_convex_leaves_all_penalties_unchanged() {
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        assert_eq!(scoring.gap_mode(), GapMode::Convex);
        assert_eq!(scoring.g, -8);
        assert_eq!(scoring.e, -6);
        assert_eq!(scoring.q, -10);
        assert_eq!(scoring.c, -4);
    }

    #[test]
    fn new_rejects_positive_gap_open() {
        let err = Scoring::new(5, -4, 1, -6, -10, -4).unwrap_err();
        assert_eq!(err, ScoringError::GapOpenPositive);
    }

    #[test]
    fn new_rejects_positive_second_gap_open() {
        let err = Scoring::new(5, -4, -8, -6, 1, -4).unwrap_err();
        assert_eq!(err, ScoringError::GapOpenPositive);
    }

    #[test]
    fn new_rejects_positive_gap_extend() {
        let err = Scoring::new(5, -4, -8, 1, -10, -4).unwrap_err();
        assert_eq!(err, ScoringError::GapExtendPositive);
    }

    #[test]
    fn new_rejects_positive_second_gap_extend() {
        let err = Scoring::new(5, -4, -8, -6, -10, 1).unwrap_err();
        assert_eq!(err, ScoringError::GapExtendPositive);
    }

    #[test]
    fn new_gap_open_check_takes_precedence_over_gap_extend_check() {
        // Mirrors alignment_engine.cpp:46-55: the g>0||q>0 check runs BEFORE e>0||c>0, so when
        // both are violated the gap-open error must win.
        let err = Scoring::new(5, -4, 1, 1, -10, -4).unwrap_err();
        assert_eq!(err, ScoringError::GapOpenPositive);
    }

    #[test]
    fn new_allows_zero_penalties() {
        // Non-positive means <= 0; zero must be accepted.
        Scoring::new(5, -4, 0, 0, 0, 0).expect("zero gap penalties are non-positive, hence valid");
    }

    #[test]
    fn spoa_default_is_convex_with_expected_penalties() {
        let s = Scoring::spoa_default();
        assert_eq!((s.m, s.n, s.g, s.e, s.q, s.c), (5, -4, -8, -6, -10, -4));
        assert_eq!(s.gap_mode(), GapMode::Convex);
    }

    #[test]
    fn align_and_add_returns_sequence_index_and_builds_graph() {
        let mut graph = Graph::new();
        let mut engine = SisdEngine::new(AlignmentType::Global, Scoring::spoa_default());
        assert_eq!(
            align_and_add(&mut graph, &mut engine, b"ACGT", 1).unwrap(),
            0
        );
        assert_eq!(
            align_and_add(&mut graph, &mut engine, b"ACGT", 1).unwrap(),
            1
        );
        assert_eq!(
            align_and_add(&mut graph, &mut engine, b"ACGT", 1).unwrap(),
            2
        );
        assert_eq!(graph.sequence_starts().len(), 3);
        assert_eq!(graph.generate_consensus(), "ACGT");
    }

    #[test]
    fn align_and_add_quality_returns_sequence_index() {
        let mut graph = Graph::new();
        let mut engine = SisdEngine::new(AlignmentType::Global, Scoring::spoa_default());
        let quality = [30u8; 4];
        assert_eq!(
            align_and_add_quality(&mut graph, &mut engine, b"ACGT", &quality).unwrap(),
            0
        );
        assert_eq!(
            align_and_add_quality(&mut graph, &mut engine, b"ACGT", &quality).unwrap(),
            1
        );
        assert_eq!(graph.sequence_starts().len(), 2);
    }

    #[test]
    fn alignment_to_optional_maps_minus_one_to_none() {
        let alignment: Alignment = vec![(0, 0), (-1, 1), (2, -1)];
        let optional = alignment_to_optional(&alignment);
        assert_eq!(
            optional,
            vec![
                (Some(NodeId(0)), Some(0)),
                (None, Some(1)),
                (Some(NodeId(2)), None),
            ]
        );
    }
}
