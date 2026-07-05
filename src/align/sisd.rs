//! The SISD (single-instruction-single-data, i.e. scalar) alignment engine: the reference DP
//! implementation the SIMD kernels are checked against.
//!
//! Mirrors `spoa::SisdAlignmentEngine` (`third_party/spoa/src/sisd_alignment_engine.cpp`): the DP
//! score buffers, buffer (re)allocation (`Realloc`, `:82-116`), per-alignment initialization of the
//! sequence profile and the DP matrices' boundary row/column (`Initialize`, `:118-257`), and the DP
//! fill and backtrack for all three gap modes (`Linear`/`Affine`/`Convex`, `:295-...` onward).

use super::backtrack::{backtrack_affine, backtrack_convex, backtrack_linear};
use super::{Alignment, AlignmentEngine, AlignmentType, GapMode, Scoring};
use crate::graph::{EdgeId, Graph};

/// A DP-cell "negative infinity" sentinel, chosen so that adding any single penalty to it still
/// leaves plenty of headroom before wrapping `i32`.
///
/// Ports `kNegativeInfinity` (`sisd_alignment_engine.cpp:13-14`) EXACTLY:
/// `i32::MIN + 1024`.
pub(crate) const NEG_INF: i32 = i32::MIN + 1024;

/// The scalar (non-SIMD) reference alignment engine.
///
/// Mirrors `spoa::SisdAlignmentEngine` and its private `Implementation` struct
/// (`sisd_alignment_engine.cpp:29-49`). Where upstream backs `H`/`F`/`E`/`O`/`Q` with one
/// `Vec<int32_t> M` and raw pointers into it, this port uses five independent `Vec<i32>` buffers
/// (no `unsafe`, so no raw-pointer aliasing is possible in the first place). Every buffer, once
/// allocated for a given `(matrix_width, matrix_height)`, is laid out row-major with cell `(i,
/// j)` — `i` a graph-node rank + 1 (row 0 is the "before any graph node" boundary), `j` a
/// sequence position + 1 (column 0 is the "before any sequence base" boundary) — at flat index
/// `i * matrix_width + j`, exactly matching upstream's `H[i * matrix_width + j]` indexing.
///
/// Buffers are only ever grown, never shrunk or re-laid-out (mirroring `Realloc`'s `if (...size()
/// < ...) resize(...)` guards): a later, smaller alignment call simply uses a smaller
/// `matrix_width`/`matrix_height` to compute its own offsets into a possibly-oversized buffer,
/// which stays self-consistent because every offset used in a given call is always computed from
/// that same call's `matrix_width`.
pub struct SisdEngine {
    alignment_type: AlignmentType,
    gap_mode: GapMode,
    scoring: Scoring,

    /// Maps a [`crate::graph::NodeId`] to its topological rank. Ports
    /// `Implementation::node_id_to_rank`.
    node_id_to_rank: Vec<u32>,
    /// Per-code, per-sequence-position match/mismatch score row: `sequence_profile[code *
    /// matrix_width + j]` is the score of aligning graph symbol `code` against `seq[j - 1]` (or
    /// `0` at the `j == 0` boundary). Ports `Implementation::sequence_profile`.
    sequence_profile: Vec<i32>,

    /// The main DP score matrix. Ports `Implementation::H`.
    h: Vec<i32>,
    /// Best score ending in a gap along the graph (column) axis. Ports `Implementation::F`.
    /// Empty (unused) under [`GapMode::Linear`].
    f: Vec<i32>,
    /// Best score ending in a gap along the sequence (row) axis. Ports `Implementation::E`.
    /// Empty (unused) under [`GapMode::Linear`].
    e: Vec<i32>,
    /// Best score ending in a *second-affine-layer* gap along the graph axis. Ports
    /// `Implementation::O`. Empty (unused) outside [`GapMode::Convex`].
    o: Vec<i32>,
    /// Best score ending in a *second-affine-layer* gap along the sequence axis. Ports
    /// `Implementation::Q`. Empty (unused) outside [`GapMode::Convex`].
    q: Vec<i32>,
}

impl SisdEngine {
    /// Creates a new SISD engine for `alignment_type`, with `scoring`'s (already validated and
    /// normalized) penalties.
    ///
    /// Mirrors `spoa::SisdAlignmentEngine::SisdAlignmentEngine`
    /// (`sisd_alignment_engine.cpp:51-62`); the `subtype_` upstream stores alongside `type_` is
    /// derived here from `scoring.gap_mode()` rather than threaded through as a separate
    /// constructor argument, since [`Scoring`] already normalizes consistently with it.
    pub fn new(alignment_type: AlignmentType, scoring: Scoring) -> SisdEngine {
        SisdEngine {
            alignment_type,
            gap_mode: scoring.gap_mode(),
            scoring,
            node_id_to_rank: Vec::new(),
            sequence_profile: Vec::new(),
            h: Vec::new(),
            f: Vec::new(),
            e: Vec::new(),
            o: Vec::new(),
            q: Vec::new(),
        }
    }

    /// Grows the DP buffers, if needed, to accommodate a `matrix_width` (`sequence_len + 1`)
    /// by `matrix_height` (`graph node count + 1`) matrix over an alphabet of `num_codes`
    /// symbols.
    ///
    /// Ports `spoa::SisdAlignmentEngine::Realloc` (`sisd_alignment_engine.cpp:82-116`) EXACTLY:
    /// every buffer only grows (never shrinks/reallocates smaller), and only the buffers needed
    /// by `self.gap_mode` are touched.
    fn realloc(&mut self, matrix_width: usize, matrix_height: usize, num_codes: usize) {
        if self.node_id_to_rank.len() < matrix_height - 1 {
            self.node_id_to_rank.resize(matrix_height - 1, 0);
        }
        if self.sequence_profile.len() < num_codes * matrix_width {
            self.sequence_profile.resize(num_codes * matrix_width, 0);
        }

        let cells = matrix_width * matrix_height;
        match self.gap_mode {
            GapMode::Linear => {
                if self.h.len() < cells {
                    self.h.resize(cells, 0);
                }
            }
            GapMode::Affine => {
                if self.h.len() < cells {
                    self.h.resize(cells, 0);
                    self.f.resize(cells, 0);
                    self.e.resize(cells, 0);
                }
            }
            GapMode::Convex => {
                if self.h.len() < cells {
                    self.h.resize(cells, 0);
                    self.f.resize(cells, 0);
                    self.e.resize(cells, 0);
                    self.o.resize(cells, 0);
                    self.q.resize(cells, 0);
                }
            }
        }
    }

    /// Builds the sequence profile and initializes the DP matrices' boundary row/column for
    /// aligning `seq` against `graph`.
    ///
    /// Ports `spoa::SisdAlignmentEngine::Initialize` (`sisd_alignment_engine.cpp:118-257`)
    /// EXACTLY, including its `switch` fallthrough behavior (a [`GapMode::Convex`] engine also
    /// runs the [`GapMode::Affine`] boundary init, which in turn also sets `h[0] = 0`) — see
    /// [`SisdEngine::initialize_affine_boundary`] and [`SisdEngine::initialize_convex_boundary`].
    fn initialize(&mut self, seq: &[u8], graph: &Graph) {
        let matrix_width = seq.len() + 1;
        let matrix_height = graph.nodes.len() + 1;
        let num_codes = graph.num_codes as usize;
        self.realloc(matrix_width, matrix_height, num_codes);

        // Sequence profile (sisd_alignment_engine.cpp:124-131).
        for code in 0..num_codes {
            let decoded = graph.decoder[code];
            self.sequence_profile[code * matrix_width] = 0;
            for (j, &base) in seq.iter().enumerate() {
                let score = if decoded == base as i32 {
                    self.scoring.m
                } else {
                    self.scoring.n
                };
                self.sequence_profile[code * matrix_width + (j + 1)] = i32::from(score);
            }
        }

        // node_id_to_rank (sisd_alignment_engine.cpp:133-136).
        for (rank, &node_id) in graph.rank_to_node.iter().enumerate() {
            self.node_id_to_rank[node_id.0 as usize] = rank as u32;
        }

        // Secondary-matrix boundary init (sisd_alignment_engine.cpp:138-181): the C++ `switch`
        // falls through Convex -> Affine -> Linear, so replicate that as explicit, cumulative
        // calls rather than duplicating the Affine/Linear bodies inside a Convex arm.
        if self.gap_mode == GapMode::Convex {
            self.initialize_convex_boundary(graph, matrix_width, matrix_height);
        }
        if self.gap_mode == GapMode::Convex || self.gap_mode == GapMode::Affine {
            self.initialize_affine_boundary(graph, matrix_width, matrix_height);
        }
        self.h[0] = 0;

        // Primary-matrix (H) boundary init (sisd_alignment_engine.cpp:183-256).
        match self.alignment_type {
            AlignmentType::Local => self.initialize_h_boundary_local(matrix_width, matrix_height),
            AlignmentType::Global => {
                self.initialize_h_boundary_global(graph, matrix_width, matrix_height)
            }
            AlignmentType::Overlap => {
                self.initialize_h_boundary_overlap(graph, matrix_width, matrix_height)
            }
        }
    }

    /// Initializes `O`/`Q`'s boundary row (`j` from 1) and boundary column (`i` from 1).
    ///
    /// Ports the `AlignmentSubtype::kConvex` case body (`sisd_alignment_engine.cpp:140-156`).
    fn initialize_convex_boundary(
        &mut self,
        graph: &Graph,
        matrix_width: usize,
        matrix_height: usize,
    ) {
        self.o[0] = 0;
        self.q[0] = 0;
        let (q_open, c_extend) = (i32::from(self.scoring.q), i32::from(self.scoring.c));
        for j in 1..matrix_width {
            self.o[j] = NEG_INF;
            self.q[j] = q_open + (j as i32 - 1) * c_extend;
        }
        for i in 1..matrix_height {
            let value = boundary_column_value(
                graph,
                &self.node_id_to_rank,
                &self.o,
                matrix_width,
                i - 1,
                q_open - c_extend,
            ) + c_extend;
            self.o[i * matrix_width] = value;
            self.q[i * matrix_width] = NEG_INF;
        }
    }

    /// Initializes `F`/`E`'s boundary row (`j` from 1) and boundary column (`i` from 1).
    ///
    /// Ports the `AlignmentSubtype::kAffine` case body (`sisd_alignment_engine.cpp:158-174`).
    fn initialize_affine_boundary(
        &mut self,
        graph: &Graph,
        matrix_width: usize,
        matrix_height: usize,
    ) {
        self.f[0] = 0;
        self.e[0] = 0;
        let (g_open, e_extend) = (i32::from(self.scoring.g), i32::from(self.scoring.e));
        for j in 1..matrix_width {
            self.f[j] = NEG_INF;
            self.e[j] = g_open + (j as i32 - 1) * e_extend;
        }
        for i in 1..matrix_height {
            let value = boundary_column_value(
                graph,
                &self.node_id_to_rank,
                &self.f,
                matrix_width,
                i - 1,
                g_open - e_extend,
            ) + e_extend;
            self.f[i * matrix_width] = value;
            self.e[i * matrix_width] = NEG_INF;
        }
    }

    /// Initializes `H`'s boundary row and column for [`AlignmentType::Local`] (`kSW`): both are
    /// all-zero, regardless of gap mode.
    ///
    /// Ports the `AlignmentType::kSW` case body (`sisd_alignment_engine.cpp:185-192`).
    fn initialize_h_boundary_local(&mut self, matrix_width: usize, matrix_height: usize) {
        for j in 1..matrix_width {
            self.h[j] = 0;
        }
        for i in 1..matrix_height {
            self.h[i * matrix_width] = 0;
        }
    }

    /// Initializes `H`'s boundary row and column for [`AlignmentType::Global`] (`kNW`): leading
    /// gaps are penalized on both axes.
    ///
    /// Ports the `AlignmentType::kNW` case body (`sisd_alignment_engine.cpp:193-229`).
    fn initialize_h_boundary_global(
        &mut self,
        graph: &Graph,
        matrix_width: usize,
        matrix_height: usize,
    ) {
        match self.gap_mode {
            GapMode::Convex => {
                for j in 1..matrix_width {
                    self.h[j] = self.q[j].max(self.e[j]);
                }
                for i in 1..matrix_height {
                    let idx = i * matrix_width;
                    self.h[idx] = self.o[idx].max(self.f[idx]);
                }
            }
            GapMode::Affine => {
                for j in 1..matrix_width {
                    self.h[j] = self.e[j];
                }
                for i in 1..matrix_height {
                    self.h[i * matrix_width] = self.f[i * matrix_width];
                }
            }
            GapMode::Linear => {
                let g_open = i32::from(self.scoring.g);
                for j in 1..matrix_width {
                    self.h[j] = j as i32 * g_open;
                }
                for i in 1..matrix_height {
                    let value = boundary_column_value(
                        graph,
                        &self.node_id_to_rank,
                        &self.h,
                        matrix_width,
                        i - 1,
                        0,
                    ) + g_open;
                    self.h[i * matrix_width] = value;
                }
            }
        }
    }

    /// Initializes `H`'s boundary row and column for [`AlignmentType::Overlap`] (`kOV`): leading
    /// gaps in the sequence (row) are penalized, but leading gaps in the graph (column) are free.
    ///
    /// Ports the `AlignmentType::kOV` case body (`sisd_alignment_engine.cpp:230-253`).
    fn initialize_h_boundary_overlap(
        &mut self,
        _graph: &Graph,
        matrix_width: usize,
        matrix_height: usize,
    ) {
        match self.gap_mode {
            GapMode::Convex => {
                for j in 1..matrix_width {
                    self.h[j] = self.q[j].max(self.e[j]);
                }
            }
            GapMode::Affine => {
                for j in 1..matrix_width {
                    self.h[j] = self.e[j];
                }
            }
            GapMode::Linear => {
                let g_open = i32::from(self.scoring.g);
                for j in 1..matrix_width {
                    self.h[j] = j as i32 * g_open;
                }
            }
        }
        for i in 1..matrix_height {
            self.h[i * matrix_width] = 0;
        }
    }

    /// Fills the DP matrix and backtracks the optimal alignment under a linear gap penalty.
    ///
    /// Ports `spoa::SisdAlignmentEngine::Linear` (`sisd_alignment_engine.cpp:295-463`) VERBATIM,
    /// including its per-`AlignmentType` max-score selection (`:353-361`) and the exact
    /// backtrack tie-break precedence (`:395-451`): match/mismatch is preferred over a graph-axis
    /// deletion, which is preferred over a sequence-axis insertion; within a step, in-edges are
    /// scanned in insertion order and the first exact-score match wins. Preserving this ordering
    /// byte-for-byte is what keeps consensus/MSA parity with spoa on any score-tie.
    ///
    /// Returns the alignment as `(node_id, seq_index)` pairs — `node_id` is the actual
    /// [`crate::graph::NodeId`] value (not its rank), matching the `add_alignment` contract on
    /// [`crate::graph::Graph`] — and the max score. Every `i8` penalty is promoted to `i32`
    /// before being added to an `i32` DP cell (Rust does not auto-promote), matching upstream's
    /// implicit `int` promotion.
    ///
    /// `seq_len` is the sequence length; [`SisdEngine::initialize`] must already have run for this
    /// `seq`/`graph` pair (the caller [`SisdEngine::align`] guarantees this).
    fn linear(&mut self, seq_len: usize, graph: &Graph) -> (Alignment, i32) {
        let matrix_width = seq_len + 1;
        let g = i32::from(self.scoring.g);
        let align_type = self.alignment_type;

        // Disjoint field borrows so the fill can mutate `h` while reading `node_id_to_rank` /
        // `sequence_profile` (and `graph`) through the `pred_row` helper below.
        let node_id_to_rank = &self.node_id_to_rank;
        let sequence_profile = &self.sequence_profile;
        let h = &mut self.h;

        // Rank (+1, i.e. the DP row) of an in-edge's tail node (its predecessor).
        let pred_row = |edge_id: EdgeId| -> usize {
            let tail = graph.edges[edge_id.0 as usize].tail;
            node_id_to_rank[tail.0 as usize] as usize + 1
        };

        let mut max_score = match align_type {
            AlignmentType::Local => 0,
            AlignmentType::Global | AlignmentType::Overlap => NEG_INF,
        };
        let mut max_i = 0usize;
        let mut max_j = 0usize;

        // Fill (sisd_alignment_engine.cpp:318-363).
        for &node_id in &graph.rank_to_node {
            let node = &graph.nodes[node_id.0 as usize];
            let profile_base = node.code as usize * matrix_width;
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;

            // First predecessor (:322-334).
            let mut pred_i = if node.inedges.is_empty() {
                0
            } else {
                pred_row(node.inedges[0])
            };
            for j in 1..matrix_width {
                let m = h[pred_i * matrix_width + (j - 1)] + sequence_profile[profile_base + j];
                let del = h[pred_i * matrix_width + j] + g;
                h[i * matrix_width + j] = m.max(del);
            }

            // Additional predecessors (:336-348).
            for p in 1..node.inedges.len() {
                pred_i = pred_row(node.inedges[p]);
                for j in 1..matrix_width {
                    let m = h[pred_i * matrix_width + (j - 1)] + sequence_profile[profile_base + j];
                    let cur = h[i * matrix_width + j];
                    let del = h[pred_i * matrix_width + j] + g;
                    h[i * matrix_width + j] = m.max(cur.max(del));
                }
            }

            // Gap-left, then per-type max-score tracking (:350-362). `update_max_score` uses a
            // STRICT `<`, so a tie keeps the EARLIER (i, j) — preserve this exactly.
            for j in 1..matrix_width {
                let mut value = (h[i * matrix_width + (j - 1)] + g).max(h[i * matrix_width + j]);
                match align_type {
                    AlignmentType::Local => {
                        value = value.max(0);
                        h[i * matrix_width + j] = value;
                        if max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                    AlignmentType::Global => {
                        h[i * matrix_width + j] = value;
                        if node.outedges.is_empty() && j == matrix_width - 1 && max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                    AlignmentType::Overlap => {
                        h[i * matrix_width + j] = value;
                        if node.outedges.is_empty() && max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                }
            }
        }

        let score = max_score;
        let alignment = backtrack_linear(
            graph,
            node_id_to_rank,
            sequence_profile,
            &h[..],
            matrix_width,
            align_type,
            &self.scoring,
            max_i,
            max_j,
            max_score,
        );
        (alignment, score)
    }

    /// Fills the DP matrices and backtracks the optimal alignment under an affine gap penalty.
    ///
    /// Ports `spoa::SisdAlignmentEngine::Affine` (`sisd_alignment_engine.cpp:465-679`) VERBATIM.
    /// The fill (`:487-545`) maintains three matrices: `H` (best score ending in a match/mismatch
    /// or either gap), `F` (best score ending in a gap along the graph/column axis), and `E` (best
    /// score ending in a gap along the sequence/row axis), with `F`/`E` distinguishing gap-open
    /// (`+ g`) from gap-extend (`+ e`). The backtrack (`:556-673`) shares [`SisdEngine::linear`]'s
    /// match-step tie-break precedence (match/mismatch, then graph-axis deletion, then
    /// sequence-axis insertion; in-edges scanned in insertion order, first exact-score match wins)
    /// but additionally unwinds affine gap *runs* via `extend_left` (walk the `E` insertion run
    /// leftward, `:645-653`) and `extend_up` (walk the `F` deletion run upward across
    /// predecessors, `:654-673`).
    ///
    /// Returns the alignment as `(node_id, seq_index)` pairs (see [`SisdEngine::linear`] for the
    /// sentinel/return contract) and the max score. Every `i8` penalty is promoted to `i32` before
    /// being added to an `i32` DP cell, matching upstream's implicit `int` promotion.
    ///
    /// `seq_len` is the sequence length; [`SisdEngine::initialize`] must already have run for this
    /// `seq`/`graph` pair (the caller [`SisdEngine::align`] guarantees this).
    fn affine(&mut self, seq_len: usize, graph: &Graph) -> (Alignment, i32) {
        let matrix_width = seq_len + 1;
        let g = i32::from(self.scoring.g);
        let e = i32::from(self.scoring.e);
        let align_type = self.alignment_type;

        // Disjoint field borrows so the fill can mutate `h`/`f`/`e_buf` while reading
        // `node_id_to_rank` / `sequence_profile` (and `graph`) through the `pred_row` helper.
        let node_id_to_rank = &self.node_id_to_rank;
        let sequence_profile = &self.sequence_profile;
        let h = &mut self.h;
        let f = &mut self.f;
        let e_buf = &mut self.e;

        // Rank (+1, i.e. the DP row) of an in-edge's tail node (its predecessor).
        let pred_row = |edge_id: EdgeId| -> usize {
            let tail = graph.edges[edge_id.0 as usize].tail;
            node_id_to_rank[tail.0 as usize] as usize + 1
        };

        let mut max_score = match align_type {
            AlignmentType::Local => 0,
            AlignmentType::Global | AlignmentType::Overlap => NEG_INF,
        };
        let mut max_i = 0usize;
        let mut max_j = 0usize;

        // Fill (sisd_alignment_engine.cpp:487-545).
        for &node_id in &graph.rank_to_node {
            let node = &graph.nodes[node_id.0 as usize];
            let profile_base = node.code as usize * matrix_width;
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;

            // First predecessor: update F and H (:502-508).
            let mut pred_i = if node.inedges.is_empty() {
                0
            } else {
                pred_row(node.inedges[0])
            };
            for j in 1..matrix_width {
                f[i * matrix_width + j] =
                    (h[pred_i * matrix_width + j] + g).max(f[pred_i * matrix_width + j] + e);
                h[i * matrix_width + j] =
                    h[pred_i * matrix_width + (j - 1)] + sequence_profile[profile_base + j];
            }

            // Additional predecessors (:510-526).
            for p in 1..node.inedges.len() {
                pred_i = pred_row(node.inedges[p]);
                for j in 1..matrix_width {
                    let cur_f = f[i * matrix_width + j];
                    f[i * matrix_width + j] = cur_f.max(
                        (h[pred_i * matrix_width + j] + g).max(f[pred_i * matrix_width + j] + e),
                    );
                    let cur_h = h[i * matrix_width + j];
                    h[i * matrix_width + j] = cur_h.max(
                        h[pred_i * matrix_width + (j - 1)] + sequence_profile[profile_base + j],
                    );
                }
            }

            // Update E and H, then per-type max-score tracking (:528-543). `update_max_score` uses
            // a STRICT `<`, so a tie keeps the EARLIER (i, j) — preserve this exactly.
            for j in 1..matrix_width {
                e_buf[i * matrix_width + j] =
                    (h[i * matrix_width + (j - 1)] + g).max(e_buf[i * matrix_width + (j - 1)] + e);
                let value = h[i * matrix_width + j]
                    .max(f[i * matrix_width + j].max(e_buf[i * matrix_width + j]));
                h[i * matrix_width + j] = value;

                match align_type {
                    AlignmentType::Local => {
                        let value = value.max(0);
                        h[i * matrix_width + j] = value;
                        if max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                    AlignmentType::Global => {
                        if node.outedges.is_empty() && j == matrix_width - 1 && max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                    AlignmentType::Overlap => {
                        if node.outedges.is_empty() && max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                }
            }
        }

        let score = max_score;
        let alignment = backtrack_affine(
            graph,
            node_id_to_rank,
            sequence_profile,
            &h[..],
            &e_buf[..],
            &f[..],
            matrix_width,
            align_type,
            &self.scoring,
            max_i,
            max_j,
            max_score,
        );
        (alignment, score)
    }

    /// Fills the DP matrices and backtracks the optimal alignment under a convex (double-affine)
    /// gap penalty.
    ///
    /// Ports `spoa::SisdAlignmentEngine::Convex` (`sisd_alignment_engine.cpp:681-926`) VERBATIM.
    /// Convex is affine plus a *second* affine gap function run in parallel: alongside `F`/`E`
    /// (gap-open `g`, gap-extend `e`) it maintains `O`/`Q` (second gap-open `q`, second gap-extend
    /// `c`), and `H` takes the 4-way max over both graph-axis matrices (`F`, `O`) and both
    /// sequence-axis matrices (`E`, `Q`). This lets a single gap be scored by whichever affine
    /// function is cheaper for its length (`min(g + (i-1)e, q + (i-1)c)`).
    ///
    /// The backtrack (`:799-922`) shares [`SisdEngine::affine`]'s match-step tie-break precedence
    /// (match/mismatch, then graph-axis deletion, then sequence-axis insertion; in-edges scanned
    /// in insertion order, first exact-score match wins), but its deletion/insertion steps use a
    /// four-term compound condition testing *both* gap functions (`F`+`e` OR `H`+`g` OR `O`+`c` OR
    /// `H`+`q`, via the upstream `(extend_* |= ...) || ...` short-circuit idiom, `:837-865`). The
    /// gap-run unwinds mirror that: `extend_left` walks leftward while *either* the `E` or `Q` run
    /// continues (`:879-887`, breaking only when NEITHER does), and `extend_up` walks upward in
    /// two phases per step — first try to continue an `F`/`O` extend across predecessors, and
    /// only if none continues fall back to an `F`/`O` gap-open scan (`:888-921`).
    ///
    /// Returns the alignment as `(node_id, seq_index)` pairs (see [`SisdEngine::linear`] for the
    /// sentinel/return contract) and the max score. Every `i8` penalty is promoted to `i32` before
    /// being added to an `i32` DP cell, matching upstream's implicit `int` promotion.
    ///
    /// `seq_len` is the sequence length; [`SisdEngine::initialize`] must already have run for this
    /// `seq`/`graph` pair (the caller [`SisdEngine::align`] guarantees this).
    fn convex(&mut self, seq_len: usize, graph: &Graph) -> (Alignment, i32) {
        let matrix_width = seq_len + 1;
        let g = i32::from(self.scoring.g);
        let e = i32::from(self.scoring.e);
        let q = i32::from(self.scoring.q);
        let c = i32::from(self.scoring.c);
        let align_type = self.alignment_type;

        // Disjoint field borrows so the fill can mutate `h`/`f`/`e_buf`/`o`/`q_buf` while reading
        // `node_id_to_rank` / `sequence_profile` (and `graph`) through the `pred_row` helper.
        let node_id_to_rank = &self.node_id_to_rank;
        let sequence_profile = &self.sequence_profile;
        let h = &mut self.h;
        let f = &mut self.f;
        let e_buf = &mut self.e;
        let o = &mut self.o;
        let q_buf = &mut self.q;

        // Rank (+1, i.e. the DP row) of an in-edge's tail node (its predecessor).
        let pred_row = |edge_id: EdgeId| -> usize {
            let tail = graph.edges[edge_id.0 as usize].tail;
            node_id_to_rank[tail.0 as usize] as usize + 1
        };

        let mut max_score = match align_type {
            AlignmentType::Local => 0,
            AlignmentType::Global | AlignmentType::Overlap => NEG_INF,
        };
        let mut max_i = 0usize;
        let mut max_j = 0usize;

        // Fill (sisd_alignment_engine.cpp:704-772).
        for &node_id in &graph.rank_to_node {
            let node = &graph.nodes[node_id.0 as usize];
            let profile_base = node.code as usize * matrix_width;
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;

            // First predecessor: update F, O and H (:721-726).
            let mut pred_i = if node.inedges.is_empty() {
                0
            } else {
                pred_row(node.inedges[0])
            };
            for j in 1..matrix_width {
                f[i * matrix_width + j] =
                    (h[pred_i * matrix_width + j] + g).max(f[pred_i * matrix_width + j] + e);
                o[i * matrix_width + j] =
                    (h[pred_i * matrix_width + j] + q).max(o[pred_i * matrix_width + j] + c);
                h[i * matrix_width + j] =
                    h[pred_i * matrix_width + (j - 1)] + sequence_profile[profile_base + j];
            }

            // Additional predecessors (:728-748).
            for p in 1..node.inedges.len() {
                pred_i = pred_row(node.inedges[p]);
                for j in 1..matrix_width {
                    let cur_f = f[i * matrix_width + j];
                    f[i * matrix_width + j] = cur_f.max(
                        (h[pred_i * matrix_width + j] + g).max(f[pred_i * matrix_width + j] + e),
                    );
                    let cur_o = o[i * matrix_width + j];
                    o[i * matrix_width + j] = cur_o.max(
                        (h[pred_i * matrix_width + j] + q).max(o[pred_i * matrix_width + j] + c),
                    );
                    let cur_h = h[i * matrix_width + j];
                    h[i * matrix_width + j] = cur_h.max(
                        h[pred_i * matrix_width + (j - 1)] + sequence_profile[profile_base + j],
                    );
                }
            }

            // Update E, Q and H (4-way max), then per-type max-score tracking (:750-771).
            // `update_max_score` uses a STRICT `<`, so a tie keeps the EARLIER (i, j).
            for j in 1..matrix_width {
                e_buf[i * matrix_width + j] =
                    (h[i * matrix_width + (j - 1)] + g).max(e_buf[i * matrix_width + (j - 1)] + e);
                q_buf[i * matrix_width + j] =
                    (h[i * matrix_width + (j - 1)] + q).max(q_buf[i * matrix_width + (j - 1)] + c);
                let gap_max = f[i * matrix_width + j]
                    .max(e_buf[i * matrix_width + j])
                    .max(o[i * matrix_width + j])
                    .max(q_buf[i * matrix_width + j]);
                let value = h[i * matrix_width + j].max(gap_max);
                h[i * matrix_width + j] = value;

                match align_type {
                    AlignmentType::Local => {
                        let value = value.max(0);
                        h[i * matrix_width + j] = value;
                        if max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                    AlignmentType::Global => {
                        if node.outedges.is_empty() && j == matrix_width - 1 && max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                    AlignmentType::Overlap => {
                        if node.outedges.is_empty() && max_score < value {
                            max_score = value;
                            max_i = i;
                            max_j = j;
                        }
                    }
                }
            }
        }

        let score = max_score;
        let alignment = backtrack_convex(
            graph,
            node_id_to_rank,
            sequence_profile,
            &h[..],
            &e_buf[..],
            &f[..],
            &o[..],
            &q_buf[..],
            matrix_width,
            align_type,
            &self.scoring,
            max_i,
            max_j,
            max_score,
        );
        (alignment, score)
    }
}

/// Computes a boundary-column cell's value at graph rank `rank` (0-based; row `rank + 1` in the
/// DP matrix): the maximum of `empty_penalty` (used when `rank`'s node has no in-edges) and, for
/// every in-edge, `buffer`'s already-computed value at that in-edge's tail's row.
///
/// This is the shared shape of three upstream loops that are otherwise identical apart from
/// which buffer/base-penalty/increment they use: the `O` column
/// (`sisd_alignment_engine.cpp:147-154`), the `F` column (`:165-172`), and `H`'s column under
/// `kNW`+`kLinear` (`:217-224`). The caller adds its own `increment` (`c_`/`e_`/`g_`
/// respectively) to the returned value.
fn boundary_column_value(
    graph: &Graph,
    node_id_to_rank: &[u32],
    buffer: &[i32],
    matrix_width: usize,
    rank: usize,
    empty_penalty: i32,
) -> i32 {
    let node_id = graph.rank_to_node[rank];
    let inedges = &graph.nodes[node_id.0 as usize].inedges;
    let mut penalty = if inedges.is_empty() {
        empty_penalty
    } else {
        NEG_INF
    };
    for &edge_id in inedges {
        let tail = graph.edges[edge_id.0 as usize].tail;
        let pred_row = node_id_to_rank[tail.0 as usize] as usize + 1;
        penalty = penalty.max(buffer[pred_row * matrix_width]);
    }
    penalty
}

/// Row-major DP boundary buffers, `sequence_profile`, and `node_id_to_rank` produced by running
/// the exact scalar [`SisdEngine::initialize`] setup for a `(graph, seq, scoring,
/// alignment_type)` combination.
///
/// Returned by [`seed_scalar_buffers`] — the SIMD kernels plan's "C2 fix": upstream's striped
/// SIMD matrices carry neither a column 0 (kept separately in a `first_column` array) nor a
/// row-major `sequence_profile` (the striped profile has no column-0 boundary, since its
/// `normal_matrix_width` is `seq.len()`, not `seq.len() + 1`), yet the shared scalar backtrack
/// (`backtrack_linear`/`backtrack_affine`/`backtrack_convex`) reads both, plus row 0. Seeding
/// those three things from the verified scalar [`SisdEngine::initialize`] — rather than
/// re-deriving the boundary formulas a second time for the SIMD path — is what keeps the two
/// paths from silently diverging.
///
/// Column 0 of `h`/`e`/`f`/`o`/`q` here doubles as upstream's vector `first_column` seed (the one
/// the SIMD fill shifts into lane 0): linear-NW's column formula
/// (`simd_alignment_engine_implementation.hpp:583-592`) is `boundary_column_value(..,0) + g`,
/// affine's F column (`:535-543`) is exactly [`SisdEngine::initialize_affine_boundary`], and the
/// NW boundary row (`:593-601`) matches [`SisdEngine::initialize_h_boundary_global`]'s Linear arm
/// — verified bit-identical during the plan's Task 5 review. So no separate "build first_column"
/// step is needed on the SIMD side: a later fill reads column 0 directly out of these buffers.
///
/// Only the buffers `alignment_type`'s [`GapMode`] actually uses are non-empty, mirroring
/// [`SisdEngine::realloc`]'s per-gap-mode allocation (e.g. under [`GapMode::Linear`], `e`/`f`/`o`/
/// `q` are empty).
// Not yet constructed outside this module's own tests and `simd::profile`'s tests: a later SIMD
// kernels plan task wires `simd::profile::seed_scalar_buffers` into the real vectorized fill
// pipeline, at which point this `allow` is removed (see `SimdEngine`'s identical rationale in
// `simd/mod.rs`).
#[allow(dead_code)]
#[derive(Default)]
pub(crate) struct ScalarInit {
    /// The main DP score matrix's boundary row 0 / column 0 (and, in general, its full buffer as
    /// allocated by [`SisdEngine::realloc`], though only row 0/column 0 are guaranteed seeded — a
    /// later SIMD fill overwrites the interior via `destripe_interior`).
    pub(crate) h: Vec<i32>,
    /// The sequence-axis gap matrix's boundary. Empty under [`GapMode::Linear`].
    pub(crate) e: Vec<i32>,
    /// The graph-axis gap matrix's boundary. Empty under [`GapMode::Linear`].
    pub(crate) f: Vec<i32>,
    /// The second-affine-layer graph-axis gap matrix's boundary. Empty outside
    /// [`GapMode::Convex`].
    pub(crate) o: Vec<i32>,
    /// The second-affine-layer sequence-axis gap matrix's boundary. Empty outside
    /// [`GapMode::Convex`].
    pub(crate) q: Vec<i32>,
    /// The row-major match/mismatch score table: `sequence_profile[code * matrix_width + j]`.
    pub(crate) sequence_profile: Vec<i32>,
    /// Maps a [`crate::graph::NodeId`] to its topological rank.
    pub(crate) node_id_to_rank: Vec<u32>,
    /// `seq.len() + 1` — the row-major buffers' width, matching every other row-major layout in
    /// this crate.
    pub(crate) matrix_width: usize,
}

/// Runs [`SisdEngine::initialize`] for `(alignment_type, scoring, seq, graph)` and returns its
/// row 0 / column 0 boundary state, `sequence_profile`, and `node_id_to_rank` as a
/// [`ScalarInit`].
///
/// This is a thin, deliberately non-reimplementing wrapper: it builds a fresh [`SisdEngine`] and
/// calls straight through to its private, already-verified `initialize` method, then moves the
/// resulting buffers out. See [`ScalarInit`]'s doc for why this call-through — rather than a
/// second, independent port of the boundary formulas — is the SIMD kernels plan's "C2 fix".
// See `ScalarInit`'s identical `#[allow(dead_code)]` rationale immediately above.
#[allow(dead_code)]
pub(crate) fn seed_scalar_buffers(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
) -> ScalarInit {
    let mut engine = SisdEngine::new(alignment_type, scoring);
    engine.initialize(seq, graph);
    ScalarInit {
        h: engine.h,
        e: engine.e,
        f: engine.f,
        o: engine.o,
        q: engine.q,
        sequence_profile: engine.sequence_profile,
        node_id_to_rank: engine.node_id_to_rank,
        matrix_width: seq.len() + 1,
    }
}

/// Re-seeds `scratch` in place for a new `(seq, graph)` under `(alignment_type, scoring)`, REUSING
/// `scratch`'s already-allocated buffers instead of allocating (and zeroing) fresh ones — the
/// grow-only-buffer-reuse the SIMD engine relies on for its per-`align` performance (P2). Same
/// result as [`seed_scalar_buffers`], but without the per-call `Vec` allocation of the (largest,
/// and always-zeroed) row-major DP buffers.
///
/// The mechanism moves `scratch`'s owned buffers into a throwaway [`SisdEngine`] (whose own `Vec`
/// fields start empty, so the moves are O(1) pointer swaps — no heap allocation), runs the same
/// verified [`SisdEngine::initialize`] (whose `Realloc` grows the moved-in buffers only if the new
/// matrix is larger, never shrinking or reallocating them smaller), then moves the — now
/// re-seeded, possibly-grown — buffers back into `scratch`. This deliberately routes through the
/// SAME `initialize` as [`seed_scalar_buffers`] rather than re-deriving the boundary formulas (the
/// C2-fix call-through — see [`ScalarInit`]'s doc).
///
/// # Buffer-reuse correctness (the output invariant)
///
/// [`SisdEngine::initialize`] seeds only the boundary (row 0 / column 0) of every DP buffer; it
/// does NOT clear the interior. On reuse, the interior therefore holds stale values from a prior
/// (possibly larger) alignment. This is sound for exactly the reason [`SisdEngine::realloc`]'s own
/// grow-only reuse is: every interior cell a later read touches is first WRITTEN by that same
/// alignment (here, by the SIMD fill's `destripe_interior` over rows `1..`, columns `1..=seq_len`),
/// and every offset is computed from the CURRENT call's `matrix_width`, so no stale tail cell of an
/// over-sized buffer is ever addressed. The seeded boundary plus the destriped interior together
/// cover every cell the backtrack subsequently reads — bit-for-bit identical to a freshly-allocated
/// [`seed_scalar_buffers`] call.
// See `seed_scalar_buffers`'s identical `#[allow(dead_code)]` rationale (unused on targets without
// a vectorized backend wired in).
#[allow(dead_code)]
pub(crate) fn reseed_scalar_buffers(
    scratch: &mut ScalarInit,
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
) {
    let mut engine = SisdEngine::new(alignment_type, scoring);
    engine.h = std::mem::take(&mut scratch.h);
    engine.e = std::mem::take(&mut scratch.e);
    engine.f = std::mem::take(&mut scratch.f);
    engine.o = std::mem::take(&mut scratch.o);
    engine.q = std::mem::take(&mut scratch.q);
    engine.sequence_profile = std::mem::take(&mut scratch.sequence_profile);
    engine.node_id_to_rank = std::mem::take(&mut scratch.node_id_to_rank);

    engine.initialize(seq, graph);

    scratch.h = engine.h;
    scratch.e = engine.e;
    scratch.f = engine.f;
    scratch.o = engine.o;
    scratch.q = engine.q;
    scratch.sequence_profile = engine.sequence_profile;
    scratch.node_id_to_rank = engine.node_id_to_rank;
    scratch.matrix_width = seq.len() + 1;
}

impl AlignmentEngine for SisdEngine {
    /// Aligns `seq` against `graph`.
    ///
    /// Validates and initializes exactly as upstream's `Align` does
    /// (`sisd_alignment_engine.cpp:259-283`) — the `sequence_len` overflow guard, the
    /// [`Scoring::worst_case_alignment_score`] overflow guard, `Realloc`, then `Initialize` — and
    /// then dispatches on the gap mode (`:285-291`) to [`SisdEngine::linear`],
    /// [`SisdEngine::affine`], or [`SisdEngine::convex`].
    ///
    /// # Panics
    ///
    /// Panics (mirroring upstream's `std::invalid_argument` throws, since this trait's signature
    /// is infallible) if `seq` is longer than `i32::MAX`, or if the worst-case alignment score
    /// for `seq`/`graph`'s lengths would underflow [`NEG_INF`].
    fn align(&mut self, seq: &[u8], graph: &Graph) -> (super::Alignment, i32) {
        assert!(
            seq.len() <= i32::MAX as usize,
            "[spoars::SisdEngine::align] error: too large sequence!"
        );

        if graph.nodes.is_empty() || seq.is_empty() {
            return (Vec::new(), 0);
        }

        let worst_case = self
            .scoring
            .worst_case_alignment_score(seq.len() as i64, graph.nodes.len() as i64);
        assert!(
            worst_case >= i64::from(NEG_INF),
            "[spoars::SisdEngine::align] error: possible overflow!"
        );

        self.initialize(seq, graph);

        match self.gap_mode {
            GapMode::Linear => self.linear(seq.len(), graph),
            GapMode::Affine => self.affine(seq.len(), graph),
            GapMode::Convex => self.convex(seq.len(), graph),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    fn linear_scoring() -> Scoring {
        Scoring::new(5, -4, -8, -8, -8, -8).unwrap()
    }

    fn affine_scoring() -> Scoring {
        Scoring::new(5, -4, -8, -6, -8, -6).unwrap()
    }

    fn convex_scoring() -> Scoring {
        Scoring::new(5, -4, -8, -6, -10, -4).unwrap()
    }

    #[test]
    fn initialize_linear_global_on_tiny_graph_does_not_panic() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();

        let mut engine = SisdEngine::new(AlignmentType::Global, linear_scoring());
        engine.initialize(b"AG", &g);

        // H's boundary row follows j * g (NW + Linear).
        let matrix_width = 3; // seq_len + 1
        assert_eq!(engine.h[0], 0);
        assert_eq!(engine.h[1], -8);
        assert_eq!(engine.h[2], -16);
        // Boundary column: node 0 (rank 0) has no inedges -> empty_penalty 0, + g.
        assert_eq!(engine.h[matrix_width], -8);
    }

    #[test]
    fn initialize_affine_local_on_tiny_graph_does_not_panic() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();

        let mut engine = SisdEngine::new(AlignmentType::Local, affine_scoring());
        engine.initialize(b"AG", &g);

        // kSW's H boundary is all-zero regardless of gap mode.
        assert!(engine.h[..3].iter().all(|&v| v == 0));
        assert_eq!(engine.f[0], 0);
        assert_eq!(engine.e[0], 0);
    }

    #[test]
    fn initialize_convex_overlap_on_tiny_graph_does_not_panic() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();

        let mut engine = SisdEngine::new(AlignmentType::Overlap, convex_scoring());
        engine.initialize(b"AG", &g);

        assert_eq!(engine.o[0], 0);
        assert_eq!(engine.q[0], 0);
        assert_eq!(engine.f[0], 0);
        assert_eq!(engine.e[0], 0);
        // kOV's H boundary column is always 0.
        let matrix_width = 3;
        assert_eq!(engine.h[matrix_width], 0);
    }

    #[test]
    fn initialize_on_empty_graph_and_empty_sequence_does_not_panic() {
        let g = Graph::new();
        let mut engine = SisdEngine::new(AlignmentType::Global, linear_scoring());
        engine.initialize(b"", &g);
        assert_eq!(engine.h[0], 0);
    }

    /// [`seed_scalar_buffers`] must reproduce a manually constructed-and-`initialize`d
    /// [`SisdEngine`]'s buffers EXACTLY — this is the regression guard for the SIMD kernels
    /// plan's "C2 fix" call-through: if a future refactor ever made `seed_scalar_buffers` stop
    /// calling straight through to `initialize` (e.g. inlining a "simplified" copy), this test
    /// fails immediately. Covers [`AlignmentType::Global`] (`kNW`, penalized boundary).
    #[test]
    fn seed_scalar_buffers_matches_manually_initialized_engine_for_nw() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        let scoring = linear_scoring();
        let seq: &[u8] = b"AG";

        let mut engine = SisdEngine::new(AlignmentType::Global, scoring);
        engine.initialize(seq, &g);

        let seeded = seed_scalar_buffers(AlignmentType::Global, scoring, seq, &g);

        assert_eq!(seeded.h, engine.h);
        assert_eq!(seeded.e, engine.e);
        assert_eq!(seeded.f, engine.f);
        assert_eq!(seeded.o, engine.o);
        assert_eq!(seeded.q, engine.q);
        assert_eq!(seeded.sequence_profile, engine.sequence_profile);
        assert_eq!(seeded.node_id_to_rank, engine.node_id_to_rank);
        assert_eq!(seeded.matrix_width, seq.len() + 1);
    }

    /// See [`seed_scalar_buffers_matches_manually_initialized_engine_for_nw`]; this covers
    /// [`AlignmentType::Local`] (`kSW`, all-zero boundary) under [`GapMode::Affine`], which zeroes
    /// (rather than penalizes) `H`'s boundary and differs in shape from the NW case above.
    #[test]
    fn seed_scalar_buffers_matches_manually_initialized_engine_for_sw() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        let scoring = affine_scoring();
        let seq: &[u8] = b"AG";

        let mut engine = SisdEngine::new(AlignmentType::Local, scoring);
        engine.initialize(seq, &g);

        let seeded = seed_scalar_buffers(AlignmentType::Local, scoring, seq, &g);

        assert_eq!(seeded.h, engine.h);
        assert_eq!(seeded.e, engine.e);
        assert_eq!(seeded.f, engine.f);
        assert_eq!(seeded.o, engine.o);
        assert_eq!(seeded.q, engine.q);
        assert_eq!(seeded.sequence_profile, engine.sequence_profile);
        assert_eq!(seeded.node_id_to_rank, engine.node_id_to_rank);
    }

    /// See [`seed_scalar_buffers_matches_manually_initialized_engine_for_nw`]; this covers
    /// [`AlignmentType::Overlap`] (`kOV`, free graph-axis leading gaps) under [`GapMode::Convex`],
    /// which populates `o`/`q` in addition to `f`/`e`.
    #[test]
    fn seed_scalar_buffers_matches_manually_initialized_engine_for_ov() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        let scoring = convex_scoring();
        let seq: &[u8] = b"AG";

        let mut engine = SisdEngine::new(AlignmentType::Overlap, scoring);
        engine.initialize(seq, &g);

        let seeded = seed_scalar_buffers(AlignmentType::Overlap, scoring, seq, &g);

        assert_eq!(seeded.h, engine.h);
        assert_eq!(seeded.e, engine.e);
        assert_eq!(seeded.f, engine.f);
        assert_eq!(seeded.o, engine.o);
        assert_eq!(seeded.q, engine.q);
        assert_eq!(seeded.sequence_profile, engine.sequence_profile);
        assert_eq!(seeded.node_id_to_rank, engine.node_id_to_rank);
    }

    /// [`reseed_scalar_buffers`] must produce buffers bit-identical to a fresh
    /// [`seed_scalar_buffers`], EVEN when the scratch it reuses was previously seeded for a LARGER
    /// alignment (so its buffers are over-sized and carry stale interior/tail values). This is the
    /// grow-only-reuse output invariant the SIMD engine's P2 buffer reuse depends on: only the
    /// boundary is re-seeded here (the interior is the caller's responsibility to overwrite), so
    /// this compares boundary/profile/rank exactly and confirms the reused, over-sized buffers'
    /// current-`matrix_width` offsets agree with a freshly-allocated seed's.
    #[test]
    fn reseed_scalar_buffers_matches_fresh_seed_after_reuse_from_a_larger_alignment() {
        let scoring = convex_scoring();
        let alignment_type = AlignmentType::Global;

        // First, seed the scratch for a LARGE alignment so its buffers are over-sized with real
        // (non-zero) content everywhere.
        let mut big_graph = Graph::new();
        big_graph
            .add_alignment_weight(&[], b"ACGTACGTACGTACGT", 1)
            .unwrap();
        big_graph
            .add_alignment_weight(&[], b"ACGTTCGTACGATCGT", 1)
            .unwrap();
        let mut scratch = ScalarInit::default();
        reseed_scalar_buffers(
            &mut scratch,
            alignment_type,
            scoring,
            b"ACGTACGTACGTACGT",
            &big_graph,
        );

        // Now re-seed the SAME scratch for a SMALLER alignment; it must match a fresh seed of that
        // smaller alignment on every cell addressed at the smaller `matrix_width`.
        let mut small_graph = Graph::new();
        small_graph.add_alignment_weight(&[], b"ACGT", 1).unwrap();
        let seq: &[u8] = b"AGGT";
        reseed_scalar_buffers(&mut scratch, alignment_type, scoring, seq, &small_graph);
        let fresh = seed_scalar_buffers(alignment_type, scoring, seq, &small_graph);

        let matrix_width = scratch.matrix_width;
        assert_eq!(matrix_width, fresh.matrix_width);
        let matrix_height = small_graph.nodes.len() + 1;
        let num_codes = small_graph.num_codes as usize;

        // node_id_to_rank / sequence_profile compare over their (smaller) logical extents.
        assert_eq!(
            scratch.node_id_to_rank[..small_graph.nodes.len()],
            fresh.node_id_to_rank[..small_graph.nodes.len()]
        );
        for code in 0..num_codes {
            for j in 0..matrix_width {
                assert_eq!(
                    scratch.sequence_profile[code * matrix_width + j],
                    fresh.sequence_profile[code * matrix_width + j],
                    "sequence_profile code={code} j={j}"
                );
            }
        }

        // Every DP boundary cell (row 0 and column 0 of h/e/f/o/q) must match, addressed at the
        // smaller matrix_width even though the reused buffers are physically larger.
        for buf_pair in [
            (&scratch.h, &fresh.h),
            (&scratch.e, &fresh.e),
            (&scratch.f, &fresh.f),
            (&scratch.o, &fresh.o),
            (&scratch.q, &fresh.q),
        ] {
            let (reused, fresh_buf) = buf_pair;
            if fresh_buf.is_empty() {
                continue;
            }
            for j in 0..matrix_width {
                assert_eq!(reused[j], fresh_buf[j], "boundary row0 j={j}");
            }
            for i in 0..matrix_height {
                assert_eq!(
                    reused[i * matrix_width],
                    fresh_buf[i * matrix_width],
                    "boundary col0 i={i}"
                );
            }
        }
    }
}
