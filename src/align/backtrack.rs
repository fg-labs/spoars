//! The scalar backtrack shared by every gap mode, extracted from [`super::sisd::SisdEngine`] so
//! upcoming SIMD kernels can destripe their vectorized fill's matrices into these same row-major
//! `i32` buffers and reuse this exact backtrack — guaranteeing bit-identical tie-breaks with the
//! scalar engine (spoa's SIMD engine vectorizes only the DP fill; it backtracks scalar-ly on the
//! fully-stored matrices, per `third_party/spoa/src/simd_alignment_engine_implementation.hpp`).
//!
//! Each `backtrack_*` function below is a VERBATIM port of the corresponding backtrack half of
//! `spoa::SisdAlignmentEngine::{Linear,Affine,Convex}` (`sisd_alignment_engine.cpp`), reading
//! from caller-supplied `&[i32]` buffer slices — instead of `self`/`Implementation` fields — plus
//! the fill's already-computed `(max_i, max_j, max_score)` start cell. This is a pure extraction:
//! the tie-break precedence (match/mismatch, then graph-axis deletion, then sequence-axis
//! insertion; in-edges scanned in insertion order with first-exact-score-match winning), the
//! affine `extend_left`/`extend_up` gap-run unwinds, and the convex two-phase `extend_up` `|=`
//! compounds are unchanged from [`super::sisd`].

use super::{Alignment, AlignmentType, Scoring};
use crate::graph::{EdgeId, Graph};

/// Read-only access to a DP matrix cell `(i, j)` (graph row, sequence column) as `i32`, abstracting
/// over the layout the values live in. The scalar engine (and any destriped SIMD path) reads them
/// row-major via [`RowMajor`]; the SIMD path can instead index its striped fill output directly
/// (see the striped view in [`super::simd`]), letting it skip the full-matrix destripe and touch
/// only the cells the backtrack actually walks.
pub(crate) trait CellRead {
    /// The DP value at graph row `i`, sequence column `j`.
    fn get(&self, i: usize, j: usize) -> i32;
}

/// Row-major view of a contiguous `i32` DP buffer: cell `(i, j)` is `buf[i * width + j]`.
pub(crate) struct RowMajor<'a> {
    pub buf: &'a [i32],
    pub width: usize,
}

impl CellRead for RowMajor<'_> {
    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> i32 {
        self.buf[i * self.width + j]
    }
}

/// Backtrack step 1 (match/mismatch) — the highest-precedence predecessor test, shared verbatim
/// by [`backtrack_linear`], [`backtrack_affine`], and [`backtrack_convex`]
/// (`sisd_alignment_engine.cpp:395-420,577-601,805-829` — byte-identical in all three).
///
/// Returns `Some((prev_i, prev_j))` for the first in-edge whose match/mismatch transition
/// explains `h_ij`, scanning the first in-edge then `inedges[1..]` in insertion order and taking
/// the first exact-score equality; returns `None` if none does (including at the
/// `i == 0 || j == 0` boundary). Reads only `h`/`sequence_profile`/`graph`/`node_id_to_rank`.
/// **The in-edge iteration order and first-equality-wins semantics are the tie-break that keeps
/// consensus/MSA parity with spoa — they must not be reordered.**
#[inline]
fn backtrack_match_step<V: CellRead>(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &V,
    matrix_width: usize,
    i: usize,
    j: usize,
) -> Option<(usize, usize)> {
    if i == 0 || j == 0 {
        return None;
    }
    let h_ij = h.get(i, j);
    let node = &graph.nodes[graph.rank_to_node[i - 1].0 as usize];
    let match_cost = sequence_profile[node.code as usize * matrix_width + j];
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };
    let pred_first = if node.inedges.is_empty() {
        0
    } else {
        pred_row(node.inedges[0])
    };
    if h_ij == h.get(pred_first, j - 1) + match_cost {
        return Some((pred_first, j - 1));
    }
    for p in 1..node.inedges.len() {
        let pred_i = pred_row(node.inedges[p]);
        if h_ij == h.get(pred_i, j - 1) + match_cost {
            return Some((pred_i, j - 1));
        }
    }
    None
}

/// Backtracks the optimal alignment under a linear gap penalty from the fill's best cell.
///
/// Ports the backtrack half of `spoa::SisdAlignmentEngine::Linear`
/// (`sisd_alignment_engine.cpp:372-462`) VERBATIM. `(max_i, max_j, max_score)` are the fill's
/// already-selected best cell/score (per `alignment_type`'s per-type max-score rule); if
/// `max_i == 0 && max_j == 0` (no cell was ever selected — an empty graph or sequence), returns an
/// empty alignment immediately, matching `sisd_alignment_engine.cpp:365-367`. Otherwise
/// `h[max_i * matrix_width + max_j]` must equal `max_score` (checked via `debug_assert_eq!`, which
/// also doubles as a destripe sanity check for the upcoming SIMD fills: if a vectorized fill
/// destripes `H` incorrectly, this assertion is the first line of defense).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn backtrack_linear(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &[i32],
    matrix_width: usize,
    alignment_type: AlignmentType,
    scoring: &Scoring,
    max_i: usize,
    max_j: usize,
    max_score: i32,
) -> Alignment {
    backtrack_linear_impl(
        graph,
        node_id_to_rank,
        sequence_profile,
        &RowMajor {
            buf: h,
            width: matrix_width,
        },
        matrix_width,
        alignment_type,
        scoring,
        max_i,
        max_j,
        max_score,
    )
}

/// The linear backtrack proper, generic over a [`CellRead`] view of `H`. [`backtrack_linear`]
/// feeds it a [`RowMajor`] view; the SIMD path feeds a striped view to skip the destripe (see
/// [`backtrack_convex_impl`] for the rationale).
#[allow(clippy::too_many_arguments)]
pub(crate) fn backtrack_linear_impl<V: CellRead>(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &V,
    matrix_width: usize,
    alignment_type: AlignmentType,
    scoring: &Scoring,
    max_i: usize,
    max_j: usize,
    max_score: i32,
) -> Alignment {
    if max_i == 0 && max_j == 0 {
        return Vec::new();
    }
    debug_assert_eq!(
        h.get(max_i, max_j),
        max_score,
        "fill's max_score must match H at (max_i, max_j)"
    );
    let g = i32::from(scoring.g);

    // Rank (+1, i.e. the DP row) of an in-edge's tail node (its predecessor).
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };

    let mut alignment: Alignment = Vec::new();
    let mut i = max_i;
    let mut j = max_j;

    loop {
        let keep_going = match alignment_type {
            AlignmentType::Local => h.get(i, j) != 0,
            AlignmentType::Global => !(i == 0 && j == 0),
            AlignmentType::Overlap => !(i == 0 || j == 0),
        };
        if !keep_going {
            break;
        }

        let h_ij = h.get(i, j);
        let mut prev_i = 0usize;
        let mut prev_j = 0usize;
        let mut predecessor_found = false;

        // 1. Match/mismatch (:395-420) — highest precedence.
        if let Some((pi, pj)) = backtrack_match_step(
            graph,
            node_id_to_rank,
            sequence_profile,
            h,
            matrix_width,
            i,
            j,
        ) {
            prev_i = pi;
            prev_j = pj;
            predecessor_found = true;
        }

        // 2. Deletion / gap along the graph axis (:422-445).
        if !predecessor_found && i != 0 {
            let node = &graph.nodes[graph.rank_to_node[i - 1].0 as usize];
            let pred_first = if node.inedges.is_empty() {
                0
            } else {
                pred_row(node.inedges[0])
            };
            if h_ij == h.get(pred_first, j) + g {
                prev_i = pred_first;
                prev_j = j;
                predecessor_found = true;
            } else {
                for p in 1..node.inedges.len() {
                    let pred_i = pred_row(node.inedges[p]);
                    if h_ij == h.get(pred_i, j) + g {
                        prev_i = pred_i;
                        prev_j = j;
                        predecessor_found = true;
                        break;
                    }
                }
            }
        }

        // 3. Insertion / gap along the sequence axis (:447-451) — lowest precedence. The
        // `j != 0` guard mirrors the affine/convex backtracks: at the left edge column 0 is
        // always reconstructed by the deletion step above, so this branch is unreachable
        // there for well-formed matrices, but the guard keeps `j - 1` from underflowing.
        if !predecessor_found && j != 0 && h_ij == h.get(i, j - 1) + g {
            prev_i = i;
            prev_j = j - 1;
        }

        let node_slot = if i == prev_i {
            -1
        } else {
            graph.rank_to_node[i - 1].0 as i32
        };
        let seq_slot = if j == prev_j { -1 } else { (j - 1) as i32 };
        alignment.push((node_slot, seq_slot));

        i = prev_i;
        j = prev_j;
    }

    alignment.reverse();
    alignment
}

/// Backtracks the optimal alignment under an affine gap penalty from the fill's best cell.
///
/// Ports the backtrack half of `spoa::SisdAlignmentEngine::Affine`
/// (`sisd_alignment_engine.cpp:553-677`) VERBATIM, additionally unwinding affine gap *runs* via
/// `extend_left` (walk the `e` insertion run leftward, `:645-653`) and `extend_up` (walk the `f`
/// deletion run upward across predecessors, `:654-673`). See [`backtrack_linear`] for the
/// `(max_i, max_j, max_score)` contract and the empty-alignment early return.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn backtrack_affine(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &[i32],
    e: &[i32],
    f: &[i32],
    matrix_width: usize,
    alignment_type: AlignmentType,
    scoring: &Scoring,
    max_i: usize,
    max_j: usize,
    max_score: i32,
) -> Alignment {
    backtrack_affine_impl(
        graph,
        node_id_to_rank,
        sequence_profile,
        &RowMajor {
            buf: h,
            width: matrix_width,
        },
        &RowMajor {
            buf: e,
            width: matrix_width,
        },
        &RowMajor {
            buf: f,
            width: matrix_width,
        },
        matrix_width,
        alignment_type,
        scoring,
        max_i,
        max_j,
        max_score,
    )
}

/// The affine backtrack proper, generic over a [`CellRead`] view of `H`/`E`/`F`. [`backtrack_affine`]
/// feeds it [`RowMajor`] views; the SIMD path feeds striped views to skip the destripe (see
/// [`backtrack_convex_impl`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn backtrack_affine_impl<V: CellRead>(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &V,
    e: &V,
    f: &V,
    matrix_width: usize,
    alignment_type: AlignmentType,
    scoring: &Scoring,
    max_i: usize,
    max_j: usize,
    max_score: i32,
) -> Alignment {
    if max_i == 0 && max_j == 0 {
        return Vec::new();
    }
    debug_assert_eq!(
        h.get(max_i, max_j),
        max_score,
        "fill's max_score must match H at (max_i, max_j)"
    );
    let g = i32::from(scoring.g);
    let e_penalty = i32::from(scoring.e);

    // Rank (+1, i.e. the DP row) of an in-edge's tail node (its predecessor).
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };

    let mut alignment: Alignment = Vec::new();
    let mut i = max_i;
    let mut j = max_j;

    loop {
        let keep_going = match alignment_type {
            AlignmentType::Local => h.get(i, j) != 0,
            AlignmentType::Global => !(i == 0 && j == 0),
            AlignmentType::Overlap => !(i == 0 || j == 0),
        };
        if !keep_going {
            break;
        }

        let h_ij = h.get(i, j);
        let mut prev_i = 0usize;
        let mut prev_j = 0usize;
        let mut predecessor_found = false;
        let mut extend_left = false;
        let mut extend_up = false;

        // 1. Match/mismatch (:577-601) — highest precedence.
        if let Some((pi, pj)) = backtrack_match_step(
            graph,
            node_id_to_rank,
            sequence_profile,
            h,
            matrix_width,
            i,
            j,
        ) {
            prev_i = pi;
            prev_j = pj;
            predecessor_found = true;
        }

        // 2. Deletion / gap along the graph axis (:603-627). Faithfully ports the C++
        // `(extend_up = A) || B` idiom: `extend_up` is set ONLY from the F-extend test `A`, and
        // the `||` short-circuits so `B` (the H-open test) is not evaluated when `A` holds.
        if !predecessor_found && i != 0 {
            let node = &graph.nodes[graph.rank_to_node[i - 1].0 as usize];
            let pred_first = if node.inedges.is_empty() {
                0
            } else {
                pred_row(node.inedges[0])
            };
            let a = h_ij == f.get(pred_first, j) + e_penalty;
            extend_up = a;
            if a || h_ij == h.get(pred_first, j) + g {
                prev_i = pred_first;
                prev_j = j;
                predecessor_found = true;
            } else {
                for p in 1..node.inedges.len() {
                    let pred_i = pred_row(node.inedges[p]);
                    let a = h_ij == f.get(pred_i, j) + e_penalty;
                    extend_up = a;
                    if a || h_ij == h.get(pred_i, j) + g {
                        prev_i = pred_i;
                        prev_j = j;
                        predecessor_found = true;
                        break;
                    }
                }
            }
        }

        // 3. Insertion / gap along the sequence axis (:629-636) — lowest precedence. Same
        // `(extend_left = A) || B` short-circuit idiom.
        if !predecessor_found && j != 0 {
            let a = h_ij == e.get(i, j - 1) + e_penalty;
            extend_left = a;
            if a || h_ij == h.get(i, j - 1) + g {
                prev_i = i;
                prev_j = j - 1;
            }
        }

        let node_slot = if i == prev_i {
            -1
        } else {
            graph.rank_to_node[i - 1].0 as i32
        };
        let seq_slot = if j == prev_j { -1 } else { (j - 1) as i32 };
        alignment.push((node_slot, seq_slot));

        i = prev_i;
        j = prev_j;

        // Gap-run unwinding (:645-674).
        if extend_left {
            // Walk the E insertion run leftward until the affine extension no longer holds.
            loop {
                alignment.push((-1, (j - 1) as i32));
                j -= 1;
                if e.get(i, j) + e_penalty != e.get(i, j + 1) {
                    break;
                }
            }
        } else if extend_up {
            // Walk the F deletion run upward. `stop` is set ONLY from the H-open (`+ g`) test, and
            // the loop stops on gap-open or on reaching the boundary row (`i == 0`).
            loop {
                let mut stop = false;
                prev_i = 0;
                for &edge_id in &graph.nodes[graph.rank_to_node[i - 1].0 as usize].inedges {
                    let pred_i = pred_row(edge_id);
                    let s = f.get(i, j) == h.get(pred_i, j) + g;
                    stop = s;
                    if s || f.get(i, j) == f.get(pred_i, j) + e_penalty {
                        prev_i = pred_i;
                        break;
                    }
                }
                alignment.push((graph.rank_to_node[i - 1].0 as i32, -1));
                i = prev_i;
                if stop || i == 0 {
                    break;
                }
            }
        }
    }

    alignment.reverse();
    alignment
}

/// Backtracks the optimal alignment under a convex (double-affine) gap penalty from the fill's
/// best cell.
///
/// Ports the backtrack half of `spoa::SisdAlignmentEngine::Convex`
/// (`sisd_alignment_engine.cpp:781-924`) VERBATIM. Its deletion/insertion steps use a four-term
/// compound condition testing *both* gap functions (`f`+`e` OR `h`+`g` OR `o`+`c` OR `h`+`q`, via
/// the upstream `(extend_* |= ...) || ...` short-circuit idiom, `:837-865`). The gap-run unwinds
/// mirror that: `extend_left` walks leftward while *either* the `e` or `q` run continues
/// (`:879-887`, breaking only when NEITHER does), and `extend_up` walks upward in two phases per
/// step — first try to continue an `f`/`o` extend across predecessors, and only if none continues
/// fall back to an `f`/`o` gap-open scan (`:888-921`). See [`backtrack_linear`] for the
/// `(max_i, max_j, max_score)` contract and the empty-alignment early return.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn backtrack_convex(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &[i32],
    e: &[i32],
    f: &[i32],
    o: &[i32],
    q: &[i32],
    matrix_width: usize,
    alignment_type: AlignmentType,
    scoring: &Scoring,
    max_i: usize,
    max_j: usize,
    max_score: i32,
) -> Alignment {
    backtrack_convex_impl(
        graph,
        node_id_to_rank,
        sequence_profile,
        &RowMajor {
            buf: h,
            width: matrix_width,
        },
        &RowMajor {
            buf: e,
            width: matrix_width,
        },
        &RowMajor {
            buf: f,
            width: matrix_width,
        },
        &RowMajor {
            buf: o,
            width: matrix_width,
        },
        &RowMajor {
            buf: q,
            width: matrix_width,
        },
        matrix_width,
        alignment_type,
        scoring,
        max_i,
        max_j,
        max_score,
    )
}

/// The convex backtrack proper, generic over a [`CellRead`] view of the H/E/F/O/Q matrices.
///
/// [`backtrack_convex`] feeds it [`RowMajor`] views (the scalar/destriped path); the SIMD engine
/// feeds striped views so it can skip the full-matrix destripe and index the striped fill output
/// directly along the (short) backtrack path. `matrix_width` is retained only for the row-major
/// `sequence_profile` read inside [`backtrack_match_step`]; all H/E/F/O/Q reads go through `.get`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backtrack_convex_impl<V: CellRead>(
    graph: &Graph,
    node_id_to_rank: &[u32],
    sequence_profile: &[i32],
    h: &V,
    e: &V,
    f: &V,
    o: &V,
    q: &V,
    matrix_width: usize,
    alignment_type: AlignmentType,
    scoring: &Scoring,
    max_i: usize,
    max_j: usize,
    max_score: i32,
) -> Alignment {
    if max_i == 0 && max_j == 0 {
        return Vec::new();
    }
    debug_assert_eq!(
        h.get(max_i, max_j),
        max_score,
        "fill's max_score must match H at (max_i, max_j)"
    );
    let g = i32::from(scoring.g);
    let e_penalty = i32::from(scoring.e);
    let q_penalty = i32::from(scoring.q);
    let c_penalty = i32::from(scoring.c);

    // Rank (+1, i.e. the DP row) of an in-edge's tail node (its predecessor).
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };

    let mut alignment: Alignment = Vec::new();
    let mut i = max_i;
    let mut j = max_j;

    loop {
        let keep_going = match alignment_type {
            AlignmentType::Local => h.get(i, j) != 0,
            AlignmentType::Global => !(i == 0 && j == 0),
            AlignmentType::Overlap => !(i == 0 || j == 0),
        };
        if !keep_going {
            break;
        }

        let h_ij = h.get(i, j);
        let mut prev_i = 0usize;
        let mut prev_j = 0usize;
        let mut predecessor_found = false;
        let mut extend_left = false;
        let mut extend_up = false;

        // 1. Match/mismatch (:805-829) — highest precedence.
        if let Some((pi, pj)) = backtrack_match_step(
            graph,
            node_id_to_rank,
            sequence_profile,
            h,
            matrix_width,
            i,
            j,
        ) {
            prev_i = pi;
            prev_j = pj;
            predecessor_found = true;
        }

        // 2. Deletion / gap along the graph axis (:831-859). Faithfully ports the four-term
        // `(extend_up |= F-extend) || H-open || (extend_up |= O-extend) || H-2nd-open` compound:
        // `extend_up` accumulates via `|=` ONLY from the two gap-EXTEND tests (F+e, O+c), and the
        // `||` short-circuits so a later test — including the second `extend_up |= ...` — is not
        // evaluated once an earlier term holds. Thus `extend_up` ends true iff a gap-extend matched
        // before any gap-open short-circuited the chain.
        if !predecessor_found && i != 0 {
            let node = &graph.nodes[graph.rank_to_node[i - 1].0 as usize];
            let pred_first = if node.inedges.is_empty() {
                0
            } else {
                pred_row(node.inedges[0])
            };
            let a = h_ij == f.get(pred_first, j) + e_penalty;
            extend_up |= a;
            let cond = a
                || h_ij == h.get(pred_first, j) + g
                || {
                    let b = h_ij == o.get(pred_first, j) + c_penalty;
                    extend_up |= b;
                    b
                }
                || h_ij == h.get(pred_first, j) + q_penalty;
            if cond {
                prev_i = pred_first;
                prev_j = j;
                predecessor_found = true;
            } else {
                for p in 1..node.inedges.len() {
                    let pred_i = pred_row(node.inedges[p]);
                    let a = h_ij == f.get(pred_i, j) + e_penalty;
                    extend_up |= a;
                    let cond = a
                        || h_ij == h.get(pred_i, j) + g
                        || {
                            let b = h_ij == o.get(pred_i, j) + c_penalty;
                            extend_up |= b;
                            b
                        }
                        || h_ij == h.get(pred_i, j) + q_penalty;
                    if cond {
                        prev_i = pred_i;
                        prev_j = j;
                        predecessor_found = true;
                        break;
                    }
                }
            }
        }

        // 3. Insertion / gap along the sequence axis (:861-870) — lowest precedence. Same
        // four-term `(extend_left |= E-extend) || H-open || (extend_left |= Q-extend) ||
        // H-2nd-open` short-circuit idiom.
        if !predecessor_found && j != 0 {
            let a = h_ij == e.get(i, j - 1) + e_penalty;
            extend_left |= a;
            let cond = a
                || h_ij == h.get(i, j - 1) + g
                || {
                    let b = h_ij == q.get(i, j - 1) + c_penalty;
                    extend_left |= b;
                    b
                }
                || h_ij == h.get(i, j - 1) + q_penalty;
            if cond {
                prev_i = i;
                prev_j = j - 1;
            }
        }

        let node_slot = if i == prev_i {
            -1
        } else {
            graph.rank_to_node[i - 1].0 as i32
        };
        let seq_slot = if j == prev_j { -1 } else { (j - 1) as i32 };
        alignment.push((node_slot, seq_slot));

        i = prev_i;
        j = prev_j;

        // Gap-run unwinding (:879-921).
        if extend_left {
            // Walk the E/Q insertion run leftward; stop only when NEITHER the E run nor the Q run
            // continues (`&&` in the break condition, :883-886).
            loop {
                alignment.push((-1, (j - 1) as i32));
                j -= 1;
                let e_stops = e.get(i, j) + e_penalty != e.get(i, j + 1);
                let q_stops = q.get(i, j) + c_penalty != q.get(i, j + 1);
                if e_stops && q_stops {
                    break;
                }
            }
        } else if extend_up {
            // Walk the F/O deletion run upward in two phases per step (:889-919). `stop` starts
            // true; Phase A clears it if an F- or O-extend continues across some predecessor;
            // Phase B (a gap-open scan) runs ONLY when no extend was found.
            loop {
                let mut stop = true;
                prev_i = 0;
                // Phase A: try to continue an F or O extend.
                for &edge_id in &graph.nodes[graph.rank_to_node[i - 1].0 as usize].inedges {
                    let pred_i = pred_row(edge_id);
                    if f.get(i, j) == f.get(pred_i, j) + e_penalty
                        || o.get(i, j) == o.get(pred_i, j) + c_penalty
                    {
                        prev_i = pred_i;
                        stop = false;
                        break;
                    }
                }
                // Phase B: fall back to an F or O gap-open scan.
                if stop {
                    for &edge_id in &graph.nodes[graph.rank_to_node[i - 1].0 as usize].inedges {
                        let pred_i = pred_row(edge_id);
                        if f.get(i, j) == h.get(pred_i, j) + g
                            || o.get(i, j) == h.get(pred_i, j) + q_penalty
                        {
                            prev_i = pred_i;
                            break;
                        }
                    }
                }
                alignment.push((graph.rank_to_node[i - 1].0 as i32, -1));
                i = prev_i;
                if stop || i == 0 {
                    break;
                }
            }
        }
    }

    alignment.reverse();
    alignment
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

    /// A single-node graph ("A") aligned globally against `"A"` under linear scoring: a trivial
    /// 2x2 matrix (`matrix_width = 2`) with one match cell. Hand-computed:
    /// `H = [[0, -8], [-8, 5]]` (row 0/col 0 are the NW boundary; `H[1][1] = 5` is the match).
    #[test]
    fn backtrack_linear_matches_single_node_on_hand_built_matrix() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"A", 1).unwrap();

        let scoring = linear_scoring();
        let node_id_to_rank = vec![0u32];
        // sequence_profile[code * matrix_width + j]; only one code ('A') is used here:
        // [j=0 boundary (unused), j=1 match].
        let sequence_profile = vec![0, 5];
        let matrix_width = 2;
        let h = vec![0, -8, -8, 5];

        let alignment = backtrack_linear(
            &g,
            &node_id_to_rank,
            &sequence_profile,
            &h,
            matrix_width,
            AlignmentType::Global,
            &scoring,
            1,
            1,
            5,
        );

        assert_eq!(alignment, vec![(0, 0)]);
    }

    /// A single-node graph ("A") aligned globally against a length-2 sequence under affine
    /// scoring (`g=-8, e=-6`), with hand-picked `H`/`E`/`F` buffers engineered so that
    /// `(max_i, max_j) = (1, 2)`'s only explanation is an `E`-run gap-open (forcing
    /// `extend_left`), which then unwinds one more step before the row-0 boundary is reached via
    /// a graph-axis deletion. This exercises [`backtrack_affine`]'s `extend_left` unwind loop
    /// independent of any real DP fill.
    #[test]
    fn backtrack_affine_unwinds_e_run_then_deletes() {
        use crate::align::sisd::NEG_INF;

        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"A", 1).unwrap();
        let node_id = g.rank_to_node[0].0 as i32;

        let scoring = affine_scoring(); // g = -8, e = -6
        let node_id_to_rank = vec![0u32];
        // sequence_profile[code * matrix_width + j]; only code 0 ("A") is used. j=1 -> match (5).
        let sequence_profile = vec![0, 5, 5];
        let matrix_width = 3;

        // H[0][*] = NW boundary row (j * g): [0, -8, -16].
        // H[1][0] = 0 + g = -8 (deletion from the boundary column).
        // H[1][2] = -26, engineered to equal E[1][1] + e (-20 + -6) so the E-run-open test fires,
        // and to NOT equal the match-step or gap-open-deletion alternatives at (1, 2).
        let h = vec![0, -8, -16, -8, NEG_INF, -26];
        // E[1][0] and E[1][1] are set so the unwind's continuation check fails immediately after
        // one step: E[1][0] + e (0 + -6 = -6) != E[1][1] (-20).
        let e = vec![0, -8, -16, 0, -20, NEG_INF];
        let f = vec![0, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF];

        let alignment = backtrack_affine(
            &g,
            &node_id_to_rank,
            &sequence_profile,
            &h,
            &e,
            &f,
            matrix_width,
            AlignmentType::Global,
            &scoring,
            1,
            2,
            -26,
        );

        assert_eq!(alignment, vec![(node_id, -1), (-1, 0), (-1, 1)]);
    }

    /// A single-node graph ("A") aligned globally against `"A"` under convex scoring: identical
    /// shape to the linear hand case, since with no gaps neither `O`/`Q` matters for the traceback
    /// of the match step itself; asserts the convex signature/wiring by using non-trivial (but
    /// unreachable-during-this-trace) `O`/`Q` buffers.
    #[test]
    fn backtrack_convex_matches_single_node_on_hand_built_matrix() {
        use crate::align::sisd::NEG_INF;

        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"A", 1).unwrap();

        let scoring = convex_scoring();
        let node_id_to_rank = vec![0u32];
        let sequence_profile = vec![0, 5];
        let matrix_width = 2;
        let h = vec![0, -8, -8, 5];
        let e = vec![0, -8, NEG_INF, NEG_INF];
        let f = vec![0, NEG_INF, NEG_INF, NEG_INF];
        let o = vec![0, NEG_INF, NEG_INF, NEG_INF];
        let q = vec![0, -10, NEG_INF, NEG_INF];

        let alignment = backtrack_convex(
            &g,
            &node_id_to_rank,
            &sequence_profile,
            &h,
            &e,
            &f,
            &o,
            &q,
            matrix_width,
            AlignmentType::Global,
            &scoring,
            1,
            1,
            5,
        );

        assert_eq!(alignment, vec![(0, 0)]);
    }
}
