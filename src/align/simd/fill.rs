//! The generic, vectorized DP fills the SIMD engine runs before destriping into the shared scalar
//! backtrack. This module holds all three gap-mode fills — [`fill_linear`], [`fill_affine`], and
//! [`fill_convex`].
//!
//! Ports `spoa::SimdAlignmentEngine<A>::Linear`'s FILL half
//! (`third_party/spoa/src/simd_alignment_engine_implementation.hpp:727-898`) as a single function
//! generic over the [`Simd`] trait, so one implementation serves every ISA backend (SSE4.1 and AVX2
//! on x86_64, NEON on aarch64). The recurrence computed per DP cell is identical to the scalar
//! [`crate::align::sisd::SisdEngine::linear`] fill (`sisd_alignment_engine.cpp:295-363`) — this is
//! just its lane-parallel, striped form:
//!
//! - Each graph row is `matrix_width_vecs = ceil(seq_len / LANES)` striped vectors; segment `j`,
//!   lane `k` holds the DP value at 0-based query position `j * LANES + k` (row-major column
//!   `j * LANES + k + 1`). There is no striped column 0 — the column-0 boundary lives in the
//!   scalar `first_column` (`seeded.h[row * matrix_width]`, the SIMD kernels plan's "C2 fix").
//! - **Diagonal + vertical** (`impl:783-795`): the diagonal is `H_pred` shifted up one lane
//!   ([`Simd::slli_one_lane`]) with the inter-segment carry `x` OR'd into lane 0, plus the char
//!   profile; the vertical is `H_pred + g`; take the lane-wise `max`. Additional predecessors fold
//!   in with another `max` (`impl:796-821`).
//! - **Horizontal gap** (`impl:830-846`): resolve the left-gap dependency within a vector via
//!   [`Simd::prefix_max`], carrying the previous segment's top lane forward through `x` and the
//!   carry mask.
//! - **Per-type max tracking** (`impl:848-898`): NW takes the last query column's value at each
//!   sink node; SW reduces every row (clamped at 0, [`Simd::horizontal_max`]); OV reduces only
//!   sink-node rows (any column) but, unlike SW/upstream's SIMD engine, WITHOUT the 0 clamp
//!   ([`row_max`] — see its doc for why `SisdEngine` parity requires diverging from upstream
//!   here). `max_i` uses a STRICT `<` (ties keep the earlier row), and `max_j` is resolved after
//!   the sweep. A "no cell selected" result maps to the `(0, 0)` early-return the shared backtrack
//!   treats as an empty alignment (`sisd.rs`'s `max_i == 0 && max_j == 0` guard, mirroring
//!   `impl:875-877`).
//!
//! The caller ([`super::SimdEngine`]) then [`super::profile::destripe_interior`]s the returned
//! striped `H` over the seeded row-major buffer and runs [`crate::align::backtrack::backtrack_linear`].

use super::band::BandState;
use super::lanes::Simd;
use super::profile::{ElemFromI32, ElemToI32};
use crate::align::sisd::{ScalarInit, NEG_INF};
use crate::align::{AlignmentType, Scoring};
use crate::graph::{EdgeId, Graph, NodeId};

/// Extracts lane `lane` of `v`, widened to `i32`. Ports `_mmxxx_value_at` (`impl:253-258`):
/// store-then-index, used for the NW per-row score at the last query column.
#[inline]
fn value_at<S>(v: S::Vec, lane: usize) -> i32
where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    let mut buf = vec![S::Elem::from_i32(0); S::LANES];
    S::storeu(v, &mut buf);
    buf[lane].to_i32()
}

/// The true (non-clamped) maximum of `v`'s lanes, widened to `i32`.
///
/// [`Simd::horizontal_max`] intentionally floors at `0` to port `_mmxxx_max_value`'s Smith-Waterman
/// clamp (`impl:240-250`) — upstream's SIMD `Linear` reuses that SAME floored reduction for the
/// Overlap row-score too (`impl:850-856` calls it for `kSW`; the `kOV` branch immediately below,
/// `impl:858-862`, calls the identical `_mmxxx_max_value`). But `SisdEngine`'s scalar Overlap fill
/// has no such floor — it just compares the raw `H_row[j]` against the running max
/// (`sisd_alignment_engine.cpp:305-310,359-360`), so a genuinely negative Overlap row max (e.g. a
/// single-base mismatch against a free-leading-gap boundary) must survive as negative. Since the
/// SIMD kernels plan's acceptance bar is bit-exactness against [`crate::align::sisd::SisdEngine`]
/// (not against upstream's SIMD engine), Overlap's row-score reduction uses this unclamped max
/// instead of [`Simd::horizontal_max`]. Padding lanes are always [`Simd::NEG_INF`], far below any
/// in-range real DP value (the [`super::super::Escalation`] guard guarantees this), so folding them
/// in via a plain (non-floored) max is safe.
#[inline]
fn row_max<S>(v: S::Vec) -> i32
where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    let mut buf = vec![S::Elem::from_i32(0); S::LANES];
    S::storeu(v, &mut buf);
    buf.iter()
        .map(|elem| elem.to_i32())
        .max()
        .expect("LANES >= 1")
}

/// The lowest flat index (`segment * LANES + lane`, i.e. a 0-based query position) in `row` whose
/// lane equals `value`, or `-1` if none. Ports `_mmxxx_index_of` (`impl:261-289`) — the SW/OV
/// `max_j` resolver. Trailing padding lanes always sit at flat indices `>= seq_len`, strictly above
/// every real position, so a real cell achieving `value` is always found first.
#[inline]
fn index_of<S>(row: &[S::Vec], row_width: usize, value: i32) -> i32
where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    let mut buf = vec![S::Elem::from_i32(0); S::LANES];
    for (segment, &vec) in row.iter().take(row_width).enumerate() {
        S::storeu(vec, &mut buf);
        for (lane, &elem) in buf.iter().enumerate() {
            if elem.to_i32() == value {
                return (segment * S::LANES + lane) as i32;
            }
        }
    }
    -1
}

/// Seeds striped row 0 (the boundary row) of `striped` from the row-major boundary row `row_major`
/// (`row_major[pos + 1]` is row 0's column `pos + 1`); trailing lanes at or past `seq_len` get
/// [`Simd::NEG_INF`]. Shared by the affine fill's H and F boundary-row seeding (the linear fill
/// inlines the same loop for H alone).
///
/// The scalar boundary buffers carry `SisdEngine`'s *i32*-scaled [`NEG_INF`] sentinel
/// (`i32::MIN + 1024`) in cells like the affine F boundary row's `j >= 1` entries; a naive
/// `value as i16` truncates that to `1024` (a large *positive*), so any i32 value at or below the
/// scalar sentinel is mapped to the lane type's own [`Simd::NEG_INF`] instead. Real boundary values
/// (finite gap costs, well within the int16 range under the [`super::super::Escalation`] guard)
/// cast losslessly. The linear fill's H boundary never reaches the sentinel, which is why it never
/// needed this guard.
#[inline]
fn seed_striped_row0<S>(
    striped: &mut [S::Vec],
    row_major: &[i32],
    matrix_width_vecs: usize,
    seq_len: usize,
) where
    S: Simd,
    S::Elem: ElemFromI32,
{
    let lanes = S::LANES;
    let mut lane_buf = vec![S::Elem::from_i32(0); lanes];
    for (segment, slot) in striped.iter_mut().take(matrix_width_vecs).enumerate() {
        for (k, lane) in lane_buf.iter_mut().enumerate() {
            let pos = segment * lanes + k;
            *lane = if pos >= seq_len || row_major[pos + 1] <= NEG_INF {
                S::NEG_INF
            } else {
                S::Elem::from_i32(row_major[pos + 1])
            };
        }
        *slot = S::loadu(&lane_buf);
    }
}

/// Computes a graph row's banded window as striped segments `(beg_sn, end_sn, beg_col)`, where
/// `beg_col = beg_sn * lanes` is the first query column of the first in-band segment (needed only to
/// translate an in-band [`index_of`] result back into an absolute `best_col`).
///
/// `None` yields the full row `(0, matrix_width_vecs, 0)`, so callers reduce to the byte-identical
/// unbanded path (the loops and carry seeds below become byte-identical to the pre-band code).
/// `Some` derives the window from the node's anchor unioned with its predecessors' recorded
/// `best_col` (§Band computation of the design doc), then floors/ceils to segment boundaries via
/// [`super::band::segment_range`]. Shared verbatim by [`fill_linear`] and [`fill_affine`] (DRY) so
/// the two fills clip identically; the test helper `recompute_window` mirrors this derivation.
#[inline]
fn row_band(
    band: Option<&BandState>,
    graph: &Graph,
    node_id: NodeId,
    node_id_to_rank: &[u32],
    seq_len: usize,
    lanes: usize,
    matrix_width_vecs: usize,
) -> (usize, usize, usize) {
    match band {
        Some(state) => {
            let node = &graph.nodes[node_id.0 as usize];
            let rank = node_id_to_rank[node_id.0 as usize] as usize;
            let anchor = super::band::anchor(state.r[rank], seq_len);
            let (mstart, mend) = if node.inedges.is_empty() {
                (anchor, anchor)
            } else {
                let mut lo = usize::MAX;
                let mut hi = 0usize;
                for &edge_id in &node.inedges {
                    let tail = graph.edges[edge_id.0 as usize].tail;
                    let pr = node_id_to_rank[tail.0 as usize] as usize;
                    let bc = state.best_col[pr] as usize;
                    lo = lo.min(bc);
                    hi = hi.max(bc);
                }
                (lo, hi)
            };
            let (beg, end) = super::band::node_window(anchor, mstart, mend, state.w, seq_len);
            let (bs, es) = super::band::segment_range(beg, end, lanes, matrix_width_vecs);
            (bs, es, bs * lanes)
        }
        None => (0, matrix_width_vecs, 0),
    }
}

/// Runs the vectorized linear-gap DP fill for `seq` (length `seq_len`) against `graph`.
///
/// Returns the striped `H` matrix (`matrix_height * matrix_width_vecs` vectors, row `r`'s block at
/// `[r * matrix_width_vecs ..]`; **row 0 is the seeded boundary row**, rows `1..` are the interior
/// the caller destripes) together with the fill's `(max_i, max_j, max_score)` start cell for the
/// shared scalar backtrack. `max_i`/`max_j` are in row-major scalar coordinates (`max_j` is a
/// column into a `seq_len + 1`-wide row); `(0, 0)` signals "no cell selected".
///
/// `seeded` supplies the column-0 `first_column` seed and the boundary row (its `h`'s row 0 /
/// column 0), `profile` is the striped char profile ([`super::profile::build_profile`]), and
/// `masks`/`penalties` are the prefix-max ladder inputs ([`super::profile::build_masks`] /
/// [`super::profile::build_penalties`]). Ports `impl:727-898`.
///
/// The caller guarantees `seq_len >= 1` and a non-empty `graph` (both short-circuited earlier in
/// [`super::SimdEngine::align`]), so `matrix_width_vecs >= 1`.
///
/// # Banded mode (`band = Some(..)`)
///
/// When `band` is supplied each graph row computes only the striped segments `[beg_sn, end_sn)` of
/// its abPOA-style window (§Fill clip of the design doc); out-of-band segments keep their
/// resize-refilled [`Simd::NEG_INF`] init. `None` reproduces today's full-matrix behavior **exactly**
/// (`(0, matrix_width_vecs, 0)` window, no carry-seed change), which the parity suite pins.
///
/// The clip's correctness crux is that the two lane-0 carries are seeded **oppositely** at the band's
/// left edge (`beg_sn > 0`):
///
/// - **Diagonal carry** (the first-/additional-predecessor pass) — seeded from the predecessor buffer
///   at segment `beg_sn - 1` (`srli_top_lane(striped_h[pred_base + beg_sn - 1])`), NOT closed. That
///   cell is normally inside the predecessor's overlapping band and holds the real match transition
///   `H_pred[beg-1] + s`, the dominant term for near-identical reads; closing it to `NEG_INF` would
///   discard that transition and make left-edge in-band cells differ from exact. Because the buffer is
///   full-width and refilled to `NEG_INF`, this naturally yields `NEG_INF` iff that predecessor cell
///   is itself out of band — the correct behavior with no special case.
/// - **Horizontal carry** (the `prefix_max` pass) — closed to `NEG_INF` (`splat(NEG_INF)`), ksw2-style
///   edge closure. It represents *this row's own* running value at column `beg-1`, which a banded row
///   genuinely never computes (and which is not the boundary column 0); injecting any finite value
///   would fabricate a horizontal gap from column 0 with the skipped-segment penalties missing.
///
/// The banded arithmetic uses saturating [`Simd::adds`] so an interior out-of-band `NEG_INF` sentinel,
/// which a drifting band edge can penalize across many rows (`NEG_INF + k·g`), stays pinned at
/// `NEG_INF` instead of wrapping int16/int32 to a large positive and winning a `max`. On real
/// (non-sentinel) DP values `adds` is bit-identical to `add` (the escalation guard keeps real values
/// far from the saturation threshold), so it is used uniformly on both the `None` and `Some` paths —
/// the parity suite confirms `None` stays bit-exact.
///
/// SW/OV max-tracking (and the recorded `best_col`) reduce over the in-band segments only. Global
/// endpoint handling is unchanged here (a later task makes it band-aware); the banded gate targets
/// [`AlignmentType::Local`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn fill_linear<S>(
    graph: &Graph,
    seq_len: usize,
    scoring: Scoring,
    alignment_type: AlignmentType,
    seeded: &ScalarInit,
    profile: &[S::Vec],
    masks: &[S::Vec],
    penalties: &[S::Vec],
    striped_h: &mut Vec<S::Vec>,
    mut band: Option<&mut BandState>,
) -> (usize, usize, i32)
where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    let lanes = S::LANES;
    let matrix_width_vecs = seq_len.div_ceil(lanes);
    let matrix_width = seeded.matrix_width; // seq_len + 1 (row-major width)
    let matrix_height = graph.nodes.len() + 1;
    let node_id_to_rank = &seeded.node_id_to_rank;

    let g_vec = S::splat(S::Elem::from_i32(i32::from(scoring.g)));
    let zeroes = S::splat(S::Elem::from_i32(0));
    // The inter-segment carry mask (`masks[LOG_LANES]`, `impl:750-752`): lane 0 = 0, rest neg-inf.
    let carry_mask = masks[S::LOG_LANES as usize];
    // Row-major column 0 of DP row `r` — spoa's `first_column[r]` (the SIMD kernels plan's C2 fix).
    let first_column = |r: usize| -> i32 { seeded.h[r * matrix_width] };
    // Rank (+1, i.e. the striped DP row) of an in-edge's tail node.
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };

    // Striped H, all rows seeded to neg-inf (reusing the caller's grow-only buffer — clear keeps
    // the allocation, resize refills every used cell with neg-inf, so no stale value from a prior
    // larger align survives); row 0 then overwritten with the boundary row.
    let cells = matrix_height * matrix_width_vecs;
    striped_h.clear();
    striped_h.resize(cells, S::splat(S::NEG_INF));
    {
        let mut lane_buf = vec![S::Elem::from_i32(0); lanes];
        for (segment, slot) in striped_h.iter_mut().take(matrix_width_vecs).enumerate() {
            for (k, lane) in lane_buf.iter_mut().enumerate() {
                let pos = segment * lanes + k;
                *lane = if pos < seq_len {
                    S::Elem::from_i32(seeded.h[pos + 1]) // row 0, column pos + 1
                } else {
                    S::NEG_INF
                };
            }
            *slot = S::loadu(&lane_buf);
        }
    }

    let mut max_score: i32 = match alignment_type {
        AlignmentType::Local => 0,
        AlignmentType::Global | AlignmentType::Overlap => NEG_INF,
    };
    let mut max_i: usize = 0; // 0 == "not found" (real rows are >= 1)
    let last_column_id = (seq_len - 1) % lanes;

    for &node_id in &graph.rank_to_node {
        let node = &graph.nodes[node_id.0 as usize];
        let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
        let profile_base = node.code as usize * matrix_width_vecs;
        let row_base = i * matrix_width_vecs;

        // This row's banded window in striped segments `[beg_sn, end_sn)` with its first query column
        // `beg_col` (see [`row_band`]). `None` yields the full row `(0, matrix_width_vecs, 0)`, so the
        // loops below and the carry seeds are byte-identical to the unbanded path.
        let (beg_sn, end_sn, beg_col) = row_band(
            band.as_deref(),
            graph,
            node_id,
            node_id_to_rank,
            seq_len,
            lanes,
            matrix_width_vecs,
        );

        // First predecessor: diagonal (with carry) + vertical (impl:773-795).
        let mut pred_i = if node.inedges.is_empty() {
            0
        } else {
            pred_row(node.inedges[0])
        };
        let pred_base = pred_i * matrix_width_vecs;
        // DIAGONAL carry: at the band's left edge seed from the PREDECESSOR buffer's top lane at
        // segment `beg_sn - 1` (real value, or `NEG_INF` if that pred cell is itself out of band).
        // Do NOT close to `NEG_INF` — that would drop the match transition `H_pred[beg-1] + s`.
        let mut x = if beg_sn == 0 {
            S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))))
        } else {
            S::srli_top_lane(striped_h[pred_base + (beg_sn - 1)])
        };
        for j in beg_sn..end_sn {
            let h_pred = striped_h[pred_base + j];
            let t1 = S::srli_top_lane(h_pred);
            let diag = S::or(S::slli_one_lane(h_pred), x);
            x = t1;
            let value = S::max(
                S::adds(diag, profile[profile_base + j]),
                S::adds(h_pred, g_vec),
            );
            striped_h[row_base + j] = value;
        }

        // Additional predecessors: fold in with max (impl:796-821). Same diagonal-carry seeding.
        for p in 1..node.inedges.len() {
            pred_i = pred_row(node.inedges[p]);
            let pred_base = pred_i * matrix_width_vecs;
            let mut x = if beg_sn == 0 {
                S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))))
            } else {
                S::srli_top_lane(striped_h[pred_base + (beg_sn - 1)])
            };
            for j in beg_sn..end_sn {
                let h_pred = striped_h[pred_base + j];
                let t1 = S::srli_top_lane(h_pred);
                let m = S::or(S::slli_one_lane(h_pred), x);
                x = t1;
                let cur = striped_h[row_base + j];
                let candidate = S::max(
                    S::adds(m, profile[profile_base + j]),
                    S::adds(h_pred, g_vec),
                );
                striped_h[row_base + j] = S::max(cur, candidate);
            }
        }

        // Horizontal gap (prefix_max) + inter-segment carry, then the per-row score (impl:823-846).
        // HORIZONTAL carry: at the band's left edge close to `NEG_INF` (ksw2-style edge closure) — the
        // column `beg-1` is genuinely uncomputed in THIS row and is not the boundary column 0.
        let mut score = S::splat(S::NEG_INF);
        let mut x = if beg_sn == 0 {
            S::srli_top_lane(S::add(S::splat(S::Elem::from_i32(first_column(i))), g_vec))
        } else {
            S::splat(S::NEG_INF)
        };
        for j in beg_sn..end_sn {
            let mut hv = striped_h[row_base + j];
            hv = S::max(hv, S::or(x, carry_mask));
            // NOTE: trait arg order is (penalties, masks) — transposed vs upstream's (masks,
            // penalties) — see the `Simd::prefix_max` doc and the Task 6 review.
            hv = S::prefix_max(hv, penalties, masks);
            x = S::srli_top_lane(S::adds(hv, g_vec));
            if alignment_type == AlignmentType::Local {
                hv = S::max(hv, zeroes);
            }
            striped_h[row_base + j] = hv;
            score = S::max(score, hv);
        }

        // Record this row's max query column into `best_col[rank]` (banded only). Uses the
        // `LANES`-independent `index_of` flat-scan over the IN-BAND slice (determinism across ISAs,
        // §best_col propagation), offset by `beg_col`; the next row's Mstart/Mend read it.
        if let Some(state) = band.as_deref_mut() {
            let rank = node_id_to_rank[node_id.0 as usize] as usize;
            let row_best = S::horizontal_max(score).to_i32();
            let col = index_of::<S>(
                &striped_h[row_base + beg_sn..row_base + end_sn],
                end_sn - beg_sn,
                row_best,
            );
            state.best_col[rank] = (beg_col as i32 + col).max(0) as u32;
        }

        // Per-type max-score tracking (impl:848-872). STRICT `<` keeps the earlier row on a tie.
        match alignment_type {
            AlignmentType::Local => {
                let row_score = S::horizontal_max(score).to_i32();
                if max_score < row_score {
                    max_score = row_score;
                    max_i = i;
                }
            }
            AlignmentType::Overlap => {
                if node.outedges.is_empty() {
                    let row_score = row_max::<S>(score);
                    if max_score < row_score {
                        max_score = row_score;
                        max_i = i;
                    }
                }
            }
            AlignmentType::Global => {
                if node.outedges.is_empty() {
                    let last = striped_h[row_base + (matrix_width_vecs - 1)];
                    let row_score = value_at::<S>(last, last_column_id);
                    if max_score < row_score {
                        max_score = row_score;
                        max_i = i;
                    }
                }
            }
        }
    }

    // Resolve max_j (impl:875-898). "Not found" -> (0, 0), the backtrack's empty-alignment guard.
    if max_i == 0 {
        return (0, 0, max_score);
    }
    let max_j = match alignment_type {
        // NW: last query column. Scalar row-major column = normal_matrix_width = seq_len.
        AlignmentType::Global => seq_len,
        // SW/OV: the lowest column achieving max_score. index_of returns a 0-based query position;
        // the scalar row-major column is one greater (column 0 is the boundary).
        AlignmentType::Local | AlignmentType::Overlap => {
            let row = &striped_h[max_i * matrix_width_vecs..(max_i + 1) * matrix_width_vecs];
            let idx = index_of::<S>(row, matrix_width_vecs, max_score);
            if idx < 0 {
                return (0, 0, max_score);
            }
            idx as usize + 1
        }
    };

    (max_i, max_j, max_score)
}

/// Runs the vectorized affine-gap DP fill for `seq` (length `seq_len`) against `graph`.
///
/// Fills the caller-owned (grow-only, reused-across-calls) striped `H`, `E` (horizontal-gap) and
/// `F` (vertical-gap) matrix buffers (each grown to `matrix_height * matrix_width_vecs` vectors and
/// fully reset to neg-inf, **row 0 then overwritten with the boundary row** — H's and F's are read
/// as predecessors of the first interior rows; E's row 0 is unused) and returns the fill's `(max_i,
/// max_j, max_score)` start cell for the shared scalar [`backtrack_affine`]. The coordinate/`(0, 0)`
/// conventions match [`fill_linear`].
///
/// Ports `spoa::SimdAlignmentEngine<A>::Affine`'s FILL half
/// (`third_party/spoa/src/simd_alignment_engine_implementation.hpp:1119-1239`) — the lane-parallel,
/// striped form of the scalar [`crate::align::sisd::SisdEngine::affine`] recurrence
/// (`sisd_alignment_engine.cpp:487-573`):
///
/// - **F (vertical gap)** has no horizontal dependency, so it is trivially lane-parallel:
///   `F_row[j] = max(H_pred[j] + g, F_pred[j]) + e` for the first predecessor (`impl:1139-1143`),
///   `max`-folded over additional predecessors (`impl:1166-1172`). Upstream splits the affine open
///   penalty as `g = g_ - e_` (added *before* the extend `+ e_`) so a single `+ e` covers both the
///   open (`g_`) and extend (`e_`) terms; this port keeps that exact split.
/// - **H diagonal** comes only from the diagonal now (the vertical is F): `H_row[j] =
///   diag(H_pred) + profile` (`impl:1146-1151`), max-folded over additional predecessors
///   (`impl:1174-1183`).
/// - **E (horizontal gap)** resolves the left-gap dependency via [`Simd::prefix_max`] (the ladder
///   built from the *extend* penalty `e_`, not `g_`) after `H = max(H, F)`, carrying the previous
///   segment's finalized cell forward through `x = max(H, E - g)` (`impl:1188-1206`).
/// - **H = max(H, F, E)** then per-type max tracking (`impl:1191,1203,1208-1238`): NW takes the
///   last query column's value at each sink node; SW/OV reduce rows (OV via the unclamped
///   [`row_max`], see its doc — mirrored from [`fill_linear`] so Task 9b's SW/OV branch is correct).
///
/// # Banded mode (`band = Some(..)`)
///
/// Mirrors [`fill_linear`]'s clip ([`row_band`] gives each row's `[beg_sn, end_sn)` window; the loops
/// iterate it and out-of-band segments keep their resize-refilled [`Simd::NEG_INF`] init), extended to
/// affine's three matrices. `None` reproduces the full-matrix behavior **exactly** (the parity suite
/// pins it). Each matrix's band-edge treatment differs by its dependency direction:
///
/// - **F (vertical gap) — NO carry to close.** `F_row[j]` reads `striped_{h,f}[pred_base + j]`
///   directly (same column, no cross-lane dependency), so an out-of-band predecessor segment already
///   holds `NEG_INF`; there is nothing to seed. The loop simply clips to `[beg_sn, end_sn)`.
/// - **H (diagonal carry) — seed from the predecessor, do NOT close.** Identical to [`fill_linear`]'s
///   diagonal carry, in BOTH the first- and additional-predecessor passes: at the band's left edge
///   (`beg_sn > 0`) seed `x = srli_top_lane(striped_h[pred_base + beg_sn - 1])` (the real match
///   transition `H_pred[beg-1] + s`, or `NEG_INF` iff that pred cell is itself out of band), NOT
///   `NEG_INF`. Closing it would drop the dominant diagonal term and fail the per-cell gate.
/// - **E (horizontal gap) — close to `NEG_INF`.** The `prefix_max` carry `x` is *this row's own*
///   running value at column `beg-1`, which a banded row never computes and which is not the boundary
///   column 0; at `beg_sn > 0` seed `x = splat(NEG_INF)` (ksw2-style edge closure) so no horizontal
///   gap is fabricated across the skipped region. (Affine has no `y` carry; only convex does.)
///
/// As in [`fill_linear`] the banded arithmetic uses saturating [`Simd::adds`]/[`Simd::subs`] so an
/// interior out-of-band `NEG_INF` sentinel stays pinned instead of wrapping to a large positive and
/// winning a `max`; on real DP values these are bit-identical to `add`/`sub` (escalation guard), so
/// they are used uniformly on the `None` and `Some` paths and the parity suite confirms `None` stays
/// bit-exact. SW/OV max-tracking and the recorded `best_col` reduce over the in-band segments only;
/// Global endpoint handling is unchanged here (a later task makes it band-aware), so the banded gate
/// targets [`AlignmentType::Local`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn fill_affine<S>(
    graph: &Graph,
    seq_len: usize,
    scoring: Scoring,
    alignment_type: AlignmentType,
    seeded: &ScalarInit,
    profile: &[S::Vec],
    masks: &[S::Vec],
    penalties: &[S::Vec],
    striped_h: &mut Vec<S::Vec>,
    striped_e: &mut Vec<S::Vec>,
    striped_f: &mut Vec<S::Vec>,
    mut band: Option<&mut BandState>,
) -> (usize, usize, i32)
where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    let lanes = S::LANES;
    let matrix_width_vecs = seq_len.div_ceil(lanes);
    let matrix_width = seeded.matrix_width; // seq_len + 1 (row-major width)
    let matrix_height = graph.nodes.len() + 1;
    let node_id_to_rank = &seeded.node_id_to_rank;

    // Upstream splits the affine open penalty (`impl:1095-1096`): `g = g_ - e_` (added before the
    // F/E extend), `e = e_`. So `max(H + g, F) + e == max(H + g_, F + e_)` and the E open term
    // `slli(H) + g + e == slli(H) + g_`.
    let g_minus_e = S::splat(S::Elem::from_i32(
        i32::from(scoring.g) - i32::from(scoring.e),
    ));
    let e_vec = S::splat(S::Elem::from_i32(i32::from(scoring.e)));
    let zeroes = S::splat(S::Elem::from_i32(0));
    let first_column = |r: usize| -> i32 { seeded.h[r * matrix_width] };
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };

    // Striped H/E/F, all rows reset to neg-inf (reusing the caller's grow-only buffers — clear
    // keeps the allocation, resize refills every used cell, so no stale value from a prior larger
    // align survives); H and F row 0 then overwritten with the boundary rows (read as predecessors
    // of the first interior rows). E's row 0 is never read by the fill.
    let cells = matrix_height * matrix_width_vecs;
    for buf in [&mut *striped_h, &mut *striped_e, &mut *striped_f] {
        buf.clear();
        buf.resize(cells, S::splat(S::NEG_INF));
    }
    seed_striped_row0::<S>(striped_h, &seeded.h, matrix_width_vecs, seq_len);
    seed_striped_row0::<S>(striped_f, &seeded.f, matrix_width_vecs, seq_len);

    let mut max_score: i32 = match alignment_type {
        AlignmentType::Local => 0,
        AlignmentType::Global | AlignmentType::Overlap => NEG_INF,
    };
    let mut max_i: usize = 0; // 0 == "not found" (real rows are >= 1)
    let last_column_id = (seq_len - 1) % lanes;

    for &node_id in &graph.rank_to_node {
        let node = &graph.nodes[node_id.0 as usize];
        let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
        let profile_base = node.code as usize * matrix_width_vecs;
        let row_base = i * matrix_width_vecs;

        // This row's banded window `[beg_sn, end_sn)` and first query column (see [`row_band`]).
        // `None` yields the full row `(0, matrix_width_vecs, 0)`, so the loops and carry seeds below
        // are byte-identical to the unbanded path.
        let (beg_sn, end_sn, beg_col) = row_band(
            band.as_deref(),
            graph,
            node_id,
            node_id_to_rank,
            seq_len,
            lanes,
            matrix_width_vecs,
        );

        // First predecessor: F (vertical) + H (diagonal) in one pass (impl:1127-1152).
        let mut pred_i = if node.inedges.is_empty() {
            0
        } else {
            pred_row(node.inedges[0])
        };
        let mut pred_base = pred_i * matrix_width_vecs;
        // DIAGONAL carry (as in `fill_linear`): at the band's left edge seed from the PREDECESSOR
        // buffer's top lane at segment `beg_sn - 1` (real match transition, or `NEG_INF` if that pred
        // cell is itself out of band). Do NOT close to `NEG_INF` — that would drop `H_pred[beg-1]+s`.
        let mut x = if beg_sn == 0 {
            S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))))
        } else {
            S::srli_top_lane(striped_h[pred_base + (beg_sn - 1)])
        };
        for j in beg_sn..end_sn {
            let h_pred = striped_h[pred_base + j];
            let f_pred = striped_f[pred_base + j];
            // F_row[j] = max(H_pred + g, F_pred) + e. VERTICAL: reads pred column `j` directly, so an
            // out-of-band pred segment already holds `NEG_INF` — no carry to seed.
            striped_f[row_base + j] = S::adds(S::max(S::adds(h_pred, g_minus_e), f_pred), e_vec);
            // H_row[j] = diag(H_pred) + profile (diagonal only; vertical lives in F).
            let diag = S::or(S::slli_one_lane(h_pred), x);
            x = S::srli_top_lane(h_pred);
            striped_h[row_base + j] = S::adds(diag, profile[profile_base + j]);
        }

        // Additional predecessors: fold F and H with max (impl:1154-1184). Same diagonal-carry seeding
        // as the first pass (this is the additional-pred `beg_sn > 0` seed the affine gate guards).
        for p in 1..node.inedges.len() {
            pred_i = pred_row(node.inedges[p]);
            pred_base = pred_i * matrix_width_vecs;
            let mut x = if beg_sn == 0 {
                S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))))
            } else {
                S::srli_top_lane(striped_h[pred_base + (beg_sn - 1)])
            };
            for j in beg_sn..end_sn {
                let h_pred = striped_h[pred_base + j];
                let f_pred = striped_f[pred_base + j];
                let cur_f = striped_f[row_base + j];
                let cand_f = S::adds(S::max(S::adds(h_pred, g_minus_e), f_pred), e_vec);
                striped_f[row_base + j] = S::max(cur_f, cand_f);
                let diag = S::or(S::slli_one_lane(h_pred), x);
                x = S::srli_top_lane(h_pred);
                let cur_h = striped_h[row_base + j];
                let cand_h = S::adds(diag, profile[profile_base + j]);
                striped_h[row_base + j] = S::max(cur_h, cand_h);
            }
        }

        // H = max(H, F); E (horizontal via prefix_max) with the inter-segment x carry; H = max(H,
        // E); per-row score (impl:1186-1212). `x` starts as the full column-0 seed (NOT srli'd) at the
        // band's left edge (`beg_sn > 0`) close the HORIZONTAL carry to `NEG_INF` (ksw2 edge closure):
        // column `beg-1` is genuinely uncomputed in THIS row and is not the boundary column 0.
        let mut score = S::splat(S::NEG_INF);
        let mut x = if beg_sn == 0 {
            S::splat(S::Elem::from_i32(first_column(i)))
        } else {
            S::splat(S::NEG_INF)
        };
        for j in beg_sn..end_sn {
            let hf = S::max(striped_h[row_base + j], striped_f[row_base + j]);
            // E_row[j] = or(slli(H), srli(x)) + g + e  (== left neighbor + g_).
            let e_open = S::adds(
                S::adds(S::or(S::slli_one_lane(hf), S::srli_top_lane(x)), g_minus_e),
                e_vec,
            );
            // NOTE: trait arg order is (penalties, masks) — transposed vs upstream; penalties are
            // built from the EXTEND penalty e_ for affine E (impl:1111), not g_.
            let e_row = S::prefix_max(e_open, penalties, masks);
            striped_e[row_base + j] = e_row;
            let mut hv = S::max(hf, e_row);
            x = S::max(hv, S::subs(e_row, g_minus_e));
            if alignment_type == AlignmentType::Local {
                hv = S::max(hv, zeroes);
            }
            striped_h[row_base + j] = hv;
            score = S::max(score, hv);
        }

        // Record this row's max query column into `best_col[rank]` (banded only), via the
        // `LANES`-independent `index_of` flat-scan over the IN-BAND slice offset by `beg_col`, exactly
        // as `fill_linear` does; the next row's Mstart/Mend read it.
        if let Some(state) = band.as_deref_mut() {
            let rank = node_id_to_rank[node_id.0 as usize] as usize;
            let row_best = S::horizontal_max(score).to_i32();
            let col = index_of::<S>(
                &striped_h[row_base + beg_sn..row_base + end_sn],
                end_sn - beg_sn,
                row_best,
            );
            state.best_col[rank] = (beg_col as i32 + col).max(0) as u32;
        }

        // Per-type max-score tracking (impl:1214-1238). STRICT `<` keeps the earlier row on a tie.
        match alignment_type {
            AlignmentType::Local => {
                let row_score = S::horizontal_max(score).to_i32();
                if max_score < row_score {
                    max_score = row_score;
                    max_i = i;
                }
            }
            AlignmentType::Overlap => {
                if node.outedges.is_empty() {
                    let row_score = row_max::<S>(score);
                    if max_score < row_score {
                        max_score = row_score;
                        max_i = i;
                    }
                }
            }
            AlignmentType::Global => {
                if node.outedges.is_empty() {
                    let last = striped_h[row_base + (matrix_width_vecs - 1)];
                    let row_score = value_at::<S>(last, last_column_id);
                    if max_score < row_score {
                        max_score = row_score;
                        max_i = i;
                    }
                }
            }
        }
    }

    // Resolve max_j (impl:1241-1262). "Not found" -> (0, 0), the backtrack's empty-alignment guard.
    if max_i == 0 {
        return (0, 0, max_score);
    }
    let max_j = match alignment_type {
        AlignmentType::Global => seq_len,
        AlignmentType::Local | AlignmentType::Overlap => {
            let row = &striped_h[max_i * matrix_width_vecs..(max_i + 1) * matrix_width_vecs];
            let idx = index_of::<S>(row, matrix_width_vecs, max_score);
            if idx < 0 {
                return (0, 0, max_score);
            }
            idx as usize + 1
        }
    };

    (max_i, max_j, max_score)
}

/// Runs the vectorized convex-gap DP fill for `seq` (length `seq_len`) against `graph`.
///
/// Fills the caller-owned (grow-only, reused-across-calls) striped `H`, `E`, `F`, `O` and `Q`
/// matrix buffers (each grown to `matrix_height * matrix_width_vecs` vectors and fully reset to
/// neg-inf, **row 0 then overwritten with the boundary row** for H/F/O — read as predecessors of
/// the first interior rows; E's and Q's row 0 are unused, exactly as the scalar convex fill never
/// reads a horizontal-gap matrix's row 0) and returns the fill's `(max_i, max_j, max_score)` start
/// cell for the shared scalar [`backtrack_convex`]. The coordinate/`(0, 0)` conventions match
/// [`fill_linear`].
///
/// Ports `spoa::SimdAlignmentEngine<A>::Convex`'s FILL half
/// (`third_party/spoa/src/simd_alignment_engine_implementation.hpp:1573-1736`) — the lane-parallel,
/// striped form of the scalar [`crate::align::sisd::SisdEngine::convex`] recurrence
/// (`sisd_alignment_engine.cpp:704-771`). Convex is the AFFINE fill with a SECOND affine function
/// running in parallel: `F`/`E` (penalties `g`/`e`) is joined by `O`/`Q` (penalties `q`/`c`), and
/// `H` folds in all four (`H = max(H, max(max(F, E), max(O, Q)))`). Convex is NOT normalized — all
/// four penalties `g`, `e`, `q`, `c` are distinct and used.
///
/// - **F and O (vertical gaps)** have no horizontal dependency, so they are trivially lane-parallel:
///   `F_row[j] = max(H_pred + g, F_pred) + e` and `O_row[j] = max(H_pred + q, O_pred) + c`
///   (`impl:1594-1606`), each `max`-folded over additional predecessors (`impl:1629-1645`). As in
///   [`fill_affine`], upstream splits each open penalty (`g = g_ - e_`, `q = q_ - c_`) so a single
///   `+ e`/`+ c` covers both the open and extend terms; this port keeps that exact split.
/// - **H diagonal** comes only from the diagonal (the verticals are F/O): `H_row[j] = diag(H_pred) +
///   profile` (`impl:1608-1614`), max-folded over additional predecessors (`impl:1647-1656`).
/// - **E (horizontal, first affine)** resolves the left-gap dependency via [`Simd::prefix_max`]
///   (`penalties_e`, built from the extend `e`) after `H = max(H, F, O)`, carrying the previous
///   segment's finalized cell forward through the `x` carry (`impl:1673-1681,1697-1699`).
/// - **Q (horizontal, second affine)** is a SECOND [`Simd::prefix_max`] (`penalties_c`, built from
///   the second extend `c`) with a SEPARATE `y` carry (`impl:1683-1691,1701-1703`).
/// - **H = max(H, max(E, Q))** (folding the two horizontal ladders in after the two verticals — the
///   4-way max overall) then per-type max tracking (`impl:1705-1735`): NW takes the last query
///   column's value at each sink node; SW/OV reduce rows (OV via the unclamped [`row_max`], see its
///   doc). All three types are wired: Global/NW (Task 10a) and Local/SW + Overlap/OV (Task 10b) —
///   this completes the full SSE4.1 int16 engine (all 9 gap-mode x alignment-type combinations).
///
/// `band` is plumbing for the (not-yet-active) banded fill: `None` reproduces today's full-matrix
/// behavior exactly; it is otherwise unused for now.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn fill_convex<S>(
    graph: &Graph,
    seq_len: usize,
    scoring: Scoring,
    alignment_type: AlignmentType,
    seeded: &ScalarInit,
    profile: &[S::Vec],
    masks: &[S::Vec],
    penalties_e: &[S::Vec],
    penalties_c: &[S::Vec],
    striped_h: &mut Vec<S::Vec>,
    striped_e: &mut Vec<S::Vec>,
    striped_f: &mut Vec<S::Vec>,
    striped_o: &mut Vec<S::Vec>,
    striped_q: &mut Vec<S::Vec>,
    band: Option<&mut BandState>,
) -> (usize, usize, i32)
where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    let _ = &band;
    let lanes = S::LANES;
    let matrix_width_vecs = seq_len.div_ceil(lanes);
    let matrix_width = seeded.matrix_width; // seq_len + 1 (row-major width)
    let matrix_height = graph.nodes.len() + 1;
    let node_id_to_rank = &seeded.node_id_to_rank;

    // Upstream splits BOTH convex open penalties (`impl:1541-1544`): the first affine function uses
    // `g = g_ - e_` (added before the F/E extend) with `e = e_`; the second uses `q = q_ - c_`
    // with `c = c_`. So `max(H + g, F) + e == max(H + g_, F + e_)` (and analogously for O/Q), and
    // the E/Q open terms `slli(H) + g + e == slli(H) + g_`, `slli(H) + q + c == slli(H) + q_`.
    let g_minus_e = S::splat(S::Elem::from_i32(
        i32::from(scoring.g) - i32::from(scoring.e),
    ));
    let e_vec = S::splat(S::Elem::from_i32(i32::from(scoring.e)));
    let q_minus_c = S::splat(S::Elem::from_i32(
        i32::from(scoring.q) - i32::from(scoring.c),
    ));
    let c_vec = S::splat(S::Elem::from_i32(i32::from(scoring.c)));
    let zeroes = S::splat(S::Elem::from_i32(0));
    let first_column = |r: usize| -> i32 { seeded.h[r * matrix_width] };
    let pred_row = |edge_id: EdgeId| -> usize {
        let tail = graph.edges[edge_id.0 as usize].tail;
        node_id_to_rank[tail.0 as usize] as usize + 1
    };

    // Striped H/E/F/O/Q, all rows neg-inf; H, F AND O row 0 then overwritten with the boundary rows
    // (read as predecessors of the first interior rows). E's and Q's row 0 are never read by the
    // fill (horizontal-gap matrices have no diagonal/vertical predecessor term), matching how the
    // scalar convex fill never reads e/q row 0.
    let cells = matrix_height * matrix_width_vecs;
    for buf in [
        &mut *striped_h,
        &mut *striped_e,
        &mut *striped_f,
        &mut *striped_o,
        &mut *striped_q,
    ] {
        buf.clear();
        buf.resize(cells, S::splat(S::NEG_INF));
    }
    seed_striped_row0::<S>(striped_h, &seeded.h, matrix_width_vecs, seq_len);
    seed_striped_row0::<S>(striped_f, &seeded.f, matrix_width_vecs, seq_len);
    seed_striped_row0::<S>(striped_o, &seeded.o, matrix_width_vecs, seq_len);

    let mut max_score: i32 = match alignment_type {
        AlignmentType::Local => 0,
        AlignmentType::Global | AlignmentType::Overlap => NEG_INF,
    };
    let mut max_i: usize = 0; // 0 == "not found" (real rows are >= 1)
    let last_column_id = (seq_len - 1) % lanes;

    for &node_id in &graph.rank_to_node {
        let node = &graph.nodes[node_id.0 as usize];
        let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
        let profile_base = node.code as usize * matrix_width_vecs;
        let row_base = i * matrix_width_vecs;

        // First predecessor: F and O (verticals) + H (diagonal) in one pass (impl:1582-1615).
        let mut pred_i = if node.inedges.is_empty() {
            0
        } else {
            pred_row(node.inedges[0])
        };
        let mut pred_base = pred_i * matrix_width_vecs;
        let mut x = S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))));
        for j in 0..matrix_width_vecs {
            let h_pred = striped_h[pred_base + j];
            let f_pred = striped_f[pred_base + j];
            let o_pred = striped_o[pred_base + j];
            // F_row[j] = max(H_pred + g, F_pred) + e; O_row[j] = max(H_pred + q, O_pred) + c.
            striped_f[row_base + j] = S::add(S::max(S::add(h_pred, g_minus_e), f_pred), e_vec);
            striped_o[row_base + j] = S::add(S::max(S::add(h_pred, q_minus_c), o_pred), c_vec);
            // H_row[j] = diag(H_pred) + profile (diagonal only; verticals live in F and O).
            let diag = S::or(S::slli_one_lane(h_pred), x);
            x = S::srli_top_lane(h_pred);
            striped_h[row_base + j] = S::add(diag, profile[profile_base + j]);
        }

        // Additional predecessors: fold F, O and H with max (impl:1617-1657).
        for p in 1..node.inedges.len() {
            pred_i = pred_row(node.inedges[p]);
            pred_base = pred_i * matrix_width_vecs;
            let mut x = S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))));
            for j in 0..matrix_width_vecs {
                let h_pred = striped_h[pred_base + j];
                let f_pred = striped_f[pred_base + j];
                let o_pred = striped_o[pred_base + j];
                let cur_f = striped_f[row_base + j];
                let cand_f = S::add(S::max(S::add(h_pred, g_minus_e), f_pred), e_vec);
                striped_f[row_base + j] = S::max(cur_f, cand_f);
                let cur_o = striped_o[row_base + j];
                let cand_o = S::add(S::max(S::add(h_pred, q_minus_c), o_pred), c_vec);
                striped_o[row_base + j] = S::max(cur_o, cand_o);
                let diag = S::or(S::slli_one_lane(h_pred), x);
                x = S::srli_top_lane(h_pred);
                let cur_h = striped_h[row_base + j];
                let cand_h = S::add(diag, profile[profile_base + j]);
                striped_h[row_base + j] = S::max(cur_h, cand_h);
            }
        }

        // H = max(H, max(F, O)); E and Q (horizontal via two prefix_max ladders) with the DUAL x/y
        // carries; H = max(H, max(E, Q)); per-row score (impl:1660-1709). Both `x` and `y` start as
        // the full column-0 seed (NOT srli'd — the srli happens inside the loop).
        let mut score = S::splat(S::NEG_INF);
        let mut x = S::splat(S::Elem::from_i32(first_column(i)));
        let mut y = S::splat(S::Elem::from_i32(first_column(i)));
        for j in 0..matrix_width_vecs {
            // hfo = max(H, F, O): the two verticals folded in before the horizontal ladders.
            let hfo = S::max(
                striped_h[row_base + j],
                S::max(striped_f[row_base + j], striped_o[row_base + j]),
            );
            // E_row[j] = or(slli(H), srli(x)) + g + e (== left neighbor + g_); ladder from e.
            let e_open = S::add(
                S::add(S::or(S::slli_one_lane(hfo), S::srli_top_lane(x)), g_minus_e),
                e_vec,
            );
            let e_row = S::prefix_max(e_open, penalties_e, masks);
            // Q_row[j] = or(slli(H), srli(y)) + q + c (== left neighbor + q_); ladder from c.
            let q_open = S::add(
                S::add(S::or(S::slli_one_lane(hfo), S::srli_top_lane(y)), q_minus_c),
                c_vec,
            );
            let q_row = S::prefix_max(q_open, penalties_c, masks);
            striped_e[row_base + j] = e_row;
            striped_q[row_base + j] = q_row;
            let mut hv = S::max(hfo, S::max(e_row, q_row));
            x = S::max(hv, S::sub(e_row, g_minus_e));
            y = S::max(hv, S::sub(q_row, q_minus_c));
            if alignment_type == AlignmentType::Local {
                hv = S::max(hv, zeroes);
            }
            striped_h[row_base + j] = hv;
            score = S::max(score, hv);
        }

        // Per-type max-score tracking (impl:1711-1735). STRICT `<` keeps the earlier row on a tie.
        match alignment_type {
            AlignmentType::Local => {
                let row_score = S::horizontal_max(score).to_i32();
                if max_score < row_score {
                    max_score = row_score;
                    max_i = i;
                }
            }
            AlignmentType::Overlap => {
                if node.outedges.is_empty() {
                    let row_score = row_max::<S>(score);
                    if max_score < row_score {
                        max_score = row_score;
                        max_i = i;
                    }
                }
            }
            AlignmentType::Global => {
                if node.outedges.is_empty() {
                    let last = striped_h[row_base + (matrix_width_vecs - 1)];
                    let row_score = value_at::<S>(last, last_column_id);
                    if max_score < row_score {
                        max_score = row_score;
                        max_i = i;
                    }
                }
            }
        }
    }

    // Resolve max_j (impl:1738-1760). "Not found" -> (0, 0), the backtrack's empty-alignment guard.
    if max_i == 0 {
        return (0, 0, max_score);
    }
    let max_j = match alignment_type {
        AlignmentType::Global => seq_len,
        AlignmentType::Local | AlignmentType::Overlap => {
            let row = &striped_h[max_i * matrix_width_vecs..(max_i + 1) * matrix_width_vecs];
            let idx = index_of::<S>(row, matrix_width_vecs, max_score);
            if idx < 0 {
                return (0, 0, max_score);
            }
            idx as usize + 1
        }
    };

    (max_i, max_j, max_score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::simd::band::{self, BandConfig, BandState};
    use crate::align::simd::profile::{build_masks, build_penalties, build_profile};
    use crate::align::sisd::{reseed_scalar_buffers, ScalarInit};
    use crate::graph::{Graph, NodeId};

    // A real multi-lane backend (LANES == 8) so the striped carry seeding is genuinely exercised;
    // the degenerate `ScalarSimd` (LANES == 1) would validate none of the cross-lane shift logic.
    #[cfg(target_arch = "aarch64")]
    type TestSimd = crate::align::simd::neon::NeonI16;
    #[cfg(target_arch = "x86_64")]
    type TestSimd = crate::align::simd::sse41::Sse41I16;

    /// Reads DP cell `(row i, 0-based query column j)` out of a striped H buffer.
    fn cell<S>(striped: &[S::Vec], matrix_width_vecs: usize, i: usize, j: usize) -> i32
    where
        S: Simd,
        S::Elem: ElemFromI32 + ElemToI32,
    {
        let seg = j / S::LANES;
        let lane = j % S::LANES;
        let mut buf = vec![S::Elem::from_i32(0); S::LANES];
        S::storeu(striped[i * matrix_width_vecs + seg], &mut buf);
        buf[lane].to_i32()
    }

    /// Recomputes each graph row's banded window `(beg, end, beg_sn, end_sn)` from a *finalized*
    /// [`BandState`]. Because `best_col[rank]` is written exactly once (when its row is filled) and
    /// every predecessor has a strictly-lower rank, the post-run `best_col` equals the values the
    /// fill read while building each row — so this reproduces the identical band deterministically.
    /// Mirrors the window derivation inside [`fill_linear`] exactly.
    fn recompute_window(
        graph: &Graph,
        node_id_to_rank: &[u32],
        state: &BandState,
        node_id: crate::graph::NodeId,
        seq_len: usize,
        lanes: usize,
        matrix_width_vecs: usize,
    ) -> (usize, usize, usize, usize) {
        let node = &graph.nodes[node_id.0 as usize];
        let rank = node_id_to_rank[node_id.0 as usize] as usize;
        let anchor = band::anchor(state.r[rank], seq_len);
        let (mstart, mend) = if node.inedges.is_empty() {
            (anchor, anchor)
        } else {
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            for &edge_id in &node.inedges {
                let tail = graph.edges[edge_id.0 as usize].tail;
                let pr = node_id_to_rank[tail.0 as usize] as usize;
                let bc = state.best_col[pr] as usize;
                lo = lo.min(bc);
                hi = hi.max(bc);
            }
            (lo, hi)
        };
        let (beg, end) = band::node_window(anchor, mstart, mend, state.w, seq_len);
        let (beg_sn, end_sn) = band::segment_range(beg, end, lanes, matrix_width_vecs);
        (beg, end, beg_sn, end_sn)
    }

    /// Builds the scalar seed / striped profile / prefix-max ladders for a linear-gap fill, then runs
    /// [`fill_linear`] once. Factored so the exact and banded runs share identical setup.
    fn run_linear(
        graph: &Graph,
        seq: &[u8],
        scoring: Scoring,
        alignment_type: AlignmentType,
        striped_h: &mut Vec<<TestSimd as Simd>::Vec>,
        band: Option<&mut BandState>,
    ) -> (usize, usize, i32) {
        let mut seeded = ScalarInit::default();
        reseed_scalar_buffers(&mut seeded, alignment_type, scoring, seq, graph);
        let mut profile: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        build_profile::<TestSimd>(&mut profile, graph, seq, scoring);
        let masks = build_masks::<TestSimd>(<TestSimd as Simd>::NEG_INF);
        let penalties = build_penalties::<TestSimd>(
            <<TestSimd as Simd>::Elem as ElemFromI32>::from_i32(i32::from(scoring.g)),
        );
        fill_linear::<TestSimd>(
            graph,
            seq.len(),
            scoring,
            alignment_type,
            &seeded,
            &profile,
            &masks,
            &penalties,
            striped_h,
            band,
        )
    }

    /// PRIMARY per-cell banded gate (design §Correctness gate 1). A narrow band (`base = 2`) over an
    /// identical-read fixture long enough that the window's segment start drifts to `beg_sn >= 1` on
    /// deep rows — the ONLY configuration that exercises the corrected left-edge carry seeding.
    /// Asserts (a) at least one row actually reached `beg_sn >= 1` (the gate is not vacuous), (b) the
    /// banded `(max_i, max_j, max_score)` equals the exact one, and (c) every in-band cell of the
    /// banded `H` equals the exact `H` — proving the diagonal carry recovered the match transition at
    /// the band's left edge (a closed-to-`NEG_INF` diagonal carry would make these cells too low).
    #[test]
    fn banded_linear_in_band_cells_match_exact_at_beg_sn_ge_1() {
        let lanes = <TestSimd as Simd>::LANES;
        // A non-repetitive 48-mer; identical query so the optimum is the exact diagonal (in band).
        let seq = b"ACGTTGCAGATCCGTAAGCTTACGGATCAGTTCAGGATCACGTTGCAA";
        let seq_len = seq.len();
        let matrix_width_vecs = seq_len.div_ceil(lanes);
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let alignment_type = AlignmentType::Local;

        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], seq, 1).unwrap();

        // Exact (full-matrix) fill.
        let mut h_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let exact_max = run_linear(&graph, seq, scoring, alignment_type, &mut h_exact, None);

        // Banded fill with a deliberately narrow window.
        let node_id_to_rank = {
            let mut m = vec![0u32; graph.num_nodes()];
            for (rank, &node_id) in graph.rank_order().iter().enumerate() {
                m[node_id.0 as usize] = rank as u32;
            }
            m
        };
        let mut band = BandState::new(
            &graph,
            &node_id_to_rank,
            seq_len,
            BandConfig { base: 2, frac: 0.0 },
        );
        let mut h_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let band_max = run_linear(
            &graph,
            seq,
            scoring,
            alignment_type,
            &mut h_band,
            Some(&mut band),
        );

        // (b) The banded optimum equals the exact optimum (the path stayed in band).
        assert_eq!(
            band_max, exact_max,
            "banded (max_i, max_j, max_score) != exact"
        );

        // (a) + (c): recompute every row's window from the finalized band, assert in-band equality,
        // and confirm the risky `beg_sn >= 1` seeding actually ran on at least one row.
        let mut saw_beg_sn_ge_1 = false;
        for &node_id in &graph.rank_to_node {
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
            let (beg, end, beg_sn, end_sn) = recompute_window(
                &graph,
                &node_id_to_rank,
                &band,
                node_id,
                seq_len,
                lanes,
                matrix_width_vecs,
            );
            if beg_sn >= 1 {
                saw_beg_sn_ge_1 = true;
            }
            for j in beg..end {
                assert_eq!(
                    cell::<TestSimd>(&h_band, matrix_width_vecs, i, j),
                    cell::<TestSimd>(&h_exact, matrix_width_vecs, i, j),
                    "in-band cell mismatch at (row {i}, col {j}); window [{beg},{end}) segs [{beg_sn},{end_sn})",
                );
            }
        }

        assert!(
            saw_beg_sn_ge_1,
            "gate is vacuous: no row reached beg_sn >= 1 (band too wide for the fixture)",
        );
    }

    /// Regression gate for the ADDITIONAL-predecessor diagonal-seed loop (`for p in
    /// 1..node.inedges.len()` in [`fill_linear`]) — the primary gate above only exercises a linear
    /// chain, where no node has more than one in-edge, so that loop's `beg_sn > 0` carry seed
    /// (`srli_top_lane(striped_h[pred_base + beg_sn - 1])`) never runs.
    ///
    /// Builds a genuinely branching (reconvergent) POA graph: seed sequence `seq1`, then a second
    /// sequence `seq2` identical to `seq1` except for a single interior substitution at `MISMATCH`,
    /// aligned so every position maps onto `seq1`'s existing node at that index (`(i, i)` for every
    /// `i`). At `MISMATCH` the mapped node's code differs from `seq2`'s base there, so
    /// [`Graph::add_alignment`] takes its cross-link fork path automatically (`graph.cpp:207-231`),
    /// creating a brand-new node aligned to (but distinct from) `seq1`'s node at that column. The
    /// very next base then re-joins the ORIGINAL node one column later, giving that node two
    /// in-edges: `seq1`'s original edge (first predecessor) and the new fork's edge (the additional
    /// predecessor).
    ///
    /// Running the fill with query `seq2` makes the fork the DOMINANT (matching) predecessor at the
    /// reconvergent node's row — the original edge's tail now mismatches `seq2` at `MISMATCH` — so
    /// the additional-predecessor loop's diagonal transition determines the row's value there. The
    /// fork's own row is one rank shallower and, at this `MISMATCH`, its window's left edge lands
    /// in an EARLIER striped segment than the reconvergent row's (`fork_beg_sn < recon_beg_sn`,
    /// asserted below): that is exactly what makes the seed's source cell (`pred_base + beg_sn -
    /// 1`) a REAL, in-band value in the fork's buffer rather than a NEG_INF stub that would mask a
    /// broken seed — i.e. why this specific `MISMATCH` (not an arbitrary one) makes the mutation
    /// check below actually fail. `MISMATCH` was found by scanning candidate positions for this
    /// exact `fork_beg_sn < recon_beg_sn` (with `recon_beg_sn >= 1`) property under the fixed
    /// `BandConfig { base: 2, frac: 0.0 }`; nearby positions land both rows in the same segment and
    /// would NOT catch the mutation despite `recon_beg_sn >= 1` alone holding.
    #[test]
    fn banded_linear_multi_predecessor_diagonal_seed() {
        let lanes = <TestSimd as Simd>::LANES;
        // A non-repetitive 120-mer, long enough that the reconvergent node's own row (rank
        // MISMATCH + 1) drifts to `beg_sn >= 1`, one striped segment AHEAD of the fork's own row,
        // under the same narrow `base = 2` band as the primary gate (see the doc comment above).
        let seq1 = b"AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGGCAACGACATGTGCGGCGACCCTTGCGACAGTGACGCTTTCGCCGTTGCCTAAACCTATTTGAAGGAGTCTAGCAGCCG";
        const MISMATCH: usize = 42; // interior substitution column (0-based); see doc comment.
        let seq_len = seq1.len();
        assert_eq!(
            seq_len, 120,
            "fixture invariant: MISMATCH was tuned for this exact sequence"
        );
        let mut seq2 = seq1.to_vec();
        let repl = *b"ACGT"
            .iter()
            .find(|&&b| b != seq1[MISMATCH])
            .expect("ACGT has >1 distinct base");
        seq2[MISMATCH] = repl;
        let matrix_width_vecs = seq_len.div_ceil(lanes);
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let alignment_type = AlignmentType::Local;

        // seq1 into a fresh graph: nodes 0..seq_len-1, node id == position (linear chain).
        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], seq1, 1).unwrap();

        // seq2 aligned identity-wise against seq1's existing nodes; the MISMATCH column's code
        // mismatch triggers Graph::add_alignment's cross-link fork automatically.
        let alignment: Vec<(i32, i32)> = (0..seq_len).map(|pos| (pos as i32, pos as i32)).collect();
        graph.add_alignment_weight(&alignment, &seq2, 1).unwrap();

        let reconvergent = NodeId(MISMATCH as u32 + 1);
        assert!(
            graph.nodes[reconvergent.0 as usize].inedges.len() >= 2,
            "fixture invariant: node at MISMATCH + 1 must be reconvergent (>= 2 in-edges)"
        );
        let fork_node = *graph.nodes[MISMATCH]
            .aligned_nodes
            .first()
            .expect("fixture invariant: MISMATCH's node must have forked an aligned node");

        let node_id_to_rank = {
            let mut m = vec![0u32; graph.num_nodes()];
            for (rank, &node_id) in graph.rank_order().iter().enumerate() {
                m[node_id.0 as usize] = rank as u32;
            }
            m
        };

        // Exact (full-matrix) fill, query = seq2 (the branch that makes the fork dominant).
        let mut h_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let exact_max = run_linear(&graph, &seq2, scoring, alignment_type, &mut h_exact, None);

        // Banded fill, same deliberately narrow window as the primary gate.
        let mut band = BandState::new(
            &graph,
            &node_id_to_rank,
            seq_len,
            BandConfig { base: 2, frac: 0.0 },
        );
        let mut h_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let band_max = run_linear(
            &graph,
            &seq2,
            scoring,
            alignment_type,
            &mut h_band,
            Some(&mut band),
        );

        assert_eq!(
            band_max, exact_max,
            "banded (max_i, max_j, max_score) != exact"
        );

        // The load-bearing geometry (see doc comment): the reconvergent row's window starts one
        // striped segment AFTER the fork row's window, so the additional-pred loop's `beg_sn - 1`
        // lookback into the fork's buffer lands on a REAL (in-band) cell, not a NEG_INF stub.
        let (_, _, fork_beg_sn, _) = recompute_window(
            &graph,
            &node_id_to_rank,
            &band,
            fork_node,
            seq_len,
            lanes,
            matrix_width_vecs,
        );
        let (_, _, recon_beg_sn, _) = recompute_window(
            &graph,
            &node_id_to_rank,
            &band,
            reconvergent,
            seq_len,
            lanes,
            matrix_width_vecs,
        );
        assert!(
            recon_beg_sn >= 1,
            "gate does not exercise the additional-predecessor loop's beg_sn > 0 branch: the \
             reconvergent node's own row never reached beg_sn >= 1"
        );
        assert!(
            fork_beg_sn < recon_beg_sn,
            "gate is vacuous: the fork's own row window [.., beg_sn={fork_beg_sn}) does not start \
             strictly before the reconvergent row's (beg_sn={recon_beg_sn}), so the seed's source \
             cell would be a NEG_INF stub in the fork's buffer either way"
        );

        // Per-cell in-band equality over every row (same shape as the primary gate) — proves the
        // additional-predecessor loop's corrected diagonal seed recovered the match transition.
        for &node_id in &graph.rank_to_node {
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
            let (beg, end, beg_sn, end_sn) = recompute_window(
                &graph,
                &node_id_to_rank,
                &band,
                node_id,
                seq_len,
                lanes,
                matrix_width_vecs,
            );
            for j in beg..end {
                assert_eq!(
                    cell::<TestSimd>(&h_band, matrix_width_vecs, i, j),
                    cell::<TestSimd>(&h_exact, matrix_width_vecs, i, j),
                    "in-band cell mismatch at (row {i}, col {j}); window [{beg},{end}) segs [{beg_sn},{end_sn})",
                );
            }
        }
    }

    /// Builds the scalar seed / striped profile / prefix-max ladder for an affine-gap fill, then runs
    /// [`fill_affine`] once. Mirrors [`run_linear`]; affine's prefix-max ladder is built from the
    /// EXTEND penalty `e` (matching `mod.rs`'s `align_simd_affine`, not the linear ladder's `g`).
    #[allow(clippy::too_many_arguments)]
    fn run_affine(
        graph: &Graph,
        seq: &[u8],
        scoring: Scoring,
        alignment_type: AlignmentType,
        striped_h: &mut Vec<<TestSimd as Simd>::Vec>,
        striped_e: &mut Vec<<TestSimd as Simd>::Vec>,
        striped_f: &mut Vec<<TestSimd as Simd>::Vec>,
        band: Option<&mut BandState>,
    ) -> (usize, usize, i32) {
        let mut seeded = ScalarInit::default();
        reseed_scalar_buffers(&mut seeded, alignment_type, scoring, seq, graph);
        let mut profile: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        build_profile::<TestSimd>(&mut profile, graph, seq, scoring);
        let masks = build_masks::<TestSimd>(<TestSimd as Simd>::NEG_INF);
        let penalties = build_penalties::<TestSimd>(
            <<TestSimd as Simd>::Elem as ElemFromI32>::from_i32(i32::from(scoring.e)),
        );
        fill_affine::<TestSimd>(
            graph,
            seq.len(),
            scoring,
            alignment_type,
            &seeded,
            &profile,
            &masks,
            &penalties,
            striped_h,
            striped_e,
            striped_f,
            band,
        )
    }

    /// PRIMARY affine per-cell banded gate — the affine twin of
    /// [`banded_linear_in_band_cells_match_exact_at_beg_sn_ge_1`] (design §Correctness gate 1),
    /// extended to affine's vertical `F` matrix. A narrow band (`base = 2`) over an identical-read
    /// fixture long enough that the window drifts to `beg_sn >= 1` on deep rows — the only
    /// configuration that exercises the corrected left-edge carry seeding. Asserts (a) a row reached
    /// `beg_sn >= 1`, (b) the banded `(max_i, max_j, max_score)` equals the exact one, and (c) every
    /// in-band cell of BOTH the banded `H` AND the vertical `F` equals the exact fill.
    ///
    /// `F` is the pure vertical matrix (`F[i][j] = max(H_pred[j] + g, F_pred[j]) + e`), so its inputs
    /// are the predecessor's SAME column `j`. It therefore matches exact only where that predecessor
    /// column is itself in the predecessor's computed band; the assertion is scoped to `j` whose
    /// vertical predecessor cell is in-band (the design's "in-band `F`"), which for this diagonal
    /// identity band is all of `[beg, end)` except at most the extreme right-edge column a rank's band
    /// gains over its predecessor's. `E` (horizontal) is intentionally NOT asserted per-cell: closing
    /// its band-edge carry to `NEG_INF` legitimately drops `E` at the first in-band column below the
    /// exact value (there is no horizontal-gap path into the band), while `H = max(H, F, E)` still
    /// matches because the diagonal dominates — the property the gate actually protects.
    #[test]
    fn banded_affine_in_band_cells_match_exact_at_beg_sn_ge_1() {
        let lanes = <TestSimd as Simd>::LANES;
        // A non-repetitive 48-mer; identical query so the optimum is the exact diagonal (in band).
        let seq = b"ACGTTGCAGATCCGTAAGCTTACGGATCAGTTCAGGATCACGTTGCAA";
        let seq_len = seq.len();
        let matrix_width_vecs = seq_len.div_ceil(lanes);
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let alignment_type = AlignmentType::Local;

        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], seq, 1).unwrap();

        // Exact (full-matrix) fill.
        let mut h_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut e_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut f_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let exact_max = run_affine(
            &graph,
            seq,
            scoring,
            alignment_type,
            &mut h_exact,
            &mut e_exact,
            &mut f_exact,
            None,
        );

        // Banded fill with a deliberately narrow window.
        let node_id_to_rank = {
            let mut m = vec![0u32; graph.num_nodes()];
            for (rank, &node_id) in graph.rank_order().iter().enumerate() {
                m[node_id.0 as usize] = rank as u32;
            }
            m
        };
        let mut band = BandState::new(
            &graph,
            &node_id_to_rank,
            seq_len,
            BandConfig { base: 2, frac: 0.0 },
        );
        let mut h_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut e_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut f_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let band_max = run_affine(
            &graph,
            seq,
            scoring,
            alignment_type,
            &mut h_band,
            &mut e_band,
            &mut f_band,
            Some(&mut band),
        );

        // (b) The banded optimum equals the exact optimum (the path stayed in band).
        assert_eq!(
            band_max, exact_max,
            "banded (max_i, max_j, max_score) != exact"
        );

        // (a) + (c): recompute every row's window from the finalized band, assert in-band equality of
        // H (all in-band cells) and F (in-band cells whose vertical predecessor cell is also in-band),
        // and confirm the risky `beg_sn >= 1` seeding actually ran.
        let mut saw_beg_sn_ge_1 = false;
        for &node_id in &graph.rank_to_node {
            let node = &graph.nodes[node_id.0 as usize];
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
            let (beg, end, beg_sn, end_sn) = recompute_window(
                &graph,
                &node_id_to_rank,
                &band,
                node_id,
                seq_len,
                lanes,
                matrix_width_vecs,
            );
            if beg_sn >= 1 {
                saw_beg_sn_ge_1 = true;
            }
            // The first predecessor's computed segment range, used to scope the F (vertical) check to
            // columns whose predecessor cell is actually in-band (the recurrence's only F input).
            let pred_band = node.inedges.first().map(|&edge_id| {
                let tail = graph.edges[edge_id.0 as usize].tail;
                let pred_node = graph.rank_to_node[node_id_to_rank[tail.0 as usize] as usize];
                let (_, _, pbs, pes) = recompute_window(
                    &graph,
                    &node_id_to_rank,
                    &band,
                    pred_node,
                    seq_len,
                    lanes,
                    matrix_width_vecs,
                );
                (pbs * lanes, pes * lanes)
            });
            for j in beg..end {
                assert_eq!(
                    cell::<TestSimd>(&h_band, matrix_width_vecs, i, j),
                    cell::<TestSimd>(&h_exact, matrix_width_vecs, i, j),
                    "H in-band mismatch at (row {i}, col {j}); window [{beg},{end}) segs [{beg_sn},{end_sn})",
                );
                if let Some((pbeg, pend)) = pred_band {
                    if j >= pbeg && j < pend {
                        assert_eq!(
                            cell::<TestSimd>(&f_band, matrix_width_vecs, i, j),
                            cell::<TestSimd>(&f_exact, matrix_width_vecs, i, j),
                            "F in-band mismatch at (row {i}, col {j}); window [{beg},{end}) segs [{beg_sn},{end_sn}), pred segs cols [{pbeg},{pend})",
                        );
                    }
                }
            }
        }

        assert!(
            saw_beg_sn_ge_1,
            "gate is vacuous: no row reached beg_sn >= 1 (band too wide for the fixture)",
        );
    }

    /// Affine twin of [`banded_linear_multi_predecessor_diagonal_seed`]: the regression gate for the
    /// ADDITIONAL-predecessor diagonal-seed loop in [`fill_affine`]. Builds the identical reconvergent
    /// (branching) POA fixture — `seq1` seeded, then a near-identical `seq2` with one interior
    /// substitution at `MISMATCH`, aligned identity-wise so the node at `MISMATCH + 1` gains a second
    /// in-edge (see the linear test's doc comment for the full geometry rationale). Running with query
    /// `seq2` makes the fork the DOMINANT predecessor at the reconvergent row, so the
    /// additional-predecessor loop's diagonal seed determines that row's value.
    ///
    /// Asserts `recon_beg_sn >= 1` and `fork_beg_sn < recon_beg_sn` (so the seed's source cell
    /// `pred_base + beg_sn - 1` is a REAL in-band value in the fork's buffer, not a `NEG_INF` stub
    /// that would mask a broken seed), then per-cell in-band equality of `H` and `F` over every row.
    #[test]
    fn banded_affine_multi_predecessor_diagonal_seed() {
        let lanes = <TestSimd as Simd>::LANES;
        let seq1 = b"AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGGCAACGACATGTGCGGCGACCCTTGCGACAGTGACGCTTTCGCCGTTGCCTAAACCTATTTGAAGGAGTCTAGCAGCCG";
        const MISMATCH: usize = 42; // interior substitution column (0-based); see linear test's doc.
        let seq_len = seq1.len();
        assert_eq!(
            seq_len, 120,
            "fixture invariant: MISMATCH was tuned for this exact sequence"
        );
        let mut seq2 = seq1.to_vec();
        let repl = *b"ACGT"
            .iter()
            .find(|&&b| b != seq1[MISMATCH])
            .expect("ACGT has >1 distinct base");
        seq2[MISMATCH] = repl;
        let matrix_width_vecs = seq_len.div_ceil(lanes);
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let alignment_type = AlignmentType::Local;

        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], seq1, 1).unwrap();
        let alignment: Vec<(i32, i32)> = (0..seq_len).map(|pos| (pos as i32, pos as i32)).collect();
        graph.add_alignment_weight(&alignment, &seq2, 1).unwrap();

        let reconvergent = NodeId(MISMATCH as u32 + 1);
        assert!(
            graph.nodes[reconvergent.0 as usize].inedges.len() >= 2,
            "fixture invariant: node at MISMATCH + 1 must be reconvergent (>= 2 in-edges)"
        );
        let fork_node = *graph.nodes[MISMATCH]
            .aligned_nodes
            .first()
            .expect("fixture invariant: MISMATCH's node must have forked an aligned node");

        let node_id_to_rank = {
            let mut m = vec![0u32; graph.num_nodes()];
            for (rank, &node_id) in graph.rank_order().iter().enumerate() {
                m[node_id.0 as usize] = rank as u32;
            }
            m
        };

        // Exact (full-matrix) fill, query = seq2 (the branch that makes the fork dominant).
        let mut h_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut e_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut f_exact: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let exact_max = run_affine(
            &graph,
            &seq2,
            scoring,
            alignment_type,
            &mut h_exact,
            &mut e_exact,
            &mut f_exact,
            None,
        );

        // Banded fill, same deliberately narrow window as the primary gate.
        let mut band = BandState::new(
            &graph,
            &node_id_to_rank,
            seq_len,
            BandConfig { base: 2, frac: 0.0 },
        );
        let mut h_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut e_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let mut f_band: Vec<<TestSimd as Simd>::Vec> = Vec::new();
        let band_max = run_affine(
            &graph,
            &seq2,
            scoring,
            alignment_type,
            &mut h_band,
            &mut e_band,
            &mut f_band,
            Some(&mut band),
        );

        assert_eq!(
            band_max, exact_max,
            "banded (max_i, max_j, max_score) != exact"
        );

        // The load-bearing geometry (see linear test's doc): the reconvergent row's window starts one
        // striped segment AFTER the fork row's window, so the additional-pred loop's `beg_sn - 1`
        // lookback into the fork's buffer lands on a REAL (in-band) cell, not a NEG_INF stub.
        let (_, _, fork_beg_sn, _) = recompute_window(
            &graph,
            &node_id_to_rank,
            &band,
            fork_node,
            seq_len,
            lanes,
            matrix_width_vecs,
        );
        let (_, _, recon_beg_sn, _) = recompute_window(
            &graph,
            &node_id_to_rank,
            &band,
            reconvergent,
            seq_len,
            lanes,
            matrix_width_vecs,
        );
        assert!(
            recon_beg_sn >= 1,
            "gate does not exercise the additional-predecessor loop's beg_sn > 0 branch: the \
             reconvergent node's own row never reached beg_sn >= 1"
        );
        assert!(
            fork_beg_sn < recon_beg_sn,
            "gate is vacuous: the fork's own row window [.., beg_sn={fork_beg_sn}) does not start \
             strictly before the reconvergent row's (beg_sn={recon_beg_sn}), so the seed's source \
             cell would be a NEG_INF stub in the fork's buffer either way"
        );

        // Per-cell in-band equality of H (all in-band cells) and F (cells whose vertical predecessor
        // cell is also in-band) over every row — proves the additional-predecessor loop's corrected
        // diagonal seed recovered the match transition.
        for &node_id in &graph.rank_to_node {
            let node = &graph.nodes[node_id.0 as usize];
            let i = node_id_to_rank[node_id.0 as usize] as usize + 1;
            let (beg, end, beg_sn, end_sn) = recompute_window(
                &graph,
                &node_id_to_rank,
                &band,
                node_id,
                seq_len,
                lanes,
                matrix_width_vecs,
            );
            let pred_band = node.inedges.first().map(|&edge_id| {
                let tail = graph.edges[edge_id.0 as usize].tail;
                let pred_node = graph.rank_to_node[node_id_to_rank[tail.0 as usize] as usize];
                let (_, _, pbs, pes) = recompute_window(
                    &graph,
                    &node_id_to_rank,
                    &band,
                    pred_node,
                    seq_len,
                    lanes,
                    matrix_width_vecs,
                );
                (pbs * lanes, pes * lanes)
            });
            for j in beg..end {
                assert_eq!(
                    cell::<TestSimd>(&h_band, matrix_width_vecs, i, j),
                    cell::<TestSimd>(&h_exact, matrix_width_vecs, i, j),
                    "H in-band mismatch at (row {i}, col {j}); window [{beg},{end}) segs [{beg_sn},{end_sn})",
                );
                if let Some((pbeg, pend)) = pred_band {
                    if j >= pbeg && j < pend {
                        assert_eq!(
                            cell::<TestSimd>(&f_band, matrix_width_vecs, i, j),
                            cell::<TestSimd>(&f_exact, matrix_width_vecs, i, j),
                            "F in-band mismatch at (row {i}, col {j}); window [{beg},{end}) segs [{beg_sn},{end_sn}), pred segs cols [{pbeg},{pend})",
                        );
                    }
                }
            }
        }
    }
}
