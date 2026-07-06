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
use crate::graph::{EdgeId, Graph};

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
/// `band` is plumbing for the (not-yet-active) banded fill: `None` reproduces today's full-matrix
/// behavior exactly; it is otherwise unused for now.
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

        // First predecessor: diagonal (with carry) + vertical (impl:773-795).
        let mut pred_i = if node.inedges.is_empty() {
            0
        } else {
            pred_row(node.inedges[0])
        };
        let pred_base = pred_i * matrix_width_vecs;
        let mut x = S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))));
        for j in 0..matrix_width_vecs {
            let h_pred = striped_h[pred_base + j];
            let t1 = S::srli_top_lane(h_pred);
            let diag = S::or(S::slli_one_lane(h_pred), x);
            x = t1;
            let value = S::max(
                S::add(diag, profile[profile_base + j]),
                S::add(h_pred, g_vec),
            );
            striped_h[row_base + j] = value;
        }

        // Additional predecessors: fold in with max (impl:796-821).
        for p in 1..node.inedges.len() {
            pred_i = pred_row(node.inedges[p]);
            let pred_base = pred_i * matrix_width_vecs;
            let mut x = S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))));
            for j in 0..matrix_width_vecs {
                let h_pred = striped_h[pred_base + j];
                let t1 = S::srli_top_lane(h_pred);
                let m = S::or(S::slli_one_lane(h_pred), x);
                x = t1;
                let cur = striped_h[row_base + j];
                let candidate = S::max(S::add(m, profile[profile_base + j]), S::add(h_pred, g_vec));
                striped_h[row_base + j] = S::max(cur, candidate);
            }
        }

        // Horizontal gap (prefix_max) + inter-segment carry, then the per-row score (impl:823-846).
        let mut score = S::splat(S::NEG_INF);
        let mut x = S::srli_top_lane(S::add(S::splat(S::Elem::from_i32(first_column(i))), g_vec));
        for j in 0..matrix_width_vecs {
            let mut hv = striped_h[row_base + j];
            hv = S::max(hv, S::or(x, carry_mask));
            // NOTE: trait arg order is (penalties, masks) — transposed vs upstream's (masks,
            // penalties) — see the `Simd::prefix_max` doc and the Task 6 review.
            hv = S::prefix_max(hv, penalties, masks);
            x = S::srli_top_lane(S::add(hv, g_vec));
            if alignment_type == AlignmentType::Local {
                hv = S::max(hv, zeroes);
            }
            striped_h[row_base + j] = hv;
            score = S::max(score, hv);
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
/// `band` is plumbing for the (not-yet-active) banded fill: `None` reproduces today's full-matrix
/// behavior exactly; it is otherwise unused for now.
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

        // First predecessor: F (vertical) + H (diagonal) in one pass (impl:1127-1152).
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
            // F_row[j] = max(H_pred + g, F_pred) + e.
            striped_f[row_base + j] = S::add(S::max(S::add(h_pred, g_minus_e), f_pred), e_vec);
            // H_row[j] = diag(H_pred) + profile (diagonal only; vertical lives in F).
            let diag = S::or(S::slli_one_lane(h_pred), x);
            x = S::srli_top_lane(h_pred);
            striped_h[row_base + j] = S::add(diag, profile[profile_base + j]);
        }

        // Additional predecessors: fold F and H with max (impl:1154-1184).
        for p in 1..node.inedges.len() {
            pred_i = pred_row(node.inedges[p]);
            pred_base = pred_i * matrix_width_vecs;
            let mut x = S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))));
            for j in 0..matrix_width_vecs {
                let h_pred = striped_h[pred_base + j];
                let f_pred = striped_f[pred_base + j];
                let cur_f = striped_f[row_base + j];
                let cand_f = S::add(S::max(S::add(h_pred, g_minus_e), f_pred), e_vec);
                striped_f[row_base + j] = S::max(cur_f, cand_f);
                let diag = S::or(S::slli_one_lane(h_pred), x);
                x = S::srli_top_lane(h_pred);
                let cur_h = striped_h[row_base + j];
                let cand_h = S::add(diag, profile[profile_base + j]);
                striped_h[row_base + j] = S::max(cur_h, cand_h);
            }
        }

        // H = max(H, F); E (horizontal via prefix_max) with the inter-segment x carry; H = max(H,
        // E); per-row score (impl:1186-1212). `x` starts as the full column-0 seed (NOT srli'd).
        let mut score = S::splat(S::NEG_INF);
        let mut x = S::splat(S::Elem::from_i32(first_column(i)));
        for j in 0..matrix_width_vecs {
            let hf = S::max(striped_h[row_base + j], striped_f[row_base + j]);
            // E_row[j] = or(slli(H), srli(x)) + g + e  (== left neighbor + g_).
            let e_open = S::add(
                S::add(S::or(S::slli_one_lane(hf), S::srli_top_lane(x)), g_minus_e),
                e_vec,
            );
            // NOTE: trait arg order is (penalties, masks) — transposed vs upstream; penalties are
            // built from the EXTEND penalty e_ for affine E (impl:1111), not g_.
            let e_row = S::prefix_max(e_open, penalties, masks);
            striped_e[row_base + j] = e_row;
            let mut hv = S::max(hf, e_row);
            x = S::max(hv, S::sub(e_row, g_minus_e));
            if alignment_type == AlignmentType::Local {
                hv = S::max(hv, zeroes);
            }
            striped_h[row_base + j] = hv;
            score = S::max(score, hv);
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
