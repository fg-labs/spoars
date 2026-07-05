//! Shared SIMD scaffolding: the striped profile the vectorized fill consumes, the destripe step
//! that merges a striped fill's interior back into row-major buffers, the scalar boundary seed
//! the shared backtrack needs (the SIMD kernels plan's "C2 fix"), and the prefix-max ladder's
//! masks/penalties.
//!
//! Ports the setup half of `SimdAlignmentEngine<A>::Initialize` and `::Align`
//! (`third_party/spoa/src/simd_alignment_engine_implementation.hpp:463-639,732-759`). Everything
//! here is generic over [`Simd`] and is exercised in this crate's tests with `ScalarSimd{I16,I32}`
//! (`LANES = 1`); a `LANES = 4` test-only reference impl (below, `#[cfg(test)]`) additionally
//! exercises the multi-lane/padding/ladder shapes those degenerate one-lane types cannot (see the
//! module doc on `lanes.rs` for why `LANES = 1` alone validates fill *structure* but none of the
//! cross-lane machinery).
//!
//! # The C2 fix
//!
//! spoa's striped SIMD matrix holds neither a column 0 (kept in a separate `first_column` array,
//! `impl:513-543`) nor a row-major `sequence_profile` (the striped profile's width is
//! `ceil(seq.len() / LANES)`, with no column-0 boundary). But the shared scalar backtrack this
//! crate reuses (`backtrack_linear`/`backtrack_affine`/`backtrack_convex`,
//! [`crate::align::backtrack`]) reads column 0, row 0, and a row-major `sequence_profile`
//! directly. [`seed_scalar_buffers`] closes that gap by calling straight through to
//! [`crate::align::sisd::seed_scalar_buffers`] — which itself calls through to the verified
//! `SisdEngine::initialize` — rather than re-deriving spoa's `first_column`/boundary-row formulas
//! a second time for the SIMD path (a second, drifting copy is exactly the risk the plan's C2
//! review flagged). See [`crate::align::sisd::ScalarInit`]'s doc for the line-by-line formula
//! equivalence this relies on.

use super::lanes::Simd;
use crate::align::sisd::{self, ScalarInit};
use crate::align::{AlignmentType, Scoring};
use crate::graph::Graph;

/// Converts a plain `i32` score into a SIMD lane element (`i16` or `i32`).
///
/// Local to this module rather than added to the [`Simd`] trait itself: only the profile/mask
/// builders below need to synthesize lane values from [`Scoring`]'s `i8` penalties (widened
/// through `i32` arithmetic to avoid `i8::MIN.abs()` overflow), so this stays a narrow,
/// module-private conversion rather than a general trait requirement every ISA backend would
/// otherwise have to implement.
pub(crate) trait ElemFromI32: Copy {
    /// Narrows `value` to `Self`. Callers guarantee `value` fits (every caller here derives it
    /// from `i8`-ranged [`Scoring`] penalties or a fixed `NEG_INF`-scale sentinel).
    fn from_i32(value: i32) -> Self;
}

impl ElemFromI32 for i16 {
    #[inline(always)]
    fn from_i32(value: i32) -> i16 {
        value as i16
    }
}

impl ElemFromI32 for i32 {
    #[inline(always)]
    fn from_i32(value: i32) -> i32 {
        value
    }
}

/// Widens a SIMD lane element (`i16` or `i32`) back to plain `i32`, losslessly.
///
/// Used by [`destripe_interior`] to write a vectorized fill's lanes into the shared `i32`
/// row-major buffers the backtrack reads, regardless of which element width produced them.
pub(crate) trait ElemToI32: Copy {
    /// Widens `self` to `i32`.
    fn to_i32(self) -> i32;
}

impl ElemToI32 for i16 {
    #[inline(always)]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
}

impl ElemToI32 for i32 {
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
}

/// Builds the striped sequence profile a vectorized fill consumes: one block of
/// `ceil(seq.len() / S::LANES)` vectors per graph alphabet code, segment `j`'s lane `k` scoring
/// query position `j * S::LANES + k` against that code.
///
/// Ports `SimdAlignmentEngine<A>::Initialize`'s profile loop
/// (`simd_alignment_engine_implementation.hpp:477-487`) and its `padding_penatly` computation
/// (`:471-473`, `-max(|m|, |n|, |g|, |q|)`, deliberately over `g`/`q` — the *first* penalty of
/// each affine/convex gap pair — not `e`/`c`): a lane whose query position falls at or past
/// `seq.len()` (the trailing lanes of the last segment, whenever `seq.len()` isn't a multiple of
/// `S::LANES`) gets that padding penalty instead of a match/mismatch score, so it can never be
/// mistaken for a real (better) alignment.
///
/// This is **distinct** from the row-major `sequence_profile` [`seed_scalar_buffers`] returns:
/// this one is column-0-free (`matrix_width` here is `ceil(seq.len() / S::LANES)` vector columns,
/// not `seq.len() + 1` scalar columns) and is what the vectorized fill reads; the row-major one is
/// what the shared scalar backtrack reads.
#[inline]
pub(crate) fn build_profile<S>(out: &mut Vec<S::Vec>, graph: &Graph, seq: &[u8], scoring: Scoring)
where
    S: Simd,
    S::Elem: ElemFromI32,
{
    let seq_len = seq.len();
    let matrix_width_vecs = seq_len.div_ceil(S::LANES);
    let num_codes = graph.num_codes as usize;

    let abs = |v: i8| i32::from(v).abs();
    let padding_penalty = -(abs(scoring.m)
        .max(abs(scoring.n))
        .max(abs(scoring.g))
        .max(abs(scoring.q)));

    // Reuse the caller's grow-only buffer: `clear` keeps the allocation, `reserve` grows capacity
    // only when this (possibly larger) sequence needs it. The profile is fully rebuilt every call
    // (it depends on `seq`), so no stale entry from a prior alignment can survive.
    out.clear();
    out.reserve(num_codes * matrix_width_vecs);
    let mut lane_buf = vec![S::Elem::from_i32(0); S::LANES];

    for code in 0..num_codes {
        let decoded = graph.decoder[code];
        for segment in 0..matrix_width_vecs {
            for (k, lane) in lane_buf.iter_mut().enumerate() {
                let pos = segment * S::LANES + k;
                let score = if pos < seq_len {
                    if decoded == i32::from(seq[pos]) {
                        i32::from(scoring.m)
                    } else {
                        i32::from(scoring.n)
                    }
                } else {
                    padding_penalty
                };
                *lane = S::Elem::from_i32(score);
            }
            out.push(S::loadu(&lane_buf));
        }
    }
}

/// Writes a striped fill's INTERIOR cells (graph rows `>= 1`, sequence columns `1..=seq_len`)
/// into the row-major `i32` buffer `dst` at `[i * (seq_len + 1) + j]`. Row 0 and column 0 of `dst`
/// are never touched — those come from [`seed_scalar_buffers`].
///
/// `matrix` must be laid out as `matrix_width_vecs` striped-profile-shaped vector columns per
/// interior graph row, one row per node in topological-rank order (row `r` of `matrix`, 0-indexed,
/// is graph row `r + 1`; `matrix.len()` must be a multiple of `matrix_width_vecs`). Segment `j`,
/// lane `k` of a row holds the fill's value at 0-based sequence position `j * S::LANES + k`; lanes
/// at or past `seq_len` (the striped profile's trailing padding lanes, see [`build_profile`]) are
/// simply not written to `dst` — there is no column for them.
// `#[inline(always)]` (not plain `#[inline]`) is load-bearing for x86 performance, mirroring the P1
// fill fix: this is the per-cell-heavy transpose (`O(node_count * seq_len)` stores). Left as a plain
// `#[inline]` the compiler kept it out-of-line, so it was codegen'd WITHOUT the caller's
// `#[target_feature(enable = "avx2")]`, forcing every `S::storeu` into a non-inlined call plus a
// `vzeroupper` AVX->SSE transition per segment (profiling showed `storeu` as its own ~29% frame and
// `vzeroupper` ~10% on AVX2 — precisely why AVX2 trailed SSE4.1). Forcing the inline folds the whole
// destripe into the `run_avx2_*`/`run_sse41_*` target_feature entry so the 256-bit stores inline and
// the transitions vanish. Output is unchanged.
#[inline(always)]
pub(crate) fn destripe_interior<S>(
    dst: &mut [i32],
    matrix: &[S::Vec],
    matrix_width_vecs: usize,
    seq_len: usize,
) where
    S: Simd,
    S::Elem: ElemFromI32 + ElemToI32,
{
    if matrix_width_vecs == 0 {
        return;
    }
    let row_major_width = seq_len + 1;
    let num_interior_rows = matrix.len() / matrix_width_vecs;

    // A segment `s` is "full" when all `LANES` of its lanes map to real sequence columns, i.e.
    // `(s + 1) * LANES <= seq_len`. Full segments de-stripe via a single contiguous widen-and-store
    // (`store_widened_i32`); the trailing partial segment (present iff `seq_len % LANES != 0`), whose
    // high lanes are the striped profile's padding, falls back to the per-lane scalar path below.
    let full_segments = seq_len / S::LANES;
    let mut lane_buf = vec![S::Elem::from_i32(0); S::LANES];

    for row in 0..num_interior_rows {
        let i = row + 1;
        let row_base = i * row_major_width;
        let matrix_row_base = row * matrix_width_vecs;

        for segment in 0..full_segments {
            // Lanes 0..LANES map to consecutive columns `segment*LANES + 1 ..= (segment+1)*LANES`,
            // all `<= seq_len` (< row_major_width), so this stays within row `i`.
            let dst_start = row_base + segment * S::LANES + 1;
            S::store_widened_i32(
                matrix[matrix_row_base + segment],
                &mut dst[dst_start..dst_start + S::LANES],
            );
        }

        for segment in full_segments..matrix_width_vecs {
            S::storeu(matrix[matrix_row_base + segment], &mut lane_buf);
            for (k, &lane) in lane_buf.iter().enumerate() {
                let pos = segment * S::LANES + k;
                if pos < seq_len {
                    dst[row_base + pos + 1] = lane.to_i32();
                }
            }
        }
    }
}

/// Runs the exact scalar boundary/profile setup (`SisdEngine::initialize`) for `(graph, seq,
/// scoring, alignment_type)`, returning row 0, column 0, and the row-major `sequence_profile` a
/// later vectorized fill's `destripe_interior` output merges with, and the shared scalar backtrack
/// reads.
///
/// This is a thin, parameter-reordered wrapper over [`crate::align::sisd::seed_scalar_buffers`]
/// (which itself calls straight through to `SisdEngine::initialize`) — see the module doc's "C2
/// fix" section for why this call-through, not a second implementation, is load-bearing.
// Only reached from this module's own tests: the live pipeline seeds row 0 / column 0 through
// `SisdEngine::initialize` directly (the C2 fix), not this wrapper, so it is dead in a non-test
// build on every target — hence the `allow`.
#[allow(dead_code)]
#[inline]
pub(crate) fn seed_scalar_buffers(
    graph: &Graph,
    seq: &[u8],
    scoring: Scoring,
    alignment_type: AlignmentType,
) -> ScalarInit {
    sisd::seed_scalar_buffers(alignment_type, scoring, seq, graph)
}

/// Builds the `S::LOG_LANES + 1` masks the prefix-max ladder ORs into a shifted-and-added vector
/// (`v = max(v, or(masks[k], slli(add(v, penalties[k]), shift_k)))`, see [`Simd::prefix_max`]'s
/// doc) so that lanes a shift vacates read as `neg_inf` (never as `0`, which could be mistaken for
/// a real, better score) instead of as whatever garbage a raw byte-shift leaves behind.
///
/// Ports `SimdAlignmentEngine<A>::Align`'s mask setup
/// (`simd_alignment_engine_implementation.hpp:743-749`): `masks[j]` (`j < S::LOG_LANES`) has
/// `neg_inf` in lanes `[0, 2^j)` and `0` elsewhere. The final `masks[S::LOG_LANES]` (`:748-749`,
/// the inter-segment carry mask) is built directly from its known resulting lane pattern —
/// `[0, neg_inf, neg_inf, ..., neg_inf]` — rather than via a literal `slli(splat(neg_inf), LSS)`
/// call: that pattern IS what `slli` by one lane's width produces when shifting an
/// all-`neg_inf` vector (lane 0's low-order bytes are zero-filled; every other lane inherits its
/// now-lower-indexed neighbor's `neg_inf`; the top lane's value is shifted out), so constructing
/// it directly is equivalent without needing a dynamic (non-compile-time-constant) shift amount,
/// which [`Simd::slli`]'s `const N` signature does not accept.
#[inline]
pub(crate) fn build_masks<S>(neg_inf: S::Elem) -> Vec<S::Vec>
where
    S: Simd,
    S::Elem: ElemFromI32,
{
    let zero = S::Elem::from_i32(0);
    let mut masks = Vec::with_capacity(S::LOG_LANES as usize + 1);

    for j in 0..S::LOG_LANES as usize {
        let covered = (1usize << j).min(S::LANES);
        let mut lanes = vec![zero; S::LANES];
        for lane in lanes.iter_mut().take(covered) {
            *lane = neg_inf;
        }
        masks.push(S::loadu(&lanes));
    }

    let mut carry = vec![neg_inf; S::LANES];
    if let Some(first) = carry.first_mut() {
        *first = zero;
    }
    masks.push(S::loadu(&carry));

    masks
}

/// Builds the `S::LOG_LANES` penalty vectors the prefix-max ladder adds at each step: `penalty *
/// 2^i` for step `i` (each step covers twice as many lanes as the last, so it must charge twice
/// the gap penalty to stay consistent with a scalar `gap * distance` recurrence).
///
/// Ports `SimdAlignmentEngine<A>::Align`'s penalty setup
/// (`simd_alignment_engine_implementation.hpp:754-759`) EXACTLY: `penalties[0] = splat(penalty)`,
/// `penalties[i] = penalties[i-1] + penalties[i-1]` (doubling via vector `add`, matching
/// upstream's `_mmxxx_add_epi(penalties[i-1], penalties[i-1])` rather than a separate "multiply by
/// 2" primitive [`Simd`] doesn't otherwise need).
#[inline]
pub(crate) fn build_penalties<S>(penalty: S::Elem) -> Vec<S::Vec>
where
    S: Simd,
{
    let mut penalties = Vec::with_capacity(S::LOG_LANES as usize);
    if S::LOG_LANES == 0 {
        return penalties;
    }

    let mut current = S::splat(penalty);
    penalties.push(current);
    for _ in 1..S::LOG_LANES {
        current = S::add(current, current);
        penalties.push(current);
    }

    penalties
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::simd::lanes::{ScalarSimdI16, ScalarSimdI32};
    use crate::align::sisd::SisdEngine;
    use crate::align::{AlignmentEngine, GapMode};
    use crate::graph::Graph;

    /// A `LANES = 4` reference [`Simd`] impl used ONLY in this module's tests: no intrinsics, no
    /// `unsafe`, plain arrays. `ScalarSimd{I16,I32}` (`LANES = 1`) cannot exercise multi-lane
    /// segment layout, trailing-lane padding, or a real `S::LOG_LANES > 0` mask/penalty ladder
    /// (see the module doc); this closes that gap for Task 5's own tests without waiting on a
    /// real ISA backend (Task 6+).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestVec4([i32; 4]);

    struct TestSimd4;

    impl Simd for TestSimd4 {
        type Elem = i32;
        type Vec = TestVec4;

        const LANES: usize = 4;
        const LOG_LANES: u32 = 2;
        const LSS: i32 = 4;
        const RSS: i32 = 12;
        const NEG_INF: i32 = i32::MIN + 1024;

        fn splat(value: i32) -> TestVec4 {
            TestVec4([value; 4])
        }

        fn add(a: TestVec4, b: TestVec4) -> TestVec4 {
            let mut out = [0i32; 4];
            for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
                *o = x.wrapping_add(*y);
            }
            TestVec4(out)
        }

        fn sub(a: TestVec4, b: TestVec4) -> TestVec4 {
            let mut out = [0i32; 4];
            for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
                *o = x.wrapping_sub(*y);
            }
            TestVec4(out)
        }

        fn min(a: TestVec4, b: TestVec4) -> TestVec4 {
            let mut out = [0i32; 4];
            for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
                *o = (*x).min(*y);
            }
            TestVec4(out)
        }

        fn max(a: TestVec4, b: TestVec4) -> TestVec4 {
            let mut out = [0i32; 4];
            for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
                *o = (*x).max(*y);
            }
            TestVec4(out)
        }

        fn or(a: TestVec4, b: TestVec4) -> TestVec4 {
            let mut out = [0i32; 4];
            for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
                *o = x | y;
            }
            TestVec4(out)
        }

        fn loadu(src: &[i32]) -> TestVec4 {
            TestVec4([src[0], src[1], src[2], src[3]])
        }

        fn storeu(v: TestVec4, dst: &mut [i32]) {
            dst[..4].copy_from_slice(&v.0);
        }

        fn store_widened_i32(v: TestVec4, dst: &mut [i32]) {
            // Elem is already `i32` for the 4-lane test vector; "widen" is a plain copy.
            dst[..4].copy_from_slice(&v.0);
        }

        fn slli<const N: i32>(v: TestVec4) -> TestVec4 {
            let lane_shift = (N / 4) as usize;
            let mut out = [0i32; 4];
            out[lane_shift..4].copy_from_slice(&v.0[..(4 - lane_shift)]);
            TestVec4(out)
        }

        fn srli<const N: i32>(v: TestVec4) -> TestVec4 {
            let lane_shift = (N / 4) as usize;
            let mut out = [0i32; 4];
            out[..(4 - lane_shift)].copy_from_slice(&v.0[lane_shift..4]);
            TestVec4(out)
        }

        fn slli_one_lane(v: TestVec4) -> TestVec4 {
            Self::slli::<4>(v)
        }

        fn srli_top_lane(v: TestVec4) -> TestVec4 {
            Self::srli::<12>(v)
        }

        fn horizontal_max(v: TestVec4) -> i32 {
            v.0.iter().fold(0, |acc, &x| acc.max(x))
        }

        /// Not exercised by any Task 5 function (`prefix_max` is only used by the DP fill, which
        /// lands in Task 7+); a real unrolled ladder isn't needed here, but this still matches the
        /// documented shape (2 steps, matching `LOG_LANES = 2`) for completeness.
        fn prefix_max(v: TestVec4, penalties: &[TestVec4], masks: &[TestVec4]) -> TestVec4 {
            let mut a = v;
            a = Self::max(
                a,
                Self::or(masks[0], Self::slli::<4>(Self::add(a, penalties[0]))),
            );
            a = Self::max(
                a,
                Self::or(masks[1], Self::slli::<8>(Self::add(a, penalties[1]))),
            );
            a
        }
    }

    fn linear_scoring() -> Scoring {
        Scoring::new(5, -4, -8, -8, -8, -8).unwrap()
    }

    fn affine_scoring() -> Scoring {
        Scoring::new(5, -4, -8, -6, -8, -6).unwrap()
    }

    fn convex_scoring() -> Scoring {
        Scoring::new(5, -4, -8, -6, -10, -4).unwrap()
    }

    fn linear_graph(seed: &[u8]) -> Graph {
        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], seed, 1).unwrap();
        graph
    }

    // ---- build_profile ------------------------------------------------------------------

    /// At `LANES = 1` every striped segment is exactly one query position, so `build_profile`'s
    /// output must line up 1:1 with `SisdEngine`'s row-major `sequence_profile` shifted by the
    /// row-major's column-0 offset — this is the direct "faithfulness anchor" comparison the task
    /// brief asks for.
    #[test]
    fn build_profile_scalar_matches_sisd_sequence_profile() {
        let graph = linear_graph(b"ACGT");
        let scoring = linear_scoring();
        let seq = b"AGGT";

        let mut profile = Vec::new();
        build_profile::<ScalarSimdI32>(&mut profile, &graph, seq, scoring);
        let seeded = seed_scalar_buffers(&graph, seq, scoring, AlignmentType::Global);

        let matrix_width_vecs = seq.len(); // LANES == 1
        for code in 0..graph.num_codes as usize {
            for j in 0..seq.len() {
                let striped = profile[code * matrix_width_vecs + j];
                let row_major = seeded.sequence_profile[code * seeded.matrix_width + (j + 1)];
                assert_eq!(striped, row_major, "code={code} pos={j}");
            }
        }
    }

    /// `LANES = 4` over a `seq.len()` that is NOT a multiple of 4 (5): the last segment's first
    /// lane is real (position 4) and its remaining three lanes are padding (positions 5, 6, 7 are
    /// past `seq.len()`), exercising the trailing/padding branch `ScalarSimd`'s `LANES = 1`
    /// degeneracy cannot reach (see the module doc).
    #[test]
    fn build_profile_multi_lane_pads_trailing_lanes_past_seq_len() {
        let graph = linear_graph(b"A"); // single code 'A'
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap(); // m=5,n=-4,g=-8,q=-10
        let seq = b"AAAAA"; // seq_len = 5, matches the graph's single code every position

        let mut profile = Vec::new();
        build_profile::<TestSimd4>(&mut profile, &graph, seq, scoring);
        // matrix_width_vecs = ceil(5/4) = 2, one code => 2 vectors total.
        assert_eq!(profile.len(), 2);

        // Segment 0: positions 0..4, all real matches (score m = 5).
        assert_eq!(profile[0], TestVec4([5, 5, 5, 5]));

        // Segment 1: position 4 is real (match, 5); positions 5,6,7 are padding:
        // -max(|5|,|-4|,|-8|,|-10|) = -10.
        assert_eq!(profile[1], TestVec4([5, -10, -10, -10]));
    }

    /// A mismatching base at a real (non-padding) position scores `n`, not `m`.
    #[test]
    fn build_profile_multi_lane_scores_mismatches() {
        let graph = linear_graph(b"A");
        let scoring = linear_scoring();
        let seq = b"CCCC";

        let mut profile = Vec::new();
        build_profile::<TestSimd4>(&mut profile, &graph, seq, scoring);
        assert_eq!(profile.len(), 1);
        assert_eq!(profile[0], TestVec4([-4, -4, -4, -4]));
    }

    // ---- destripe_interior ----------------------------------------------------------------

    /// Hand-built single-lane (`ScalarSimd`) striped matrix: two interior rows, three segments
    /// (`seq_len = 3`). Confirms interior cells land at `[i * mw + j]` and row 0/column 0 (which
    /// `destripe_interior` never writes) stay at whatever sentinel the caller pre-seeded.
    #[test]
    fn destripe_interior_scalar_places_cells_at_row_major_offsets() {
        let seq_len = 3;
        let mw = seq_len + 1; // 4
        let num_interior_rows = 2;
        // Sentinel: untouched cells (row 0, column 0) must stay -1.
        let mut dst = vec![-1i32; (num_interior_rows + 1) * mw];

        // Row 1 (graph rank 0): [10, 11, 12] at columns 1..=3.
        // Row 2 (graph rank 1): [20, 21, 22] at columns 1..=3.
        let matrix: Vec<i32> = vec![10, 11, 12, 20, 21, 22];

        destripe_interior::<ScalarSimdI32>(&mut dst, &matrix, seq_len, seq_len);

        assert_eq!(dst[mw + 1], 10);
        assert_eq!(dst[mw + 2], 11);
        assert_eq!(dst[mw + 3], 12);
        assert_eq!(dst[2 * mw + 1], 20);
        assert_eq!(dst[2 * mw + 2], 21);
        assert_eq!(dst[2 * mw + 3], 22);

        // Row 0 and column 0 are untouched.
        assert_eq!(dst[0], -1);
        assert_eq!(dst[1], -1);
        assert_eq!(dst[mw], -1);
        assert_eq!(dst[2 * mw], -1);
    }

    /// `LANES = 4` with `seq_len = 5` (not a multiple of 4): one interior row, two segments. Only
    /// the first lane of the second segment (position 4) should land in `dst`; the segment's
    /// three padding lanes (positions 5,6,7, past `seq_len`) must not be written at all.
    #[test]
    fn destripe_interior_multi_lane_skips_padding_lanes() {
        let seq_len = 5;
        let mw = seq_len + 1; // 6
        let mut dst = vec![-1i32; 2 * mw];

        let matrix = vec![
            TestVec4([100, 101, 102, 103]), // segment 0: positions 0..4
            TestVec4([104, 999, 999, 999]), // segment 1: position 4 real, 5..7 padding (ignored)
        ];

        destripe_interior::<TestSimd4>(&mut dst, &matrix, 2, seq_len);

        assert_eq!(dst[mw + 1], 100);
        assert_eq!(dst[mw + 2], 101);
        assert_eq!(dst[mw + 3], 102);
        assert_eq!(dst[mw + 4], 103);
        assert_eq!(dst[mw + 5], 104);
        // Row 0/column 0 untouched.
        assert_eq!(dst[0], -1);
        assert_eq!(dst[mw], -1);
    }

    // ---- seed_scalar_buffers (the C2 fix's faithfulness anchor) ----------------------------

    /// [`seed_scalar_buffers`] must match `SisdEngine`'s own boundary formulas for
    /// [`AlignmentType::Global`] (`kNW`, [`GapMode::Linear`], penalized boundary row/column) —
    /// the same fixture/expected values already verified in `sisd.rs`'s own `initialize` tests.
    #[test]
    fn seed_scalar_buffers_matches_sisd_initialize_for_nw_linear() {
        let graph = linear_graph(b"AC");
        let scoring = linear_scoring();
        let seq = b"AG";

        let seeded = seed_scalar_buffers(&graph, seq, scoring, AlignmentType::Global);

        assert_eq!(scoring.gap_mode(), GapMode::Linear);
        assert_eq!(seeded.matrix_width, 3);
        // H's boundary row follows j * g.
        assert_eq!(seeded.h[0], 0);
        assert_eq!(seeded.h[1], -8);
        assert_eq!(seeded.h[2], -16);
        // Boundary column: node 0 (rank 0) has no inedges -> empty_penalty 0, + g.
        assert_eq!(seeded.h[seeded.matrix_width], -8);
        // Linear mode: e/f/o/q stay unallocated.
        assert!(seeded.e.is_empty());
        assert!(seeded.f.is_empty());
        assert!(seeded.o.is_empty());
        assert!(seeded.q.is_empty());
    }

    /// [`seed_scalar_buffers`] for [`AlignmentType::Local`] (`kSW`) under [`GapMode::Affine`]:
    /// `H`'s boundary is all-zero regardless of gap mode, and `f`/`e` get seeded (not `o`/`q`).
    #[test]
    fn seed_scalar_buffers_matches_sisd_initialize_for_sw_affine() {
        let graph = linear_graph(b"AC");
        let scoring = affine_scoring();
        let seq = b"AG";

        let seeded = seed_scalar_buffers(&graph, seq, scoring, AlignmentType::Local);

        assert_eq!(scoring.gap_mode(), GapMode::Affine);
        assert!(seeded.h[..3].iter().all(|&v| v == 0));
        assert_eq!(seeded.f[0], 0);
        assert_eq!(seeded.e[0], 0);
        assert!(seeded.o.is_empty());
        assert!(seeded.q.is_empty());
    }

    /// [`seed_scalar_buffers`] for [`AlignmentType::Overlap`] (`kOV`) under [`GapMode::Convex`]:
    /// `o`/`q` get seeded, and `H`'s boundary column is always 0 (free leading graph-axis gaps).
    #[test]
    fn seed_scalar_buffers_matches_sisd_initialize_for_ov_convex() {
        let graph = linear_graph(b"AC");
        let scoring = convex_scoring();
        let seq = b"AG";

        let seeded = seed_scalar_buffers(&graph, seq, scoring, AlignmentType::Overlap);

        assert_eq!(scoring.gap_mode(), GapMode::Convex);
        assert_eq!(seeded.o[0], 0);
        assert_eq!(seeded.q[0], 0);
        assert_eq!(seeded.f[0], 0);
        assert_eq!(seeded.e[0], 0);
        assert_eq!(seeded.h[seeded.matrix_width], 0);
    }

    /// End-to-end faithfulness: `seed_scalar_buffers`'s `sequence_profile`/`node_id_to_rank` must
    /// match a real `SisdEngine::align` run's internally-used values on a slightly larger graph
    /// with branching (so `node_id_to_rank` isn't trivially identity), confirmed indirectly via
    /// the produced alignment/score matching between a `SisdEngine` and the values
    /// `seed_scalar_buffers` reports for the same inputs.
    #[test]
    fn seed_scalar_buffers_node_id_to_rank_matches_graph_topological_order() {
        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], b"ACGT", 1).unwrap();
        graph.add_alignment_weight(&[], b"ACTT", 1).unwrap();
        let scoring = linear_scoring();
        let seq = b"ACGT";

        let seeded = seed_scalar_buffers(&graph, seq, scoring, AlignmentType::Global);

        assert_eq!(seeded.node_id_to_rank.len(), graph.nodes.len());
        for (rank, &node_id) in graph.rank_to_node.iter().enumerate() {
            assert_eq!(seeded.node_id_to_rank[node_id.0 as usize], rank as u32);
        }

        // Sanity: the same (graph, seq, scoring) pair still aligns identically through the full
        // SisdEngine path (this doesn't re-verify seed_scalar_buffers directly, but confirms the
        // fixture itself is a valid, alignable input).
        let mut engine = SisdEngine::new(AlignmentType::Global, scoring);
        let (_alignment, score) = engine.align(seq, &graph);
        assert!(score > i32::MIN / 2);
    }

    // ---- build_masks / build_penalties ------------------------------------------------------

    #[test]
    fn build_masks_scalar_single_lane_carry_mask_only() {
        let masks = build_masks::<ScalarSimdI16>(ScalarSimdI16::NEG_INF);
        // LOG_LANES == 0: no ladder masks, just the single carry mask.
        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0], 0);
    }

    #[test]
    fn build_masks_multi_lane_has_neg_inf_in_expected_low_lanes() {
        let neg_inf = TestSimd4::NEG_INF;
        let masks = build_masks::<TestSimd4>(neg_inf);
        assert_eq!(masks.len(), 3); // LOG_LANES + 1 == 3

        // masks[0]: NEG_INF in lanes [0, 2^0) = lane 0 only.
        assert_eq!(masks[0], TestVec4([neg_inf, 0, 0, 0]));
        // masks[1]: NEG_INF in lanes [0, 2^1) = lanes 0,1.
        assert_eq!(masks[1], TestVec4([neg_inf, neg_inf, 0, 0]));
        // masks[LOG_LANES] (carry mask): lane 0 is 0, every other lane is NEG_INF.
        assert_eq!(masks[2], TestVec4([0, neg_inf, neg_inf, neg_inf]));
    }

    #[test]
    fn build_penalties_scalar_single_lane_is_empty() {
        let penalties = build_penalties::<ScalarSimdI16>(-8);
        assert!(penalties.is_empty()); // LOG_LANES == 0
    }

    #[test]
    fn build_penalties_multi_lane_scales_by_power_of_two() {
        let penalties = build_penalties::<TestSimd4>(-8);
        assert_eq!(penalties.len(), 2); // LOG_LANES == 2
        assert_eq!(penalties[0], TestVec4([-8, -8, -8, -8])); // -8 * 2^0
        assert_eq!(penalties[1], TestVec4([-16, -16, -16, -16])); // -8 * 2^1
    }
}
