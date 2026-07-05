//! The per-ISA `Simd` trait: the vectorized-fill primitives every real ISA backend
//! (SSE4.1/NEON/AVX2) will implement, plus a `ScalarSimd` (`LANES = 1`) reference impl.
//!
//! Ports the `InstructionSet<Architecture, T>` template from
//! `third_party/spoa/src/simd_alignment_engine_implementation.hpp` (`:59-220`) as a Rust trait:
//! upstream picks one of three `InstructionSet` specializations (AVX2/int16, AVX2/int32,
//! SSE4.1/int16, SSE4.1/int32) at compile time via preprocessor `#if`; this crate instead defines
//! one associated-type-generic `Simd` trait and gives each (ISA, element width) pair its own
//! implementing type, selected at runtime by the dispatch logic later tasks add.
//!
//! `ScalarSimd{I16,I32}` implement `Simd` with `Vec = Elem` and `LANES = 1`: no intrinsics, no
//! `unsafe`. They exist purely so Task 3's generic DP fill can be written and unit-tested against
//! *some* `Simd` impl before any real vectorized backend lands — a one-lane "vector" degenerates
//! every horizontal (cross-lane) operation to a no-op or identity, which validates the fill's
//! *structure* (buffer indexing, inter-segment carry) but deliberately exercises none of the
//! shift-and-max ladder machinery. That ladder is instead unit-tested directly against each real
//! ISA's `prefix_max` in Tasks 6/12/14, and exercised end-to-end starting at Task 7.

/// The per-(ISA, element-width) vectorized primitives the generic DP fill (Task 3 onward) is
/// written against.
///
/// Mirrors upstream's `InstructionSet<Architecture, T>` (`impl:59-220`): `Elem` is `T::type`
/// (`i16` or `i32`), `Vec` is `__mxxxi` (`__m128i`/`__m256i`, or — for [`ScalarSimd`] — `Elem`
/// itself), and `LANES`/`LOG_LANES` are `T::kNumVar`/`T::kLogNumVar`. `LSS`/`RSS` are the
/// left-/right-shift byte counts upstream bakes into each `_mmxxx_prefix_max` (`impl:88-91` etc.):
/// `LSS` is one lane's width in bytes (`size_of::<Elem>()`), `RSS` is "every other lane's width"
/// (`reg_bytes - size_of::<Elem>()`) — the shift that isolates the single highest lane.
// Not yet implemented by any real ISA backend outside this module's own tests (that lands in
// Task 6 onward) — see `SimdEngine`'s identical `#[allow(dead_code)]` rationale in `mod.rs`.
#[allow(dead_code)]
pub(crate) trait Simd {
    /// The lane element type: `i16` or `i32`.
    type Elem: Copy;
    /// The register type: `__m128i`/`__m256i` for real ISA backends, or (for [`ScalarSimd`])
    /// bare `Elem` — a "register" holding exactly one lane.
    type Vec: Copy;

    /// Lane count (upstream `T::kNumVar`, `impl:62,98,156,191`).
    const LANES: usize;
    /// `log2(LANES)` — the number of shift-and-max ladder steps [`Simd::prefix_max`] needs
    /// (upstream `T::kLogNumVar`, `impl:102,160,195`; AVX2/int16's `kLogNumVar = 4` matches its
    /// `LANES = 16`, though the constant itself is asserted `3` at `impl:102` on the AVX2/int32
    /// branch shown above — see each real ISA impl's own doc for its exact ladder length).
    const LOG_LANES: u32;
    /// Left-shift byte count: one lane's width, `size_of::<Elem>()` (upstream `T::kLSS`,
    /// `impl:103,161,196`).
    const LSS: i32;
    /// Right-shift byte count: every lane *except* one, `reg_bytes - size_of::<Elem>()` (upstream
    /// `T::kRSS`, `impl:104,162,197`) — shifts the single highest-index lane down to lane 0.
    const RSS: i32;
    /// The DP "negative infinity" sentinel for this element type: `Elem::MIN + 1024` (same
    /// +1024 headroom as the scalar engine's [`super::super::sisd::NEG_INF`], so a single
    /// non-saturating add never wraps past it). Upstream computes this per-call as
    /// `kNegativeInfinity` (`impl:494,738,1087,1533`); here it is a trait constant instead.
    const NEG_INF: Self::Elem;

    /// Broadcasts `value` into every lane. Ports `_mmxxx_set1_epi` (`impl:81,117,175,210`).
    fn splat(value: Self::Elem) -> Self::Vec;

    /// Lane-wise **non-saturating** (wrapping) addition. Ports `_mmxxx_add_epi`
    /// (`impl:69,105,163,198`, e.g. `_mm_add_epi16`) — deliberately the modular-arithmetic
    /// add, NOT a saturating one: per the SIMD kernels plan's Global Constraints, the `+1024`
    /// headroom baked into [`Simd::NEG_INF`] is what prevents overflow, not saturation.
    fn add(a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Lane-wise **non-saturating** (wrapping) subtraction. Ports `_mmxxx_sub_epi`
    /// (`impl:72,108,166,201`); see [`Simd::add`] for why this must not saturate.
    fn sub(a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Lane-wise signed minimum. Ports `_mmxxx_min_epi` (`impl:75,111,169,204`).
    fn min(a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Lane-wise signed maximum. Ports `_mmxxx_max_epi` (`impl:78,114,172,207`).
    fn max(a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Lane-wise (equivalently, bitwise) OR. Ports `_mmxxx_or_si` (`impl:48,143`) — used by
    /// [`Simd::prefix_max`] to splice a `NEG_INF`-patterned mask into the lanes a shift vacates.
    fn or(a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Unaligned load of `Self::LANES` elements from the front of `src`. Ports upstream's
    /// (aligned) `_mmxxx_load_si` (`impl:40,135`); this crate prefers unaligned loads/stores
    /// throughout (see the SIMD kernels plan's Global Constraints) to avoid alignment
    /// bookkeeping, at negligible cost on modern cores.
    ///
    /// `src` must have length `>= Self::LANES`.
    fn loadu(src: &[Self::Elem]) -> Self::Vec;

    /// Unaligned store of `v`'s `Self::LANES` elements to the front of `dst`. Ports upstream's
    /// (aligned) `_mmxxx_store_si` (`impl:44,139`); see [`Simd::loadu`] on the unaligned choice.
    ///
    /// `dst` must have length `>= Self::LANES`.
    fn storeu(v: Self::Vec, dst: &mut [Self::Elem]);

    /// Shifts `v` left by `N` *bytes* (not lanes), zero-filling the vacated low-order bytes.
    /// Ports `_mmxxx_slli_si` (`impl:52-55,147-148`, e.g. `_mm_slli_si128`/`_mm256_slli_si256`).
    /// `N` is a compile-time constant, matching every real ISA's immediate-operand shift
    /// instruction (see the SIMD kernels plan's Global Constraints on const-generic shifts).
    fn slli<const N: i32>(v: Self::Vec) -> Self::Vec;

    /// Shifts `v` right by `N` *bytes* (not lanes), zero-filling the vacated high-order bytes.
    /// Ports `_mmxxx_srli_si` (`impl:56-59,150-151`, e.g. `_mm_srli_si128`/`_mm256_srli_si256`).
    fn srli<const N: i32>(v: Self::Vec) -> Self::Vec;

    /// Shifts every lane up by one index (left by [`Simd::LSS`] bytes), zero-filling lane 0 — the
    /// striped-fill *diagonal* shift `_mmxxx_slli_si(v, T::kLSS)` (`impl:787,810`). This is a
    /// dedicated method rather than `slli::<{ Self::LSS }>` because Rust (stable) cannot use an
    /// associated const as a const-generic argument (`error: generic parameters may not be used in
    /// const operations`); each impl therefore supplies its own literal `LSS` byte count.
    fn slli_one_lane(v: Self::Vec) -> Self::Vec;

    /// Isolates the single highest-index lane into lane 0 (shift right by [`Simd::RSS`] bytes),
    /// zero-filling the rest — the striped-fill inter-segment *carry* shift
    /// `_mmxxx_srli_si(v, T::kRSS)` (`impl:779,785,802,824,838`). See [`Simd::slli_one_lane`] for
    /// why this is a dedicated method rather than `srli::<{ Self::RSS }>`.
    fn srli_top_lane(v: Self::Vec) -> Self::Vec;

    /// Reduces `v` to the maximum of its lanes, seeded at `0` (**not** `Self::Elem::MIN`/
    /// `NEG_INF`). Ports `_mmxxx_max_value` (`impl:240-250`) EXACTLY, including its `max_score =
    /// 0` seed (`impl:242`): that `0` is the Smith-Waterman clamp (a local alignment's score
    /// never goes negative), so folding lanes with a `0` seed is load-bearing, not an arbitrary
    /// starting point — a bare hardware horizontal-max instruction (e.g. NEON's `vmaxvq`) would
    /// be WRONG here and must not be substituted in a real ISA impl.
    fn horizontal_max(v: Self::Vec) -> Self::Elem;

    /// The shift-and-max "prefix max" ladder that resolves the horizontal (gap-left / E)
    /// dependency within one vector: after this call, lane `i` of the result holds
    /// `max(v[0]+penalties applied appropriately, ..., v[i])` per upstream's recurrence. Ports
    /// `_mmxxx_prefix_max` (`impl:84-92,120-126,178-185,213-218`) — four near-identical
    /// hand-unrolled ladders (one per (ISA, element width) pair), each a fixed sequence of
    /// `LOG_LANES` steps of the form
    /// `v = max(v, or(masks[k], slli(add(v, penalties[k]), shift_k)))`
    /// where `shift_k = (1 << k) * LSS` bytes. Because the shift amount must be a compile-time
    /// constant per the SIMD kernels plan's Global Constraints, this is a **required** method
    /// (each real ISA hand-unrolls its own ladder) rather than a generic provided method with a
    /// runtime-length loop.
    ///
    /// `penalties` and `masks` must each have length `>= Self::LOG_LANES as usize` (both are
    /// empty when `LOG_LANES == 0`, i.e. for [`ScalarSimd`] — the degenerate 1-lane case, where
    /// this is simply the identity function and neither slice is read).
    fn prefix_max(v: Self::Vec, penalties: &[Self::Vec], masks: &[Self::Vec]) -> Self::Vec;

    /// One step of the [`Simd::prefix_max`] shift-and-max ladder, factored out so every real ISA's
    /// hand-unrolled ladder (SSE4.1 int16/int32, NEON, AVX2) invokes the *same* op sequence at each
    /// step — cutting the copy-paste/typo risk across the four ladders down to a single per-step
    /// literal shift constant `N`. Computes
    /// `max(a, or(mask, slli::<N>(add(a, penalty))))` — exactly upstream's
    /// `_mmxxx_max_epi(a, _mmxxx_or_si(masks[k], _mmxxx_slli_si(_mmxxx_add_epi(a, penalties[k]),
    /// shift_k)))` (`impl:88-91,124-126,182-184,217-218`), where `N` is that step's byte-shift
    /// `shift_k = (1 << k) * LSS`.
    ///
    /// Provided as a default method (not overridden by any impl): it is expressed purely in terms
    /// of the other trait ops, so `ScalarSimd`/`TestSimd4` and every real ISA inherit it unchanged.
    fn prefix_max_step<const N: i32>(
        a: Self::Vec,
        penalty: Self::Vec,
        mask: Self::Vec,
    ) -> Self::Vec {
        Self::max(a, Self::or(mask, Self::slli::<N>(Self::add(a, penalty))))
    }
}

/// `i16::MIN + 1024`, upstream's `kNegativeInfinity` for the int16 element width
/// (`impl:494` etc., instantiated with `T::type = std::int16_t`).
#[allow(dead_code)]
const NEG_INF_I16: i16 = i16::MIN + 1024;

/// `i32::MIN + 1024`, upstream's `kNegativeInfinity` for the int32 element width — the same value
/// as [`super::super::sisd::NEG_INF`], computed independently here since `lanes` is a
/// self-contained abstraction layer.
#[allow(dead_code)]
const NEG_INF_I32: i32 = i32::MIN + 1024;

/// A one-lane, intrinsic-free [`Simd`] reference impl over `i16`, used only to exercise the
/// generic DP fill's structure before any real ISA backend exists. See the module doc for why
/// its degenerate `LANES = 1` behavior deliberately validates none of the cross-lane shift
/// machinery.
#[allow(dead_code)]
pub(crate) struct ScalarSimdI16;

impl Simd for ScalarSimdI16 {
    type Elem = i16;
    type Vec = i16;

    const LANES: usize = 1;
    const LOG_LANES: u32 = 0;
    const LSS: i32 = size_of::<i16>() as i32;
    const RSS: i32 = 0;
    const NEG_INF: i16 = NEG_INF_I16;

    fn splat(value: i16) -> i16 {
        value
    }

    fn add(a: i16, b: i16) -> i16 {
        a.wrapping_add(b)
    }

    fn sub(a: i16, b: i16) -> i16 {
        a.wrapping_sub(b)
    }

    fn min(a: i16, b: i16) -> i16 {
        a.min(b)
    }

    fn max(a: i16, b: i16) -> i16 {
        a.max(b)
    }

    fn or(a: i16, b: i16) -> i16 {
        a | b
    }

    fn loadu(src: &[i16]) -> i16 {
        src[0]
    }

    fn storeu(v: i16, dst: &mut [i16]) {
        dst[0] = v;
    }

    /// At `LANES = 1` there is no second lane to shift in, so — matching the plan's degenerate
    /// caveat — this returns [`Simd::NEG_INF`] directly rather than a raw zero-fill: it stands in
    /// for what a real ISA's `slli` composed with its `prefix_max` mask-OR would produce (a
    /// vacated lane reading as `NEG_INF`, not `0`).
    fn slli<const N: i32>(_v: i16) -> i16 {
        Self::NEG_INF
    }

    /// See [`ScalarSimdI16::slli`]: the same one-lane degenerate NEG_INF fill, mirrored for the
    /// right shift.
    fn srli<const N: i32>(_v: i16) -> i16 {
        Self::NEG_INF
    }

    /// Degenerate one-lane diagonal shift: shifting the single lane up leaves lane 0 vacated, so —
    /// like [`ScalarSimdI16::slli`] — this returns [`Simd::NEG_INF`]. Never exercised by a real
    /// fill at `LANES = 1` (see the module doc).
    fn slli_one_lane(_v: i16) -> i16 {
        Self::NEG_INF
    }

    /// Degenerate one-lane carry shift: with `LANES = 1` there is no higher lane to isolate, so —
    /// like [`ScalarSimdI16::srli`] — this returns [`Simd::NEG_INF`]. Never exercised by a real
    /// fill at `LANES = 1`.
    fn srli_top_lane(_v: i16) -> i16 {
        Self::NEG_INF
    }

    fn horizontal_max(v: i16) -> i16 {
        0i16.max(v)
    }

    /// Identity: `LOG_LANES == 0` means zero ladder steps, so `penalties`/`masks` (both always
    /// empty here) are never read.
    fn prefix_max(v: i16, _penalties: &[i16], _masks: &[i16]) -> i16 {
        v
    }
}

/// A one-lane, intrinsic-free [`Simd`] reference impl over `i32`. See [`ScalarSimdI16`] and the
/// module doc for the rationale and the exact same degenerate behavior, mirrored at the `i32`
/// element width.
#[allow(dead_code)]
pub(crate) struct ScalarSimdI32;

impl Simd for ScalarSimdI32 {
    type Elem = i32;
    type Vec = i32;

    const LANES: usize = 1;
    const LOG_LANES: u32 = 0;
    const LSS: i32 = size_of::<i32>() as i32;
    const RSS: i32 = 0;
    const NEG_INF: i32 = NEG_INF_I32;

    fn splat(value: i32) -> i32 {
        value
    }

    fn add(a: i32, b: i32) -> i32 {
        a.wrapping_add(b)
    }

    fn sub(a: i32, b: i32) -> i32 {
        a.wrapping_sub(b)
    }

    fn min(a: i32, b: i32) -> i32 {
        a.min(b)
    }

    fn max(a: i32, b: i32) -> i32 {
        a.max(b)
    }

    fn or(a: i32, b: i32) -> i32 {
        a | b
    }

    fn loadu(src: &[i32]) -> i32 {
        src[0]
    }

    fn storeu(v: i32, dst: &mut [i32]) {
        dst[0] = v;
    }

    /// See [`ScalarSimdI16::slli`]: the same one-lane degenerate NEG_INF fill.
    fn slli<const N: i32>(_v: i32) -> i32 {
        Self::NEG_INF
    }

    /// See [`ScalarSimdI16::slli`]: the same one-lane degenerate NEG_INF fill, for the right
    /// shift.
    fn srli<const N: i32>(_v: i32) -> i32 {
        Self::NEG_INF
    }

    /// See [`ScalarSimdI16::slli_one_lane`]: the same one-lane degenerate NEG_INF fill.
    fn slli_one_lane(_v: i32) -> i32 {
        Self::NEG_INF
    }

    /// See [`ScalarSimdI16::srli_top_lane`]: the same one-lane degenerate NEG_INF fill.
    fn srli_top_lane(_v: i32) -> i32 {
        Self::NEG_INF
    }

    fn horizontal_max(v: i32) -> i32 {
        0i32.max(v)
    }

    /// Identity: see [`ScalarSimdI16::prefix_max`].
    fn prefix_max(v: i32, _penalties: &[i32], _masks: &[i32]) -> i32 {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_inf_matches_elem_min_plus_1024() {
        assert_eq!(ScalarSimdI16::NEG_INF, i16::MIN + 1024);
        assert_eq!(ScalarSimdI32::NEG_INF, i32::MIN + 1024);
    }

    #[test]
    fn i16_splat_is_identity() {
        assert_eq!(ScalarSimdI16::splat(7), 7);
        assert_eq!(ScalarSimdI16::splat(-3), -3);
    }

    #[test]
    fn i16_arithmetic_matches_plain_integer_ops() {
        assert_eq!(ScalarSimdI16::add(3, 4), 7);
        assert_eq!(ScalarSimdI16::sub(10, 4), 6);
        assert_eq!(ScalarSimdI16::min(10, 4), 4);
        assert_eq!(ScalarSimdI16::max(10, 4), 10);
        assert_eq!(ScalarSimdI16::or(0b0101, 0b1010), 0b1111);
    }

    #[test]
    fn i16_add_sub_are_non_saturating() {
        // Wrapping, not saturating: i16::MAX + 1 wraps to i16::MIN, matching a plain hardware
        // `_mm_add_epi16` (modular arithmetic), NOT a saturating add.
        assert_eq!(ScalarSimdI16::add(i16::MAX, 1), i16::MIN);
        assert_eq!(ScalarSimdI16::sub(i16::MIN, 1), i16::MAX);
    }

    #[test]
    fn i16_horizontal_max_seeds_at_zero_not_elem_min() {
        // The Smith-Waterman clamp: a negative single lane still reduces to 0, not the lane's
        // own (negative) value and not `Elem::MIN`.
        assert_eq!(ScalarSimdI16::horizontal_max(-5), 0);
        assert_eq!(ScalarSimdI16::horizontal_max(7), 7);
        assert_eq!(ScalarSimdI16::horizontal_max(0), 0);
    }

    #[test]
    fn i16_prefix_max_of_one_lane_is_identity() {
        // LOG_LANES == 0 => zero ladder steps => the empty penalties/masks slices are never
        // indexed, and the input vector passes through unchanged regardless of their contents.
        assert_eq!(ScalarSimdI16::prefix_max(42, &[], &[]), 42);
        assert_eq!(ScalarSimdI16::prefix_max(-9, &[], &[]), -9);
    }

    #[test]
    fn i16_loadu_storeu_round_trip() {
        let src = [11i16];
        let v = ScalarSimdI16::loadu(&src);
        let mut dst = [0i16];
        ScalarSimdI16::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_slli_srli_of_one_lane_return_neg_inf() {
        assert_eq!(ScalarSimdI16::slli::<2>(5), ScalarSimdI16::NEG_INF);
        assert_eq!(ScalarSimdI16::srli::<2>(5), ScalarSimdI16::NEG_INF);
    }

    #[test]
    fn i32_splat_is_identity() {
        assert_eq!(ScalarSimdI32::splat(7), 7);
        assert_eq!(ScalarSimdI32::splat(-3), -3);
    }

    #[test]
    fn i32_arithmetic_matches_plain_integer_ops() {
        assert_eq!(ScalarSimdI32::add(3, 4), 7);
        assert_eq!(ScalarSimdI32::sub(10, 4), 6);
        assert_eq!(ScalarSimdI32::min(10, 4), 4);
        assert_eq!(ScalarSimdI32::max(10, 4), 10);
        assert_eq!(ScalarSimdI32::or(0b0101, 0b1010), 0b1111);
    }

    #[test]
    fn i32_add_sub_are_non_saturating() {
        assert_eq!(ScalarSimdI32::add(i32::MAX, 1), i32::MIN);
        assert_eq!(ScalarSimdI32::sub(i32::MIN, 1), i32::MAX);
    }

    #[test]
    fn i32_horizontal_max_seeds_at_zero_not_elem_min() {
        assert_eq!(ScalarSimdI32::horizontal_max(-5), 0);
        assert_eq!(ScalarSimdI32::horizontal_max(7), 7);
        assert_eq!(ScalarSimdI32::horizontal_max(0), 0);
    }

    #[test]
    fn i32_prefix_max_of_one_lane_is_identity() {
        assert_eq!(ScalarSimdI32::prefix_max(42, &[], &[]), 42);
        assert_eq!(ScalarSimdI32::prefix_max(-9, &[], &[]), -9);
    }

    #[test]
    fn i32_loadu_storeu_round_trip() {
        let src = [11i32];
        let v = ScalarSimdI32::loadu(&src);
        let mut dst = [0i32];
        ScalarSimdI32::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i32_slli_srli_of_one_lane_return_neg_inf() {
        assert_eq!(ScalarSimdI32::slli::<4>(5), ScalarSimdI32::NEG_INF);
        assert_eq!(ScalarSimdI32::srli::<4>(5), ScalarSimdI32::NEG_INF);
    }

    #[test]
    fn lanes_and_log_lanes_are_degenerate() {
        assert_eq!(ScalarSimdI16::LANES, 1);
        assert_eq!(ScalarSimdI16::LOG_LANES, 0);
        assert_eq!(ScalarSimdI32::LANES, 1);
        assert_eq!(ScalarSimdI32::LOG_LANES, 0);
    }
}
