//! SSE4.1 [`Simd`] backends: `Sse41I16` (8×`i16`) and `Sse41I32` (4×`i32`), the first *real*
//! vectorized ISA implementation of the [`Simd`] trait.
//!
//! Ports the `InstructionSet<A, std::int16_t>` / `InstructionSet<A, std::int32_t>` SSE4.1
//! specializations from `third_party/spoa/src/simd_alignment_engine_implementation.hpp`
//! (`impl:130-220`, the `#elif defined(__SSE4_1__)` branch): 128-bit `__m128i` registers, packing
//! 8 `i16` or 4 `i32` lanes. Every op maps to a single `core::arch::x86_64` intrinsic; the
//! [`Simd::prefix_max`] shift-and-max ladders are hand-unrolled per width (int16 byte-shifts
//! `[2, 4, 8]` at `impl:182-184`, int32 `[4, 8]` at `impl:217-218`) because each `_mm_slli_si128`
//! amount must be a compile-time immediate (see the plan's Global Constraints).
//!
//! # Safety
//!
//! Every intrinsic-calling helper is a `#[target_feature(enable = "sse4.1")] unsafe fn`: calling
//! one is undefined behavior unless the running CPU actually has SSE4.1. Each [`Simd`] trait method
//! below wraps its helper in an `unsafe` block whose safety precondition is *"the caller only
//! reaches this impl after `is_x86_feature_detected!(\"sse4.1\")` returned true"* — which is exactly
//! how the runtime dispatch in [`super`] selects [`Sse41I16`]/[`Sse41I32`], and how every test in
//! this module is gated. This is the standard `std::arch` "detect once, then call `target_feature`
//! code" idiom. All `unsafe` in the crate is confined to `src/align/simd/` (this file included) via
//! the module's `#![allow(unsafe_code)]`.

use super::lanes::Simd;
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16, _mm_and_si128, _mm_blendv_epi8,
    _mm_cvtepi16_epi32, _mm_loadu_si128, _mm_max_epi16, _mm_max_epi32, _mm_min_epi16,
    _mm_min_epi32, _mm_or_si128, _mm_set1_epi16, _mm_set1_epi32, _mm_slli_si128, _mm_srai_epi32,
    _mm_srli_epi32, _mm_srli_si128, _mm_storeu_si128, _mm_sub_epi16, _mm_sub_epi32, _mm_subs_epi16,
    _mm_xor_si128,
};

/// `i16::MIN + 1024`, this backend's `kNegativeInfinity` (see [`Simd::NEG_INF`]).
const NEG_INF_I16: i16 = i16::MIN + 1024;
/// `i32::MIN + 1024`, this backend's `kNegativeInfinity` (see [`Simd::NEG_INF`]).
const NEG_INF_I32: i32 = i32::MIN + 1024;

// ---- shared, type-agnostic `__m128i` helpers -------------------------------------------------
//
// `or`/`slli`/`srli`/`loadu`/`storeu` operate on the whole 128-bit register regardless of lane
// width, so both `Sse41I16` and `Sse41I32` share these; only the lane-typed arithmetic
// (`add`/`sub`/`min`/`max`/`set1`) differs between the two.

/// Bitwise OR of two registers. Ports `_mm_or_si128` (`impl:143`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn or_si(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(a, b)
}

/// Byte-shift-left by the compile-time constant `N`. Ports `_mm_slli_si128` (`impl:147-148`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn slli_si<const N: i32>(v: __m128i) -> __m128i {
    _mm_slli_si128::<N>(v)
}

/// Byte-shift-right by the compile-time constant `N`. Ports `_mm_srli_si128` (`impl:150-151`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn srli_si<const N: i32>(v: __m128i) -> __m128i {
    _mm_srli_si128::<N>(v)
}

/// Unaligned 128-bit load from `src`. Ports `_mm_loadu_si128` (unaligned per the plan's alignment
/// decision; upstream uses the aligned `_mm_load_si128` at `impl:135`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available AND that `src` points to at least 16 readable bytes.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn loadu(src: *const u8) -> __m128i {
    _mm_loadu_si128(src.cast::<__m128i>())
}

/// Unaligned 128-bit store to `dst`. Ports `_mm_storeu_si128` (see [`loadu`] on the unaligned
/// choice; upstream uses the aligned `_mm_store_si128` at `impl:139`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available AND that `dst` points to at least 16 writable bytes.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn storeu(dst: *mut u8, v: __m128i) {
    _mm_storeu_si128(dst.cast::<__m128i>(), v);
}

/// Sign-extends the 8 packed `i16` lanes of `v` to `i32` and stores them contiguously to the 8
/// `i32` slots at `dst` (32 bytes). Widens each 128-bit half with `_mm_cvtepi16_epi32` (low 4 lanes
/// directly, high 4 after a byte-shift-right of 8) and stores the two resulting 4×`i32` vectors.
///
/// # Safety
/// Caller must guarantee SSE4.1 is available AND that `dst` points to at least 8 writable `i32`
/// (32 bytes).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_widen_i16(dst: *mut i32, v: __m128i) {
    let lo = _mm_cvtepi16_epi32(v);
    let hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v));
    _mm_storeu_si128(dst.cast::<__m128i>(), lo);
    _mm_storeu_si128(dst.add(4).cast::<__m128i>(), hi);
}

// ---- Sse41I16: 8 × i16 -----------------------------------------------------------------------

/// SSE4.1 backend over 8 packed `i16` lanes (`impl:154-186`). Wired into `SimdEngine`'s int16
/// linear-NW dispatch (SIMD kernels plan Task 7).
pub(crate) struct Sse41I16;

/// Lane-typed `i16` add. Ports `_mm_add_epi16` (`impl:163`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn add16(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi16(a, b)
}

/// Lane-typed `i16` sub. Ports `_mm_sub_epi16` (`impl:166`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn sub16(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi16(a, b)
}

/// Lane-typed `i16` **saturating** add. Ports `_mm_adds_epi16` — native saturating hardware
/// support, unlike the int32 backend (see [`adds32`]) which SSE4.1 has no intrinsic for.
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn adds16(a: __m128i, b: __m128i) -> __m128i {
    _mm_adds_epi16(a, b)
}

/// Lane-typed `i16` **saturating** sub. Ports `_mm_subs_epi16`.
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn subs16(a: __m128i, b: __m128i) -> __m128i {
    _mm_subs_epi16(a, b)
}

/// Lane-typed `i16` signed min. Ports `_mm_min_epi16` (`impl:169`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
// The `min` trait op is test-only: the DP fill maximizes score (so it uses `max`, never `min`),
// so this faithful-port helper is dead in non-test builds.
#[allow(dead_code)]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn min16(a: __m128i, b: __m128i) -> __m128i {
    _mm_min_epi16(a, b)
}

/// Lane-typed `i16` signed max. Ports `_mm_max_epi16` (`impl:172`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn max16(a: __m128i, b: __m128i) -> __m128i {
    _mm_max_epi16(a, b)
}

/// Broadcast `value` into all 8 `i16` lanes. Ports `_mm_set1_epi16` (`impl:175`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn set1_16(value: i16) -> __m128i {
    _mm_set1_epi16(value)
}

impl Simd for Sse41I16 {
    type Elem = i16;
    type Vec = __m128i;

    const LANES: usize = 8;
    const LOG_LANES: u32 = 3;
    const LSS: i32 = size_of::<i16>() as i32; // 2
    const RSS: i32 = 16 - size_of::<i16>() as i32; // 14
    const NEG_INF: i16 = NEG_INF_I16;

    #[inline(always)]
    fn splat(value: i16) -> __m128i {
        // SAFETY: only reached after `is_x86_feature_detected!("sse4.1")` (see module Safety note).
        unsafe { set1_16(value) }
    }

    #[inline(always)]
    fn add(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Non-saturating `_mm_add_epi16` per the plan's Global Constraints.
        unsafe { add16(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Non-saturating `_mm_sub_epi16`.
        unsafe { sub16(a, b) }
    }

    #[inline(always)]
    fn adds(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Native saturating `_mm_adds_epi16`.
        unsafe { adds16(a, b) }
    }

    #[inline(always)]
    fn subs(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Native saturating `_mm_subs_epi16`.
        unsafe { subs16(a, b) }
    }

    #[inline(always)]
    fn min(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { min16(a, b) }
    }

    #[inline(always)]
    fn max(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { max16(a, b) }
    }

    #[inline(always)]
    fn or(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { or_si(a, b) }
    }

    #[inline(always)]
    fn loadu(src: &[i16]) -> __m128i {
        debug_assert!(src.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` above (and the trait's documented precondition)
        // guarantee `src` covers the 16 bytes read.
        unsafe { loadu(src.as_ptr().cast::<u8>()) }
    }

    #[inline(always)]
    fn storeu(v: __m128i, dst: &mut [i16]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `loadu`, mirrored for the 16-byte write.
        unsafe { storeu(dst.as_mut_ptr().cast::<u8>(), v) }
    }

    #[inline(always)]
    fn store_widened_i32(v: __m128i, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` guarantees `dst` covers the 8 `i32` written.
        unsafe { store_widen_i16(dst.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn slli<const N: i32>(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<N>(v) }
    }

    #[inline(always)]
    fn srli<const N: i32>(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<N>(v) }
    }

    /// Diagonal shift by `LSS = 2` bytes (one `i16` lane), the literal for this width.
    #[inline(always)]
    fn slli_one_lane(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<2>(v) }
    }

    /// Carry shift by `RSS = 14` bytes (isolate lane 7 into lane 0), the literal for this width.
    #[inline(always)]
    fn srli_top_lane(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<14>(v) }
    }

    #[inline(always)]
    fn horizontal_max(v: __m128i) -> i16 {
        let mut lanes = [0i16; 8];
        Self::storeu(v, &mut lanes);
        // Seed at 0 (the Smith-Waterman clamp), matching `_mmxxx_max_value` (`impl:240-250`).
        lanes.iter().fold(0i16, |acc, &x| acc.max(x))
    }

    /// Hand-unrolled 3-step ladder with byte-shifts `[2, 4, 8]` (`impl:182-184`). Each step is the
    /// shared [`Simd::prefix_max_step`] with that step's literal shift constant.
    #[inline(always)]
    fn prefix_max(v: __m128i, penalties: &[__m128i], masks: &[__m128i]) -> __m128i {
        debug_assert!(penalties.len() >= Self::LOG_LANES as usize);
        debug_assert!(masks.len() >= Self::LOG_LANES as usize);
        let mut a = v;
        a = Self::prefix_max_step::<2>(a, penalties[0], masks[0]);
        a = Self::prefix_max_step::<4>(a, penalties[1], masks[1]);
        a = Self::prefix_max_step::<8>(a, penalties[2], masks[2]);
        a
    }
}

// ---- Sse41I32: 4 × i32 -----------------------------------------------------------------------

/// SSE4.1 backend over 4 packed `i32` lanes (`impl:188-220`).
pub(crate) struct Sse41I32;

/// Lane-typed `i32` add. Ports `_mm_add_epi32` (`impl:198`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn add32(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi32(a, b)
}

/// Lane-typed `i32` sub. Ports `_mm_sub_epi32` (`impl:201`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn sub32(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi32(a, b)
}

/// The saturation target for an overflowing lane, keyed on `a`'s sign: `a >= 0 -> i32::MAX`,
/// `a < 0 -> i32::MIN`. Shared by [`adds32`]/[`subs32`] — for both signed add and sub, the
/// direction of overflow always follows the operand `a`'s sign (see each caller's doc for why).
/// Computed as `(a >>> 31) + i32::MAX`: the logical shift isolates `a`'s sign bit as a plain `0`
/// or `1`, and adding `i32::MAX` (`0x7FFF_FFFF`) either leaves it at `i32::MAX` (bit `0`) or wraps
/// it to `i32::MIN` (`0x7FFF_FFFF + 1 == 0x8000_0000`, bit `1`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn saturation_target32(a: __m128i) -> __m128i {
    _mm_add_epi32(_mm_srli_epi32::<31>(a), _mm_set1_epi32(i32::MAX))
}

/// Lane-typed `i32` **saturating** add: SSE4.1 has no `_mm_adds_epi32` (only int16 saturates
/// natively — see [`adds16`]), so this emulates it. Computes the wrapping sum `r`, detects
/// overflow via the classic signed-overflow test `(a ^ r) & (b ^ r) < 0` (overflow iff `a` and `b`
/// share a sign that differs from `r`'s), and blends in [`saturation_target32`] on overflow.
///
/// `_mm_blendv_epi8` selects **per byte** off each byte's own high bit, but the overflow test above
/// only produces a meaningful signal in bit 31 of each 32-bit lane — the other 3 bytes' high bits
/// are unrelated noise. `_mm_srai_epi32::<31>` broadcasts that one sign bit across the whole lane
/// (`0xFFFF_FFFF` on overflow, `0x0000_0000` otherwise) so every byte within a lane agrees on which
/// operand to select; skipping this step corrupts ~2/3 of overflowing lanes (verified against a
/// `saturating_add` reference).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn adds32(a: __m128i, b: __m128i) -> __m128i {
    let r = _mm_add_epi32(a, b);
    let sat = saturation_target32(a);
    let overflow = _mm_and_si128(_mm_xor_si128(a, r), _mm_xor_si128(b, r));
    let overflow_mask = _mm_srai_epi32::<31>(overflow);
    _mm_blendv_epi8(r, sat, overflow_mask)
}

/// Lane-typed `i32` **saturating** sub; see [`adds32`] for the emulation rationale (no native
/// `_mm_subs_epi32`). Overflow test is the signed-subtraction form `(a ^ b) & (a ^ r) < 0`
/// (overflow iff `a` and `b` differ in sign and `r` differs from `a`); the saturation target still
/// follows `a`'s sign (overflowing subtraction always overshoots in the direction `a` already
/// pointed).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn subs32(a: __m128i, b: __m128i) -> __m128i {
    let r = _mm_sub_epi32(a, b);
    let sat = saturation_target32(a);
    let overflow = _mm_and_si128(_mm_xor_si128(a, b), _mm_xor_si128(a, r));
    let overflow_mask = _mm_srai_epi32::<31>(overflow);
    _mm_blendv_epi8(r, sat, overflow_mask)
}

/// Lane-typed `i32` signed min. Ports `_mm_min_epi32` (`impl:204`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
// Test-only trait op; see the note on `min16` above.
#[allow(dead_code)]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn min32(a: __m128i, b: __m128i) -> __m128i {
    _mm_min_epi32(a, b)
}

/// Lane-typed `i32` signed max. Ports `_mm_max_epi32` (`impl:207`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn max32(a: __m128i, b: __m128i) -> __m128i {
    _mm_max_epi32(a, b)
}

/// Broadcast `value` into all 4 `i32` lanes. Ports `_mm_set1_epi32` (`impl:210`).
///
/// # Safety
/// Caller must guarantee SSE4.1 is available (see the module-level Safety note).
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn set1_32(value: i32) -> __m128i {
    _mm_set1_epi32(value)
}

impl Simd for Sse41I32 {
    type Elem = i32;
    type Vec = __m128i;

    const LANES: usize = 4;
    const LOG_LANES: u32 = 2;
    const LSS: i32 = size_of::<i32>() as i32; // 4
    const RSS: i32 = 16 - size_of::<i32>() as i32; // 12
    const NEG_INF: i32 = NEG_INF_I32;

    #[inline(always)]
    fn splat(value: i32) -> __m128i {
        // SAFETY: only reached after `is_x86_feature_detected!("sse4.1")` (see module Safety note).
        unsafe { set1_32(value) }
    }

    #[inline(always)]
    fn add(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Non-saturating `_mm_add_epi32` per the plan's Global Constraints.
        unsafe { add32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Non-saturating `_mm_sub_epi32`.
        unsafe { sub32(a, b) }
    }

    #[inline(always)]
    fn adds(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Emulated saturating add (no native `_mm_adds_epi32`).
        unsafe { adds32(a, b) }
    }

    #[inline(always)]
    fn subs(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`. Emulated saturating sub (no native `_mm_subs_epi32`).
        unsafe { subs32(a, b) }
    }

    #[inline(always)]
    fn min(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { min32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { max32(a, b) }
    }

    #[inline(always)]
    fn or(a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { or_si(a, b) }
    }

    #[inline(always)]
    fn loadu(src: &[i32]) -> __m128i {
        debug_assert!(src.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` (and the trait's documented precondition)
        // guarantee `src` covers the 16 bytes read.
        unsafe { loadu(src.as_ptr().cast::<u8>()) }
    }

    #[inline(always)]
    fn storeu(v: __m128i, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `loadu`, mirrored for the 16-byte write.
        unsafe { storeu(dst.as_mut_ptr().cast::<u8>(), v) }
    }

    #[inline(always)]
    fn store_widened_i32(v: __m128i, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // Elem is already `i32`; the "widen" is a plain 16-byte store.
        // SAFETY: see `loadu`, mirrored for the 16-byte write.
        unsafe { storeu(dst.as_mut_ptr().cast::<u8>(), v) }
    }

    #[inline(always)]
    fn slli<const N: i32>(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<N>(v) }
    }

    #[inline(always)]
    fn srli<const N: i32>(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<N>(v) }
    }

    /// Diagonal shift by `LSS = 4` bytes (one `i32` lane), the literal for this width.
    #[inline(always)]
    fn slli_one_lane(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<4>(v) }
    }

    /// Carry shift by `RSS = 12` bytes (isolate lane 3 into lane 0), the literal for this width.
    #[inline(always)]
    fn srli_top_lane(v: __m128i) -> __m128i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<12>(v) }
    }

    #[inline(always)]
    fn horizontal_max(v: __m128i) -> i32 {
        let mut lanes = [0i32; 4];
        Self::storeu(v, &mut lanes);
        // Seed at 0 (the Smith-Waterman clamp), matching `_mmxxx_max_value` (`impl:240-250`).
        lanes.iter().fold(0i32, |acc, &x| acc.max(x))
    }

    /// Hand-unrolled 2-step ladder with byte-shifts `[4, 8]` (`impl:217-218`). Each step is the
    /// shared [`Simd::prefix_max_step`] with that step's literal shift constant.
    #[inline(always)]
    fn prefix_max(v: __m128i, penalties: &[__m128i], masks: &[__m128i]) -> __m128i {
        debug_assert!(penalties.len() >= Self::LOG_LANES as usize);
        debug_assert!(masks.len() >= Self::LOG_LANES as usize);
        let mut a = v;
        a = Self::prefix_max_step::<4>(a, penalties[0], masks[0]);
        a = Self::prefix_max_step::<8>(a, penalties[1], masks[1]);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::simd::profile::{build_masks, build_penalties};

    /// Extracts an `Sse41I16` register's 8 lanes into a plain array for comparison.
    fn unpack16(v: __m128i) -> [i16; 8] {
        let mut out = [0i16; 8];
        Sse41I16::storeu(v, &mut out);
        out
    }

    /// Extracts an `Sse41I32` register's 4 lanes into a plain array for comparison.
    fn unpack32(v: __m128i) -> [i32; 4] {
        let mut out = [0i32; 4];
        Sse41I32::storeu(v, &mut out);
        out
    }

    /// Independent scalar reference for the prefix-max recurrence: lane `j` is the max over all
    /// `k <= j` of `a[k] + (j - k) * penalty` (computed in `i32` to avoid intermediate overflow).
    /// This is the closed form the shift-and-max ladder computes — deriving it *without* any
    /// intrinsic is exactly what makes it a valid oracle for the SSE ladder's shift constants.
    fn scalar_prefix_max(a: &[i32], penalty: i32) -> Vec<i32> {
        a.iter()
            .enumerate()
            .map(|(j, _)| {
                a.iter()
                    .take(j + 1)
                    .enumerate()
                    .map(|(k, &ak)| ak + (j - k) as i32 * penalty)
                    .max()
                    .unwrap()
            })
            .collect()
    }

    /// Reference byte-shift-left (toward higher lane indices) over 8 `i16` lanes; `nbytes` must be
    /// a multiple of 2 (one `i16` = 2 bytes), so the shift is by `nbytes / 2` whole lanes with
    /// zero fill in the vacated low lanes — matching `_mm_slli_si128`'s semantics at lane
    /// granularity.
    fn shift_left_i16(a: &[i16; 8], nbytes: usize) -> [i16; 8] {
        let sh = nbytes / 2;
        let mut out = [0i16; 8];
        for (i, &x) in a.iter().enumerate() {
            if i + sh < 8 {
                out[i + sh] = x;
            }
        }
        out
    }

    /// Reference byte-shift-right (toward lower lane indices) over 8 `i16` lanes; see
    /// [`shift_left_i16`].
    fn shift_right_i16(a: &[i16; 8], nbytes: usize) -> [i16; 8] {
        let sh = nbytes / 2;
        let mut out = [0i16; 8];
        for (i, &x) in a.iter().enumerate() {
            if i >= sh {
                out[i - sh] = x;
            }
        }
        out
    }

    /// Reference byte-shift-left over 4 `i32` lanes; `nbytes` a multiple of 4.
    fn shift_left_i32(a: &[i32; 4], nbytes: usize) -> [i32; 4] {
        let sh = nbytes / 4;
        let mut out = [0i32; 4];
        for (i, &x) in a.iter().enumerate() {
            if i + sh < 4 {
                out[i + sh] = x;
            }
        }
        out
    }

    /// Reference byte-shift-right over 4 `i32` lanes; see [`shift_left_i32`].
    fn shift_right_i32(a: &[i32; 4], nbytes: usize) -> [i32; 4] {
        let sh = nbytes / 4;
        let mut out = [0i32; 4];
        for (i, &x) in a.iter().enumerate() {
            if i >= sh {
                out[i - sh] = x;
            }
        }
        out
    }

    // ---- int16 ------------------------------------------------------------------------------

    #[test]
    fn i16_ops_match_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let a = [3i16, -2, 5, 1, 0, 7, -4, 2];
        let b = [1i16, 1, -3, 4, -5, 2, 6, -1];
        let va = Sse41I16::loadu(&a);
        let vb = Sse41I16::loadu(&b);

        // splat
        assert_eq!(unpack16(Sse41I16::splat(-9)), [-9i16; 8]);

        // add / sub / min / max / or, lane-wise vs plain integer ops.
        let mut exp_add = [0i16; 8];
        let mut exp_sub = [0i16; 8];
        let mut exp_min = [0i16; 8];
        let mut exp_max = [0i16; 8];
        let mut exp_or = [0i16; 8];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_add[i] = ai.wrapping_add(bi);
            exp_sub[i] = ai.wrapping_sub(bi);
            exp_min[i] = ai.min(bi);
            exp_max[i] = ai.max(bi);
            exp_or[i] = ai | bi;
        }
        assert_eq!(unpack16(Sse41I16::add(va, vb)), exp_add);
        assert_eq!(unpack16(Sse41I16::sub(va, vb)), exp_sub);
        assert_eq!(unpack16(Sse41I16::min(va, vb)), exp_min);
        assert_eq!(unpack16(Sse41I16::max(va, vb)), exp_max);
        assert_eq!(unpack16(Sse41I16::or(va, vb)), exp_or);
    }

    #[test]
    fn i16_add_sub_are_non_saturating() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        // i16::MAX + 1 wraps to i16::MIN (modular `_mm_add_epi16`, NOT saturating `_mm_adds_epi16`).
        let hi = Sse41I16::splat(i16::MAX);
        let lo = Sse41I16::splat(i16::MIN);
        let one = Sse41I16::splat(1);
        assert_eq!(unpack16(Sse41I16::add(hi, one)), [i16::MIN; 8]);
        assert_eq!(unpack16(Sse41I16::sub(lo, one)), [i16::MAX; 8]);
    }

    #[test]
    fn i16_adds_subs_saturate_at_bounds() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        // adds/subs must clamp (native `_mm_adds_epi16`/`_mm_subs_epi16`), unlike add/sub, which
        // wrap.
        let hi = Sse41I16::splat(i16::MAX);
        let lo = Sse41I16::splat(i16::MIN);
        let one = Sse41I16::splat(1);
        assert_eq!(unpack16(Sse41I16::adds(hi, one)), [i16::MAX; 8]);
        assert_eq!(unpack16(Sse41I16::subs(lo, one)), [i16::MIN; 8]);

        let a = [
            3i16,
            i16::MAX,
            -2,
            i16::MIN,
            0,
            i16::MAX - 1,
            -100,
            i16::MIN + 1,
        ];
        let b = [1i16, 1, -3, -1, -5, 2, -50, -2];
        let va = Sse41I16::loadu(&a);
        let vb = Sse41I16::loadu(&b);
        let mut exp_adds = [0i16; 8];
        let mut exp_subs = [0i16; 8];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_adds[i] = ai.saturating_add(bi);
            exp_subs[i] = ai.saturating_sub(bi);
        }
        assert_eq!(unpack16(Sse41I16::adds(va, vb)), exp_adds);
        assert_eq!(unpack16(Sse41I16::subs(va, vb)), exp_subs);
    }

    #[test]
    fn i16_loadu_storeu_round_trip() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let src = [10i16, 20, 30, 40, 50, 60, 70, 80];
        let v = Sse41I16::loadu(&src);
        let mut dst = [0i16; 8];
        Sse41I16::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_store_widened_i32_sign_extends_all_lanes() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let src = [-5i16, 2, -9, 11, i16::MIN, i16::MAX, 0, -1];
        let v = Sse41I16::loadu(&src);
        let mut dst = [0i32; 8];
        Sse41I16::store_widened_i32(v, &mut dst);
        let expected: [i32; 8] = std::array::from_fn(|k| i32::from(src[k]));
        assert_eq!(dst, expected);
    }

    #[test]
    fn i32_store_widened_i32_is_a_plain_store() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let src = [-5i32, 123_456, i32::MIN, i32::MAX];
        let v = Sse41I32::loadu(&src);
        let mut dst = [0i32; 4];
        Sse41I32::store_widened_i32(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_slli_srli_have_byte_shift_semantics() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let a = [1i16, 2, 3, 4, 5, 6, 7, 8];
        let v = Sse41I16::loadu(&a);
        assert_eq!(unpack16(Sse41I16::slli::<2>(v)), shift_left_i16(&a, 2));
        assert_eq!(unpack16(Sse41I16::slli::<4>(v)), shift_left_i16(&a, 4));
        assert_eq!(unpack16(Sse41I16::slli::<8>(v)), shift_left_i16(&a, 8));
        assert_eq!(unpack16(Sse41I16::srli::<2>(v)), shift_right_i16(&a, 2));
        // RSS = 14: isolates the single highest lane down to lane 0.
        assert_eq!(unpack16(Sse41I16::srli::<14>(v)), shift_right_i16(&a, 14));
    }

    #[test]
    fn i16_horizontal_max_seeds_at_zero() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        // All-negative reduces to 0 (SW clamp), not the largest (least-negative) lane.
        assert_eq!(Sse41I16::horizontal_max(Sse41I16::splat(-5)), 0);
        let mixed = Sse41I16::loadu(&[-5i16, -2, -9, -1, -3, -8, -7, -6]);
        assert_eq!(Sse41I16::horizontal_max(mixed), 0);
        let positive = Sse41I16::loadu(&[-5i16, 2, -9, 11, -3, 8, -7, 6]);
        assert_eq!(Sse41I16::horizontal_max(positive), 11);
    }

    #[test]
    fn i16_prefix_max_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let penalty: i16 = -4;
        let penalties = build_penalties::<Sse41I16>(penalty);
        let masks = build_masks::<Sse41I16>(Sse41I16::NEG_INF);

        for a in [
            [3i16, -2, 5, 1, 0, 7, -4, 2],
            [0i16, 0, 0, 0, 0, 0, 0, 0],
            [8i16, 6, 4, 2, 0, -2, -4, -6],
            [-1i16, 9, -3, 2, 12, -8, 4, 5],
            // Dominant lane 0 forces the *full* ladder: lane 7's winner is `a[0] - 7*4 = 12`
            // (distance 7), reachable only via all three byte-shifts `[2, 4, 8]` — a wrong final
            // shift constant (e.g. 4 instead of 8) leaves lane 7 at a nearer, smaller value.
            [40i16, 1, 2, 3, 4, 5, 6, 7],
            // Dominant lane 1 forces distance-6 propagation into lane 7 (`a[1] - 6*4 = 12`).
            [0i16, 36, 1, 2, 3, 4, 5, 6],
        ] {
            let v = Sse41I16::loadu(&a);
            let got = unpack16(Sse41I16::prefix_max(v, &penalties, &masks));
            let a_i32: Vec<i32> = a.iter().map(|&x| i32::from(x)).collect();
            let expected: Vec<i16> = scalar_prefix_max(&a_i32, i32::from(penalty))
                .into_iter()
                .map(|x| x as i16)
                .collect();
            assert_eq!(got.to_vec(), expected, "prefix_max i16 mismatch for {a:?}");
        }
    }

    // ---- int32 ------------------------------------------------------------------------------

    #[test]
    fn i32_ops_match_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let a = [3i32, -2, 500, -1];
        let b = [1i32, 1, -300, 4];
        let va = Sse41I32::loadu(&a);
        let vb = Sse41I32::loadu(&b);

        assert_eq!(unpack32(Sse41I32::splat(-9)), [-9i32; 4]);

        let mut exp_add = [0i32; 4];
        let mut exp_sub = [0i32; 4];
        let mut exp_min = [0i32; 4];
        let mut exp_max = [0i32; 4];
        let mut exp_or = [0i32; 4];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_add[i] = ai.wrapping_add(bi);
            exp_sub[i] = ai.wrapping_sub(bi);
            exp_min[i] = ai.min(bi);
            exp_max[i] = ai.max(bi);
            exp_or[i] = ai | bi;
        }
        assert_eq!(unpack32(Sse41I32::add(va, vb)), exp_add);
        assert_eq!(unpack32(Sse41I32::sub(va, vb)), exp_sub);
        assert_eq!(unpack32(Sse41I32::min(va, vb)), exp_min);
        assert_eq!(unpack32(Sse41I32::max(va, vb)), exp_max);
        assert_eq!(unpack32(Sse41I32::or(va, vb)), exp_or);
    }

    #[test]
    fn i32_add_sub_are_non_saturating() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let hi = Sse41I32::splat(i32::MAX);
        let lo = Sse41I32::splat(i32::MIN);
        let one = Sse41I32::splat(1);
        assert_eq!(unpack32(Sse41I32::add(hi, one)), [i32::MIN; 4]);
        assert_eq!(unpack32(Sse41I32::sub(lo, one)), [i32::MAX; 4]);
    }

    /// SSE4.1 has no native `_mm_adds_epi32`/`_mm_subs_epi32` (see [`super::adds32`]/
    /// [`super::subs32`]), so this is the one test in the crate that actually exercises the x86
    /// int32 saturation *emulation* against the scalar `saturating_add`/`saturating_sub` oracle —
    /// boundary values plus a wide pseudo-random sweep, since a hand-derived compare/blend formula
    /// is exactly the kind of thing that looks right but is subtly wrong for some inputs (as the
    /// unbroadcast-sign-bit version of this formula was during development).
    #[test]
    fn i32_adds_subs_saturate_at_bounds() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let hi = Sse41I32::splat(i32::MAX);
        let lo = Sse41I32::splat(i32::MIN);
        let one = Sse41I32::splat(1);
        assert_eq!(unpack32(Sse41I32::adds(hi, one)), [i32::MAX; 4]);
        assert_eq!(unpack32(Sse41I32::subs(lo, one)), [i32::MIN; 4]);

        let neg = Sse41I32::NEG_INF;
        let a = [3i32, i32::MAX, neg, -100];
        let b = [1i32, 1, -128, -50];
        let va = Sse41I32::loadu(&a);
        let vb = Sse41I32::loadu(&b);
        let mut exp_adds = [0i32; 4];
        let mut exp_subs = [0i32; 4];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_adds[i] = ai.saturating_add(bi);
            exp_subs[i] = ai.saturating_sub(bi);
        }
        assert_eq!(unpack32(Sse41I32::adds(va, vb)), exp_adds);
        assert_eq!(unpack32(Sse41I32::subs(va, vb)), exp_subs);

        // Pseudo-random sweep (xorshift, no external RNG dependency) covering the full `i32`
        // range, including many overflow/underflow-triggering combinations.
        let mut state: u32 = 0x9E37_79B9;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i32
        };
        for _ in 0..10_000 {
            let a = [next(), next(), next(), next()];
            let b = [next(), next(), next(), next()];
            let va = Sse41I32::loadu(&a);
            let vb = Sse41I32::loadu(&b);
            let exp_adds: [i32; 4] = std::array::from_fn(|i| a[i].saturating_add(b[i]));
            let exp_subs: [i32; 4] = std::array::from_fn(|i| a[i].saturating_sub(b[i]));
            assert_eq!(
                unpack32(Sse41I32::adds(va, vb)),
                exp_adds,
                "adds mismatch for {a:?} + {b:?}"
            );
            assert_eq!(
                unpack32(Sse41I32::subs(va, vb)),
                exp_subs,
                "subs mismatch for {a:?} - {b:?}"
            );
        }
    }

    #[test]
    fn i32_loadu_storeu_round_trip() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let src = [100i32, 200, 300, 400];
        let v = Sse41I32::loadu(&src);
        let mut dst = [0i32; 4];
        Sse41I32::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i32_slli_srli_have_byte_shift_semantics() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let a = [11i32, 22, 33, 44];
        let v = Sse41I32::loadu(&a);
        assert_eq!(unpack32(Sse41I32::slli::<4>(v)), shift_left_i32(&a, 4));
        assert_eq!(unpack32(Sse41I32::slli::<8>(v)), shift_left_i32(&a, 8));
        assert_eq!(unpack32(Sse41I32::srli::<4>(v)), shift_right_i32(&a, 4));
        // RSS = 12: isolates the single highest lane down to lane 0.
        assert_eq!(unpack32(Sse41I32::srli::<12>(v)), shift_right_i32(&a, 12));
    }

    #[test]
    fn i32_horizontal_max_seeds_at_zero() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        assert_eq!(Sse41I32::horizontal_max(Sse41I32::splat(-5)), 0);
        let mixed = Sse41I32::loadu(&[-5i32, -2, -9, -1]);
        assert_eq!(Sse41I32::horizontal_max(mixed), 0);
        let positive = Sse41I32::loadu(&[-5i32, 42, -9, 11]);
        assert_eq!(Sse41I32::horizontal_max(positive), 42);
    }

    #[test]
    fn i32_prefix_max_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let penalty: i32 = -6;
        let penalties = build_penalties::<Sse41I32>(penalty);
        let masks = build_masks::<Sse41I32>(Sse41I32::NEG_INF);

        for a in [
            [3i32, -2, 5, 1],
            [0i32, 0, 0, 0],
            [8i32, 4, 0, -4],
            [-1i32, 20, -3, 2],
            // Dominant lane 0 forces the full 2-step ladder: lane 3's winner is `a[0] - 3*6 = 12`
            // (distance 3), reachable only via both byte-shifts `[4, 8]`; a wrong second shift
            // leaves lane 3 at a nearer, smaller value.
            [30i32, 1, 2, 3],
        ] {
            let v = Sse41I32::loadu(&a);
            let got = unpack32(Sse41I32::prefix_max(v, &penalties, &masks)).to_vec();
            let expected = scalar_prefix_max(&a, penalty);
            assert_eq!(got, expected, "prefix_max i32 mismatch for {a:?}");
        }
    }

    #[test]
    fn lane_constants_match_upstream() {
        assert_eq!(Sse41I16::LANES, 8);
        assert_eq!(Sse41I16::LOG_LANES, 3);
        assert_eq!(Sse41I16::LSS, 2);
        assert_eq!(Sse41I16::RSS, 14);
        assert_eq!(Sse41I16::NEG_INF, i16::MIN + 1024);
        assert_eq!(Sse41I32::LANES, 4);
        assert_eq!(Sse41I32::LOG_LANES, 2);
        assert_eq!(Sse41I32::LSS, 4);
        assert_eq!(Sse41I32::RSS, 12);
        assert_eq!(Sse41I32::NEG_INF, i32::MIN + 1024);
    }
}
