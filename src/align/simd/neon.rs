//! NEON (aarch64) [`Simd`] backends: `NeonI16` (8×`i16`) and `NeonI32` (4×`i32`), the native
//! aarch64 vectorized implementation of the [`Simd`] trait.
//!
//! Upstream `spoa` has NO hand-written NEON path — it reaches NEON only by having SIMDe translate
//! its x86 intrinsics. This module is therefore a genuinely new hand-port whose oracle is the
//! proven [`super::sse41`] SSE4.1 backend (same `LANES`/`LOG_LANES`/`LSS`/`RSS`/`NEG_INF`
//! constants) and, one level down, the scalar [`crate::align::SisdEngine`]. `NeonI16`/`NeonI32`
//! are the *structural twins* of [`super::sse41::Sse41I16`]/[`super::sse41::Sse41I32`]: identical
//! lane geometry, differing ONLY in the intrinsic names and — the one genuinely new piece — the
//! byte-shift emulation.
//!
//! # Byte-shift emulation (the one place a NEON port silently breaks)
//!
//! x86's `_mm_slli_si128`/`_mm_srli_si128` shift the whole 128-bit register by a *byte* count.
//! NEON has no whole-register byte shift; the equivalent is `vextq_s8`, which concatenates two
//! registers and extracts a 16-byte window starting at an **ELEMENT** (here, byte) offset:
//! `vextq_s8(a, b, n)[i] = concat(a, b)[n + i]`. So, over `int8x16_t`:
//! - `_mm_slli_si128::<N>(v)` (shift left N bytes, zero-fill low) ≡ `vextq_s8::<16 - N>(zero, v)`.
//! - `_mm_srli_si128::<N>(v)` (shift right N bytes, zero-fill high) ≡ `vextq_s8::<N>(v, zero)`.
//!
//! `vextq_s8`'s immediate must be a compile-time constant in `0..=15`, and Rust (stable) cannot
//! feed it `16 - N` computed from a const-generic `N` (that needs `generic_const_exprs`). The
//! [`byte_shift_left`]/[`byte_shift_right`] helpers therefore `match N` and dispatch to a literal
//! `vextq_s8::<K>` per arm — every literal `K` (and every `16 - K`) is a plain integer constant,
//! so no unstable feature is needed. The lane-typed `slli`/`srli` reinterpret their `int16x8_t`/
//! `int32x4_t` to `int8x16_t`, byte-shift, and reinterpret back — matching `_mm_slli_si128`'s
//! whole-register (lane-width-agnostic) semantics exactly.
//!
//! # Safety
//!
//! Every intrinsic-calling helper is a `#[target_feature(enable = "neon")] unsafe fn`. On aarch64
//! NEON is architecturally baseline (always present), but each [`Simd`] method still wraps its
//! helper in an `unsafe` block whose precondition is *"reached only after
//! `is_aarch64_feature_detected!(\"neon\")` returned true"* — exactly how [`super`]'s runtime
//! dispatch selects these backends and how every test below is gated. All `unsafe` in the crate is
//! confined to `src/align/simd/` via the module's `#![allow(unsafe_code)]`.

use super::lanes::Simd;
use core::arch::aarch64::{
    int16x8_t, int32x4_t, int8x16_t, vaddq_s16, vaddq_s32, vdupq_n_s16, vdupq_n_s32, vdupq_n_s8,
    vextq_s8, vget_high_s16, vget_low_s16, vld1q_s16, vld1q_s32, vmaxq_s16, vmaxq_s32, vminq_s16,
    vminq_s32, vmovl_s16, vorrq_s16, vorrq_s32, vqaddq_s16, vqaddq_s32, vqsubq_s16, vqsubq_s32,
    vreinterpretq_s16_s8, vreinterpretq_s32_s8, vreinterpretq_s8_s16, vreinterpretq_s8_s32,
    vst1q_s16, vst1q_s32, vsubq_s16, vsubq_s32,
};

/// `i16::MIN + 1024`, this backend's `kNegativeInfinity` (see [`Simd::NEG_INF`]).
const NEG_INF_I16: i16 = i16::MIN + 1024;
/// `i32::MIN + 1024`, this backend's `kNegativeInfinity` (see [`Simd::NEG_INF`]).
const NEG_INF_I32: i32 = i32::MIN + 1024;

// ---- shared, lane-width-agnostic byte-shift helpers over `int8x16_t` --------------------------
//
// `_mm_slli_si128`/`_mm_srli_si128` (and NEON's `vextq_s8` emulation of them) operate on the whole
// 128-bit register regardless of lane width, so both `NeonI16` and `NeonI32` reinterpret to
// `int8x16_t` and share these two helpers (mirroring `sse41.rs`'s shared `slli_si`/`srli_si`).

/// Whole-register byte-shift-**left** by the compile-time constant `N` bytes, zero-filling the
/// vacated low-order bytes — the NEON emulation of `_mm_slli_si128::<N>` via
/// `vextq_s8::<16 - N>(zero, v)` (see the module-level "Byte-shift emulation" note).
///
/// `N` is matched to a literal `vextq_s8` immediate because that immediate must be a compile-time
/// constant in `0..=15` and stable Rust cannot compute `16 - N` from the const-generic `N`.
/// `N == 0` is the identity; `N == 16` is all-zero; both sit outside `vextq_s8`'s `0..=15` window,
/// so they are handled directly. Only `N` values the ladders/tests actually use are ever passed
/// (`0..=16`), but every arm is a valid `vextq_s8` immediate, so the full range compiles.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn byte_shift_left<const N: i32>(v: int8x16_t) -> int8x16_t {
    let z = vdupq_n_s8(0);
    match N {
        0 => v,
        1 => vextq_s8::<15>(z, v),
        2 => vextq_s8::<14>(z, v),
        3 => vextq_s8::<13>(z, v),
        4 => vextq_s8::<12>(z, v),
        5 => vextq_s8::<11>(z, v),
        6 => vextq_s8::<10>(z, v),
        7 => vextq_s8::<9>(z, v),
        8 => vextq_s8::<8>(z, v),
        9 => vextq_s8::<7>(z, v),
        10 => vextq_s8::<6>(z, v),
        11 => vextq_s8::<5>(z, v),
        12 => vextq_s8::<4>(z, v),
        13 => vextq_s8::<3>(z, v),
        14 => vextq_s8::<2>(z, v),
        15 => vextq_s8::<1>(z, v),
        16 => z,
        _ => unreachable!("byte_shift_left N out of range 0..=16"),
    }
}

/// Whole-register byte-shift-**right** by the compile-time constant `N` bytes, zero-filling the
/// vacated high-order bytes — the NEON emulation of `_mm_srli_si128::<N>` via
/// `vextq_s8::<N>(v, zero)` (see the module-level "Byte-shift emulation" note). See
/// [`byte_shift_left`] on the `match N` / `N == 0` / `N == 16` handling.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn byte_shift_right<const N: i32>(v: int8x16_t) -> int8x16_t {
    let z = vdupq_n_s8(0);
    match N {
        0 => v,
        1 => vextq_s8::<1>(v, z),
        2 => vextq_s8::<2>(v, z),
        3 => vextq_s8::<3>(v, z),
        4 => vextq_s8::<4>(v, z),
        5 => vextq_s8::<5>(v, z),
        6 => vextq_s8::<6>(v, z),
        7 => vextq_s8::<7>(v, z),
        8 => vextq_s8::<8>(v, z),
        9 => vextq_s8::<9>(v, z),
        10 => vextq_s8::<10>(v, z),
        11 => vextq_s8::<11>(v, z),
        12 => vextq_s8::<12>(v, z),
        13 => vextq_s8::<13>(v, z),
        14 => vextq_s8::<14>(v, z),
        15 => vextq_s8::<15>(v, z),
        16 => z,
        _ => unreachable!("byte_shift_right N out of range 0..=16"),
    }
}

// ---- NeonI16: 8 × i16 -------------------------------------------------------------------------

/// NEON backend over 8 packed `i16` lanes — the structural twin of [`super::sse41::Sse41I16`].
/// Constructed only through `SimdEngine`'s aarch64 NEON dispatch.
pub(crate) struct NeonI16;

/// Lane-typed `i16` add. Ports `_mm_add_epi16` (`impl:163`) as NEON `vaddq_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vaddq_s16(a, b)
}

/// Lane-typed `i16` sub. Ports `_mm_sub_epi16` (`impl:166`) as NEON `vsubq_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sub16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vsubq_s16(a, b)
}

/// Lane-typed `i16` **saturating** add. Ports `_mm_adds_epi16` as NEON `vqaddq_s16` — native
/// saturating hardware support, unlike the int32 backends (see [`adds32`]).
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn adds16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vqaddq_s16(a, b)
}

/// Lane-typed `i16` **saturating** sub. Ports `_mm_subs_epi16` as NEON `vqsubq_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn subs16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vqsubq_s16(a, b)
}

/// Lane-typed `i16` signed min. Ports `_mm_min_epi16` (`impl:169`) as NEON `vminq_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
// The `min` trait op is test-only: the DP fill maximizes score (so it uses `max`, never `min`),
// so this faithful-port helper is dead in non-test builds.
#[allow(dead_code)]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn min16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vminq_s16(a, b)
}

/// Lane-typed `i16` signed max. Ports `_mm_max_epi16` (`impl:172`) as NEON `vmaxq_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn max16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vmaxq_s16(a, b)
}

/// Bitwise OR of two `i16` registers. Ports `_mm_or_si128` (`impl:143`) as NEON `vorrq_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn or16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vorrq_s16(a, b)
}

/// Broadcast `value` into all 8 `i16` lanes. Ports `_mm_set1_epi16` (`impl:175`) as `vdupq_n_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn set1_16(value: i16) -> int16x8_t {
    vdupq_n_s16(value)
}

/// Unaligned load of 8 `i16` from `src`. Ports `_mm_loadu_si128` (`impl:135`) as NEON `vld1q_s16`
/// (NEON loads are unaligned-capable).
///
/// # Safety
/// Caller must guarantee NEON is available AND that `src` points to at least 8 readable `i16`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn loadu16(src: *const i16) -> int16x8_t {
    vld1q_s16(src)
}

/// Unaligned store of 8 `i16` to `dst`. Ports `_mm_storeu_si128` (`impl:139`) as NEON `vst1q_s16`.
///
/// # Safety
/// Caller must guarantee NEON is available AND that `dst` points to at least 8 writable `i16`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn storeu16(dst: *mut i16, v: int16x8_t) {
    vst1q_s16(dst, v);
}

/// Sign-extends the 8 packed `i16` lanes of `v` to `i32` and stores them contiguously to the 8
/// `i32` slots at `dst`. Widens each half with `vmovl_s16` (`vget_low_s16`/`vget_high_s16`) and
/// stores the two resulting 4×`i32` vectors with `vst1q_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available AND that `dst` points to at least 8 writable `i32`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn store_widen_i16(dst: *mut i32, v: int16x8_t) {
    vst1q_s32(dst, vmovl_s16(vget_low_s16(v)));
    vst1q_s32(dst.add(4), vmovl_s16(vget_high_s16(v)));
}

/// Byte-shift-left of an `i16` register by `N` bytes (via the shared [`byte_shift_left`] over an
/// `int8x16_t` reinterpret).
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn slli16<const N: i32>(v: int16x8_t) -> int16x8_t {
    vreinterpretq_s16_s8(byte_shift_left::<N>(vreinterpretq_s8_s16(v)))
}

/// Byte-shift-right of an `i16` register by `N` bytes (via the shared [`byte_shift_right`]).
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn srli16<const N: i32>(v: int16x8_t) -> int16x8_t {
    vreinterpretq_s16_s8(byte_shift_right::<N>(vreinterpretq_s8_s16(v)))
}

impl Simd for NeonI16 {
    type Elem = i16;
    type Vec = int16x8_t;

    const LANES: usize = 8;
    const LOG_LANES: u32 = 3;
    const LSS: i32 = size_of::<i16>() as i32; // 2
    const RSS: i32 = 16 - size_of::<i16>() as i32; // 14
    const NEG_INF: i16 = NEG_INF_I16;

    #[inline(always)]
    fn splat(value: i16) -> int16x8_t {
        // SAFETY: only reached after `is_aarch64_feature_detected!("neon")` (see module Safety note).
        unsafe { set1_16(value) }
    }

    #[inline(always)]
    fn add(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`. Non-saturating `vaddq_s16` (NOT `vqaddq_s16`) per Global Constraints.
        unsafe { add16(a, b) }
    }

    #[inline(always)]
    fn sub(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`. Non-saturating `vsubq_s16`.
        unsafe { sub16(a, b) }
    }

    #[inline(always)]
    fn adds(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`. Native saturating `vqaddq_s16`.
        unsafe { adds16(a, b) }
    }

    #[inline(always)]
    fn subs(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`. Native saturating `vqsubq_s16`.
        unsafe { subs16(a, b) }
    }

    #[inline(always)]
    fn min(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { min16(a, b) }
    }

    #[inline(always)]
    fn max(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { max16(a, b) }
    }

    #[inline(always)]
    fn or(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { or16(a, b) }
    }

    #[inline(always)]
    fn loadu(src: &[i16]) -> int16x8_t {
        debug_assert!(src.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` (and the trait's documented precondition)
        // guarantee `src` covers the 8 `i16` read.
        unsafe { loadu16(src.as_ptr()) }
    }

    #[inline(always)]
    fn storeu(v: int16x8_t, dst: &mut [i16]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `loadu`, mirrored for the 8-`i16` write.
        unsafe { storeu16(dst.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn store_widened_i32(v: int16x8_t, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` guarantees `dst` covers the 8 `i32` written.
        unsafe { store_widen_i16(dst.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn slli<const N: i32>(v: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { slli16::<N>(v) }
    }

    #[inline(always)]
    fn srli<const N: i32>(v: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { srli16::<N>(v) }
    }

    /// Diagonal shift by `LSS = 2` bytes (one `i16` lane) — the literal for this width, matching
    /// [`super::sse41::Sse41I16::slli_one_lane`]'s `slli_si::<2>`.
    #[inline(always)]
    fn slli_one_lane(v: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { slli16::<2>(v) }
    }

    /// Carry shift by `RSS = 14` bytes (isolate lane 7 into lane 0) — the literal for this width,
    /// matching [`super::sse41::Sse41I16::srli_top_lane`]'s `srli_si::<14>`.
    #[inline(always)]
    fn srli_top_lane(v: int16x8_t) -> int16x8_t {
        // SAFETY: see `splat`.
        unsafe { srli16::<14>(v) }
    }

    #[inline(always)]
    fn horizontal_max(v: int16x8_t) -> i16 {
        let mut lanes = [0i16; 8];
        Self::storeu(v, &mut lanes);
        // Seed at 0 (the Smith-Waterman clamp), matching `_mmxxx_max_value` (`impl:240-250`) — NOT
        // a bare `vmaxvq_s16`, which would omit the load-bearing `0` seed.
        lanes.iter().fold(0i16, |acc, &x| acc.max(x))
    }

    /// Hand-unrolled 3-step ladder with byte-shifts `[2, 4, 8]` (`impl:182-184`), identical to
    /// [`super::sse41::Sse41I16::prefix_max`]; each step is the shared [`Simd::prefix_max_step`]
    /// with that step's literal byte-shift constant.
    #[inline(always)]
    fn prefix_max(v: int16x8_t, penalties: &[int16x8_t], masks: &[int16x8_t]) -> int16x8_t {
        debug_assert!(penalties.len() >= Self::LOG_LANES as usize);
        debug_assert!(masks.len() >= Self::LOG_LANES as usize);
        let mut a = v;
        a = Self::prefix_max_step::<2>(a, penalties[0], masks[0]);
        a = Self::prefix_max_step::<4>(a, penalties[1], masks[1]);
        a = Self::prefix_max_step::<8>(a, penalties[2], masks[2]);
        a
    }
}

// ---- NeonI32: 4 × i32 -------------------------------------------------------------------------

/// NEON backend over 4 packed `i32` lanes — the structural twin of [`super::sse41::Sse41I32`].
pub(crate) struct NeonI32;

/// Lane-typed `i32` add. Ports `_mm_add_epi32` (`impl:198`) as NEON `vaddq_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vaddq_s32(a, b)
}

/// Lane-typed `i32` sub. Ports `_mm_sub_epi32` (`impl:201`) as NEON `vsubq_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sub32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vsubq_s32(a, b)
}

/// Lane-typed `i32` **saturating** add. NEON has native int32 saturation (`vqaddq_s32`), unlike
/// x86 SSE4.1/AVX2, which have no `_mm(256)_adds_epi32` and must emulate it (see
/// `super::sse41::adds32`/`super::avx2::adds32`).
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn adds32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vqaddq_s32(a, b)
}

/// Lane-typed `i32` **saturating** sub. Native NEON `vqsubq_s32`; see [`adds32`].
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn subs32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vqsubq_s32(a, b)
}

/// Lane-typed `i32` signed min. Ports `_mm_min_epi32` (`impl:204`) as NEON `vminq_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
// Test-only trait op; see the note on `min16` above.
#[allow(dead_code)]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn min32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vminq_s32(a, b)
}

/// Lane-typed `i32` signed max. Ports `_mm_max_epi32` (`impl:207`) as NEON `vmaxq_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn max32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vmaxq_s32(a, b)
}

/// Bitwise OR of two `i32` registers. Ports `_mm_or_si128` (`impl:143`) as NEON `vorrq_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn or32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vorrq_s32(a, b)
}

/// Broadcast `value` into all 4 `i32` lanes. Ports `_mm_set1_epi32` (`impl:210`) as `vdupq_n_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn set1_32(value: i32) -> int32x4_t {
    vdupq_n_s32(value)
}

/// Unaligned load of 4 `i32` from `src`. Ports `_mm_loadu_si128` as NEON `vld1q_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available AND that `src` points to at least 4 readable `i32`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn loadu32(src: *const i32) -> int32x4_t {
    vld1q_s32(src)
}

/// Unaligned store of 4 `i32` to `dst`. Ports `_mm_storeu_si128` as NEON `vst1q_s32`.
///
/// # Safety
/// Caller must guarantee NEON is available AND that `dst` points to at least 4 writable `i32`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn storeu32(dst: *mut i32, v: int32x4_t) {
    vst1q_s32(dst, v);
}

/// Byte-shift-left of an `i32` register by `N` bytes (via the shared [`byte_shift_left`]).
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn slli32<const N: i32>(v: int32x4_t) -> int32x4_t {
    vreinterpretq_s32_s8(byte_shift_left::<N>(vreinterpretq_s8_s32(v)))
}

/// Byte-shift-right of an `i32` register by `N` bytes (via the shared [`byte_shift_right`]).
///
/// # Safety
/// Caller must guarantee NEON is available (see the module-level Safety note).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn srli32<const N: i32>(v: int32x4_t) -> int32x4_t {
    vreinterpretq_s32_s8(byte_shift_right::<N>(vreinterpretq_s8_s32(v)))
}

impl Simd for NeonI32 {
    type Elem = i32;
    type Vec = int32x4_t;

    const LANES: usize = 4;
    const LOG_LANES: u32 = 2;
    const LSS: i32 = size_of::<i32>() as i32; // 4
    const RSS: i32 = 16 - size_of::<i32>() as i32; // 12
    const NEG_INF: i32 = NEG_INF_I32;

    #[inline(always)]
    fn splat(value: i32) -> int32x4_t {
        // SAFETY: only reached after `is_aarch64_feature_detected!("neon")` (see module Safety note).
        unsafe { set1_32(value) }
    }

    #[inline(always)]
    fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`. Non-saturating `vaddq_s32` (NOT `vqaddq_s32`) per Global Constraints.
        unsafe { add32(a, b) }
    }

    #[inline(always)]
    fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`. Non-saturating `vsubq_s32`.
        unsafe { sub32(a, b) }
    }

    #[inline(always)]
    fn adds(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`. Native saturating `vqaddq_s32` (no x86-style emulation needed).
        unsafe { adds32(a, b) }
    }

    #[inline(always)]
    fn subs(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`. Native saturating `vqsubq_s32`.
        unsafe { subs32(a, b) }
    }

    #[inline(always)]
    fn min(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { min32(a, b) }
    }

    #[inline(always)]
    fn max(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { max32(a, b) }
    }

    #[inline(always)]
    fn or(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { or32(a, b) }
    }

    #[inline(always)]
    fn loadu(src: &[i32]) -> int32x4_t {
        debug_assert!(src.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` (and the trait's documented precondition)
        // guarantee `src` covers the 4 `i32` read.
        unsafe { loadu32(src.as_ptr()) }
    }

    #[inline(always)]
    fn storeu(v: int32x4_t, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `loadu`, mirrored for the 4-`i32` write.
        unsafe { storeu32(dst.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn store_widened_i32(v: int32x4_t, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // Elem is already `i32`; the "widen" is a plain 4-`i32` store.
        // SAFETY: see `loadu`, mirrored for the 4-`i32` write.
        unsafe { storeu32(dst.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn slli<const N: i32>(v: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { slli32::<N>(v) }
    }

    #[inline(always)]
    fn srli<const N: i32>(v: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { srli32::<N>(v) }
    }

    /// Diagonal shift by `LSS = 4` bytes (one `i32` lane) — the literal for this width, matching
    /// [`super::sse41::Sse41I32::slli_one_lane`]'s `slli_si::<4>`.
    #[inline(always)]
    fn slli_one_lane(v: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { slli32::<4>(v) }
    }

    /// Carry shift by `RSS = 12` bytes (isolate lane 3 into lane 0) — the literal for this width,
    /// matching [`super::sse41::Sse41I32::srli_top_lane`]'s `srli_si::<12>`.
    #[inline(always)]
    fn srli_top_lane(v: int32x4_t) -> int32x4_t {
        // SAFETY: see `splat`.
        unsafe { srli32::<12>(v) }
    }

    #[inline(always)]
    fn horizontal_max(v: int32x4_t) -> i32 {
        let mut lanes = [0i32; 4];
        Self::storeu(v, &mut lanes);
        // Seed at 0 (the Smith-Waterman clamp), matching `_mmxxx_max_value` (`impl:240-250`) — NOT
        // a bare `vmaxvq_s32`.
        lanes.iter().fold(0i32, |acc, &x| acc.max(x))
    }

    /// Hand-unrolled 2-step ladder with byte-shifts `[4, 8]` (`impl:217-218`), identical to
    /// [`super::sse41::Sse41I32::prefix_max`]; each step is the shared [`Simd::prefix_max_step`]
    /// with that step's literal byte-shift constant.
    #[inline(always)]
    fn prefix_max(v: int32x4_t, penalties: &[int32x4_t], masks: &[int32x4_t]) -> int32x4_t {
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

    /// True on any aarch64 host (NEON is architecturally baseline); kept as an explicit runtime
    /// gate to mirror the SSE4.1 tests' `is_x86_feature_detected!` idiom and to document the
    /// `target_feature` precondition every NEON call below relies on.
    fn neon_available() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    /// Extracts a `NeonI16` register's 8 lanes into a plain array for comparison.
    fn unpack16(v: int16x8_t) -> [i16; 8] {
        let mut out = [0i16; 8];
        NeonI16::storeu(v, &mut out);
        out
    }

    /// Extracts a `NeonI32` register's 4 lanes into a plain array for comparison.
    fn unpack32(v: int32x4_t) -> [i32; 4] {
        let mut out = [0i32; 4];
        NeonI32::storeu(v, &mut out);
        out
    }

    /// Independent scalar reference for the prefix-max recurrence: lane `j` is the max over all
    /// `k <= j` of `a[k] + (j - k) * penalty` (computed in `i32` to avoid intermediate overflow) —
    /// the closed form the shift-and-max ladder computes. Deriving it WITHOUT any intrinsic is
    /// exactly what makes it a valid oracle for the NEON ladder's `vextq` byte→element constants.
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

    /// Reference byte-shift-left (toward higher lane indices) over 8 `i16` lanes; `nbytes` a
    /// multiple of 2, so the shift is by `nbytes / 2` whole lanes with zero fill — matching
    /// `_mm_slli_si128`'s semantics at lane granularity (and thus the NEON `vextq_s8` emulation).
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
        if !neon_available() {
            return;
        }
        let a = [3i16, -2, 5, 1, 0, 7, -4, 2];
        let b = [1i16, 1, -3, 4, -5, 2, 6, -1];
        let va = NeonI16::loadu(&a);
        let vb = NeonI16::loadu(&b);

        assert_eq!(unpack16(NeonI16::splat(-9)), [-9i16; 8]);

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
        assert_eq!(unpack16(NeonI16::add(va, vb)), exp_add);
        assert_eq!(unpack16(NeonI16::sub(va, vb)), exp_sub);
        assert_eq!(unpack16(NeonI16::min(va, vb)), exp_min);
        assert_eq!(unpack16(NeonI16::max(va, vb)), exp_max);
        assert_eq!(unpack16(NeonI16::or(va, vb)), exp_or);
    }

    #[test]
    fn i16_add_sub_are_non_saturating() {
        if !neon_available() {
            return;
        }
        // i16::MAX + 1 wraps to i16::MIN (modular `vaddq_s16`, NOT saturating `vqaddq_s16`).
        let hi = NeonI16::splat(i16::MAX);
        let lo = NeonI16::splat(i16::MIN);
        let one = NeonI16::splat(1);
        assert_eq!(unpack16(NeonI16::add(hi, one)), [i16::MIN; 8]);
        assert_eq!(unpack16(NeonI16::sub(lo, one)), [i16::MAX; 8]);
    }

    #[test]
    fn i16_adds_subs_saturate_at_bounds() {
        if !neon_available() {
            return;
        }
        // adds/subs must clamp at the element bound (native `vqaddq_s16`/`vqsubq_s16`), unlike
        // `add`/`sub` which wrap.
        let hi = NeonI16::splat(i16::MAX);
        let lo = NeonI16::splat(i16::MIN);
        let one = NeonI16::splat(1);
        assert_eq!(unpack16(NeonI16::adds(hi, one)), [i16::MAX; 8]);
        assert_eq!(unpack16(NeonI16::subs(lo, one)), [i16::MIN; 8]);

        // Lane-wise match against the scalar `saturating_add`/`saturating_sub` reference across a
        // mix of ordinary and boundary values.
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
        let va = NeonI16::loadu(&a);
        let vb = NeonI16::loadu(&b);
        let mut exp_adds = [0i16; 8];
        let mut exp_subs = [0i16; 8];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_adds[i] = ai.saturating_add(bi);
            exp_subs[i] = ai.saturating_sub(bi);
        }
        assert_eq!(unpack16(NeonI16::adds(va, vb)), exp_adds);
        assert_eq!(unpack16(NeonI16::subs(va, vb)), exp_subs);
    }

    #[test]
    fn i16_loadu_storeu_round_trip() {
        if !neon_available() {
            return;
        }
        let src = [10i16, 20, 30, 40, 50, 60, 70, 80];
        let v = NeonI16::loadu(&src);
        let mut dst = [0i16; 8];
        NeonI16::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_slli_srli_have_byte_shift_semantics() {
        if !neon_available() {
            return;
        }
        // This is the exact test that catches a wrong `vextq` byte→element conversion or a
        // swapped zero-vector operand order: the reference shifts are computed purely in Rust.
        let a = [1i16, 2, 3, 4, 5, 6, 7, 8];
        let v = NeonI16::loadu(&a);
        assert_eq!(unpack16(NeonI16::slli::<2>(v)), shift_left_i16(&a, 2));
        assert_eq!(unpack16(NeonI16::slli::<4>(v)), shift_left_i16(&a, 4));
        assert_eq!(unpack16(NeonI16::slli::<8>(v)), shift_left_i16(&a, 8));
        assert_eq!(unpack16(NeonI16::srli::<2>(v)), shift_right_i16(&a, 2));
        // RSS = 14: isolates the single highest lane down to lane 0.
        assert_eq!(unpack16(NeonI16::srli::<14>(v)), shift_right_i16(&a, 14));
    }

    #[test]
    fn i16_slli_one_lane_and_srli_top_lane_match_lss_rss() {
        if !neon_available() {
            return;
        }
        let a = [1i16, 2, 3, 4, 5, 6, 7, 8];
        let v = NeonI16::loadu(&a);
        // slli_one_lane = shift left by LSS = 2 bytes (1 i16 lane).
        assert_eq!(unpack16(NeonI16::slli_one_lane(v)), shift_left_i16(&a, 2));
        // srli_top_lane = shift right by RSS = 14 bytes (lane 7 into lane 0).
        assert_eq!(unpack16(NeonI16::srli_top_lane(v)), shift_right_i16(&a, 14));
    }

    #[test]
    fn i16_store_widened_i32_sign_extends_all_lanes() {
        if !neon_available() {
            return;
        }
        let src = [-5i16, 2, -9, 11, i16::MIN, i16::MAX, 0, -1];
        let v = NeonI16::loadu(&src);
        let mut dst = [0i32; 8];
        NeonI16::store_widened_i32(v, &mut dst);
        let expected: [i32; 8] = std::array::from_fn(|k| i32::from(src[k]));
        assert_eq!(dst, expected);
    }

    #[test]
    fn i32_store_widened_i32_is_a_plain_store() {
        if !neon_available() {
            return;
        }
        let src = [-5i32, 123_456, i32::MIN, i32::MAX];
        let v = NeonI32::loadu(&src);
        let mut dst = [0i32; 4];
        NeonI32::store_widened_i32(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_horizontal_max_seeds_at_zero() {
        if !neon_available() {
            return;
        }
        // All-negative reduces to 0 (SW clamp), not the largest (least-negative) lane — NOT vmaxvq.
        assert_eq!(NeonI16::horizontal_max(NeonI16::splat(-5)), 0);
        let mixed = NeonI16::loadu(&[-5i16, -2, -9, -1, -3, -8, -7, -6]);
        assert_eq!(NeonI16::horizontal_max(mixed), 0);
        let positive = NeonI16::loadu(&[-5i16, 2, -9, 11, -3, 8, -7, 6]);
        assert_eq!(NeonI16::horizontal_max(positive), 11);
    }

    #[test]
    fn i16_prefix_max_matches_scalar_reference() {
        if !neon_available() {
            return;
        }
        let penalty: i16 = -4;
        let penalties = build_penalties::<NeonI16>(penalty);
        let masks = build_masks::<NeonI16>(NeonI16::NEG_INF);

        for a in [
            [3i16, -2, 5, 1, 0, 7, -4, 2],
            [0i16, 0, 0, 0, 0, 0, 0, 0],
            [8i16, 6, 4, 2, 0, -2, -4, -6],
            [-1i16, 9, -3, 2, 12, -8, 4, 5],
            // Dominant lane 0 forces the *full* ladder: lane 7's winner is `a[0] - 7*4 = 12`
            // (distance 7), reachable only via all three byte-shifts `[2, 4, 8]` — a wrong final
            // shift constant (e.g. a bad `vextq` element count) leaves lane 7 at a smaller value.
            [40i16, 1, 2, 3, 4, 5, 6, 7],
            // Dominant lane 1 forces distance-6 propagation into lane 7 (`a[1] - 6*4 = 12`).
            [0i16, 36, 1, 2, 3, 4, 5, 6],
        ] {
            let v = NeonI16::loadu(&a);
            let got = unpack16(NeonI16::prefix_max(v, &penalties, &masks));
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
        if !neon_available() {
            return;
        }
        let a = [3i32, -2, 500, -1];
        let b = [1i32, 1, -300, 4];
        let va = NeonI32::loadu(&a);
        let vb = NeonI32::loadu(&b);

        assert_eq!(unpack32(NeonI32::splat(-9)), [-9i32; 4]);

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
        assert_eq!(unpack32(NeonI32::add(va, vb)), exp_add);
        assert_eq!(unpack32(NeonI32::sub(va, vb)), exp_sub);
        assert_eq!(unpack32(NeonI32::min(va, vb)), exp_min);
        assert_eq!(unpack32(NeonI32::max(va, vb)), exp_max);
        assert_eq!(unpack32(NeonI32::or(va, vb)), exp_or);
    }

    #[test]
    fn i32_add_sub_are_non_saturating() {
        if !neon_available() {
            return;
        }
        let hi = NeonI32::splat(i32::MAX);
        let lo = NeonI32::splat(i32::MIN);
        let one = NeonI32::splat(1);
        assert_eq!(unpack32(NeonI32::add(hi, one)), [i32::MIN; 4]);
        assert_eq!(unpack32(NeonI32::sub(lo, one)), [i32::MAX; 4]);
    }

    #[test]
    fn i32_adds_subs_saturate_at_bounds() {
        if !neon_available() {
            return;
        }
        // adds/subs must clamp at the element bound (native `vqaddq_s32`/`vqsubq_s32`), unlike
        // `add`/`sub` which wrap.
        let hi = NeonI32::splat(i32::MAX);
        let lo = NeonI32::splat(i32::MIN);
        let one = NeonI32::splat(1);
        assert_eq!(unpack32(NeonI32::adds(hi, one)), [i32::MAX; 4]);
        assert_eq!(unpack32(NeonI32::subs(lo, one)), [i32::MIN; 4]);

        // Lane-wise match against the scalar `saturating_add`/`saturating_sub` reference,
        // including the exact `NEG_INF` sentinel repeatedly penalized (the banded fill's actual
        // usage pattern).
        let a = [3i32, i32::MAX, NeonI32::NEG_INF, -100];
        let b = [1i32, 1, -128, -50];
        let va = NeonI32::loadu(&a);
        let vb = NeonI32::loadu(&b);
        let mut exp_adds = [0i32; 4];
        let mut exp_subs = [0i32; 4];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_adds[i] = ai.saturating_add(bi);
            exp_subs[i] = ai.saturating_sub(bi);
        }
        assert_eq!(unpack32(NeonI32::adds(va, vb)), exp_adds);
        assert_eq!(unpack32(NeonI32::subs(va, vb)), exp_subs);
    }

    #[test]
    fn i32_loadu_storeu_round_trip() {
        if !neon_available() {
            return;
        }
        let src = [100i32, 200, 300, 400];
        let v = NeonI32::loadu(&src);
        let mut dst = [0i32; 4];
        NeonI32::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i32_slli_srli_have_byte_shift_semantics() {
        if !neon_available() {
            return;
        }
        let a = [11i32, 22, 33, 44];
        let v = NeonI32::loadu(&a);
        assert_eq!(unpack32(NeonI32::slli::<4>(v)), shift_left_i32(&a, 4));
        assert_eq!(unpack32(NeonI32::slli::<8>(v)), shift_left_i32(&a, 8));
        assert_eq!(unpack32(NeonI32::srli::<4>(v)), shift_right_i32(&a, 4));
        // RSS = 12: isolates the single highest lane down to lane 0.
        assert_eq!(unpack32(NeonI32::srli::<12>(v)), shift_right_i32(&a, 12));
    }

    #[test]
    fn i32_slli_one_lane_and_srli_top_lane_match_lss_rss() {
        if !neon_available() {
            return;
        }
        let a = [11i32, 22, 33, 44];
        let v = NeonI32::loadu(&a);
        // slli_one_lane = shift left by LSS = 4 bytes (1 i32 lane).
        assert_eq!(unpack32(NeonI32::slli_one_lane(v)), shift_left_i32(&a, 4));
        // srli_top_lane = shift right by RSS = 12 bytes (lane 3 into lane 0).
        assert_eq!(unpack32(NeonI32::srli_top_lane(v)), shift_right_i32(&a, 12));
    }

    #[test]
    fn i32_horizontal_max_seeds_at_zero() {
        if !neon_available() {
            return;
        }
        assert_eq!(NeonI32::horizontal_max(NeonI32::splat(-5)), 0);
        let mixed = NeonI32::loadu(&[-5i32, -2, -9, -1]);
        assert_eq!(NeonI32::horizontal_max(mixed), 0);
        let positive = NeonI32::loadu(&[-5i32, 42, -9, 11]);
        assert_eq!(NeonI32::horizontal_max(positive), 42);
    }

    #[test]
    fn i32_prefix_max_matches_scalar_reference() {
        if !neon_available() {
            return;
        }
        let penalty: i32 = -6;
        let penalties = build_penalties::<NeonI32>(penalty);
        let masks = build_masks::<NeonI32>(NeonI32::NEG_INF);

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
            let v = NeonI32::loadu(&a);
            let got = unpack32(NeonI32::prefix_max(v, &penalties, &masks)).to_vec();
            let expected = scalar_prefix_max(&a, penalty);
            assert_eq!(got, expected, "prefix_max i32 mismatch for {a:?}");
        }
    }

    #[test]
    fn lane_constants_match_upstream() {
        assert_eq!(NeonI16::LANES, 8);
        assert_eq!(NeonI16::LOG_LANES, 3);
        assert_eq!(NeonI16::LSS, 2);
        assert_eq!(NeonI16::RSS, 14);
        assert_eq!(NeonI16::NEG_INF, i16::MIN + 1024);
        assert_eq!(NeonI32::LANES, 4);
        assert_eq!(NeonI32::LOG_LANES, 2);
        assert_eq!(NeonI32::LSS, 4);
        assert_eq!(NeonI32::RSS, 12);
        assert_eq!(NeonI32::NEG_INF, i32::MIN + 1024);
    }
}
