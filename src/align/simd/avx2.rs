//! AVX2 (x86_64) [`Simd`] backends: `Avx2I16` (16×`i16`) and `Avx2I32` (8×`i32`), the 256-bit
//! sibling of the [`super::sse41`] SSE4.1 backends.
//!
//! Ports the `InstructionSet<__m256i, std::int16_t>` / `InstructionSet<__m256i, std::int32_t>`
//! AVX2 specializations from `third_party/spoa/src/simd_alignment_engine_implementation.hpp`
//! (`impl:59-127`, the `#if defined(__AVX2__)` branch): 256-bit `__m256i` registers packing 16
//! `i16` or 8 `i32` lanes. Every lane-typed op maps to a single `core::arch::x86_64` `_mm256_*`
//! intrinsic; only the byte shift is genuinely more work than SSE4.1 (see below).
//!
//! # The cross-128-bit-lane byte shift (the one place an AVX2 port silently breaks)
//!
//! x86's 256-bit byte shifts `_mm256_slli_si256`/`_mm256_srli_si256` do **NOT** cross the
//! 128-bit-lane boundary: they shift each 128-bit half independently, so a byte shifted off the
//! top of the low half is *dropped* rather than carried into the high half. A whole-register
//! 256-bit byte shift (which is what [`Simd::slli`]/[`Simd::srli`] must be, matching
//! `_mm_slli_si128`'s semantics at double width) therefore has to splice the two halves with
//! `_mm256_permute2x128_si256` and re-align with `_mm256_alignr_epi8`:
//!
//! - **`slli::<N>` (left, zero-fill low), `N < 16`:**
//!   `_mm256_alignr_epi8::<16 - N>(a, _mm256_permute2x128_si256::<0x08>(a, a))`. The permute
//!   control `0x08` (`_MM_SHUFFLE(0,0,2,0)`) builds `{low = 0, high = a_low}`; per-lane `alignr`
//!   then pulls the low half's spilled bytes up into the high half.
//! - **`slli::<N>`, `N >= 16`:** `_mm256_permute2x128_si256::<0x08>(a, a)` alone (`N = 16` moves the
//!   low half into the high half and zeroes the low half; the ladder never shifts left by `> 16`).
//! - **`srli::<N>` (right, zero-fill high), `N < 16`:**
//!   `_mm256_alignr_epi8::<N>(_mm256_permute2x128_si256::<0x81>(a, a), a)`.
//! - **`srli::<N>`, `N >= 16`:**
//!   `_mm256_srli_si256::<N - 16>(_mm256_permute2x128_si256::<0x81>(a, a))`.
//!
//! **The `srli` permute control is `0x81` = `_MM_SHUFFLE(2,0,0,1)`, NOT `0x21`.** `0x81` builds
//! `{low = a_high, high = 0}`: bit 7 (`0x80`) ZEROES the high output lane, so the carry
//! `x = srli(...)` reads clean zeros in its upper lanes. `0x21` would instead leave `a`'s high half
//! sitting in the result's high lane as garbage that then ORs into H's upper lanes and corrupts
//! scores — and because the fill's inter-segment carry ([`Simd::srli_top_lane`]) shifts by
//! `RSS` = 30 (int16) / 28 (int32) bytes, both `>= 16`, this `srli` path runs on *every* fill
//! iteration, so the bug would be pervasive rather than a corner case.
//!
//! Every `_mm256_alignr_epi8`/`_mm256_srli_si256`/`_mm256_permute2x128_si256` immediate must be a
//! compile-time constant, and stable Rust cannot compute `16 - N`/`N - 16` from a const-generic
//! `N` (that needs `generic_const_exprs`). The [`slli_si`]/[`srli_si`] helpers therefore `match N`
//! and dispatch to an arm whose every immediate is a plain literal — the same pattern
//! [`super::neon`] uses for its `vextq_s8` emulation. Only the `N` values the ladders/carry shifts
//! actually use are ever passed (slli: `2, 4, 8, 16`; srli: the small cross-lane cases plus
//! `RSS` = 28/30), and each is a valid immediate, so the module compiles.
//!
//! # Safety
//!
//! Every intrinsic-calling helper is a `#[target_feature(enable = "avx2")] unsafe fn`: calling one
//! is undefined behavior unless the running CPU actually has AVX2. Each [`Simd`] trait method wraps
//! its helper in an `unsafe` block whose precondition is *"reached only after
//! `is_x86_feature_detected!(\"avx2\")` returned true"* — exactly how [`super`]'s runtime dispatch
//! selects [`Avx2I16`]/[`Avx2I32`] (AVX2 has priority over SSE4.1) and how every test below is
//! gated. All `unsafe` in the crate is confined to `src/align/simd/` via the module's
//! `#![allow(unsafe_code)]`.

use super::lanes::Simd;
use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_alignr_epi8, _mm256_castsi256_si128,
    _mm256_cvtepi16_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_max_epi16,
    _mm256_max_epi32, _mm256_min_epi16, _mm256_min_epi32, _mm256_or_si256,
    _mm256_permute2x128_si256, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_srli_si256,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32,
};

/// `i16::MIN + 1024`, this backend's `kNegativeInfinity` (see [`Simd::NEG_INF`]).
const NEG_INF_I16: i16 = i16::MIN + 1024;
/// `i32::MIN + 1024`, this backend's `kNegativeInfinity` (see [`Simd::NEG_INF`]).
const NEG_INF_I32: i32 = i32::MIN + 1024;

// ---- shared, lane-width-agnostic `__m256i` helpers -------------------------------------------
//
// `or`/`slli`/`srli`/`loadu`/`storeu` operate on the whole 256-bit register regardless of lane
// width, so both `Avx2I16` and `Avx2I32` share these; only the lane-typed arithmetic
// (`add`/`sub`/`min`/`max`/`set1`) differs between the two.

/// Bitwise OR of two registers. Ports `_mm256_or_si256` (`impl:48`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn or_si(a: __m256i, b: __m256i) -> __m256i {
    _mm256_or_si256(a, b)
}

/// Whole-register 256-bit byte-shift-**left** by the compile-time constant `N` bytes, zero-filling
/// the vacated low-order bytes — the cross-128-bit-lane emulation of a true `_mm_slli_si128` at
/// double width (see the module-level "cross-128-bit-lane byte shift" note). `match N` because the
/// `alignr` immediate `16 - N` cannot be computed from the const-generic `N` on stable Rust.
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn slli_si<const N: i32>(a: __m256i) -> __m256i {
    // `{low = 0, high = a_low}`: the source of the bytes that spill up across the lane boundary.
    let spill = _mm256_permute2x128_si256::<0x08>(a, a);
    match N {
        0 => a,
        2 => _mm256_alignr_epi8::<14>(a, spill),
        4 => _mm256_alignr_epi8::<12>(a, spill),
        8 => _mm256_alignr_epi8::<8>(a, spill),
        16 => spill,
        _ => unreachable!("slli_si N out of the ladder's expected set {{0, 2, 4, 8, 16}}"),
    }
}

/// Whole-register 256-bit byte-shift-**right** by the compile-time constant `N` bytes, zero-filling
/// the vacated high-order bytes — the cross-128-bit-lane emulation of a true `_mm_srli_si128` at
/// double width. `N < 16` re-aligns via `alignr`; `N >= 16` (the fill's `RSS` carry) shifts the
/// permuted `{low = a_high, high = 0}` half. See the module-level note on the `0x81` control.
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn srli_si<const N: i32>(a: __m256i) -> __m256i {
    // `{low = a_high, high = 0}`: bit 7 of `0x81` zeroes the high lane so no garbage carries in.
    let hi_to_lo = _mm256_permute2x128_si256::<0x81>(a, a);
    match N {
        0 => a,
        2 => _mm256_alignr_epi8::<2>(hi_to_lo, a),
        4 => _mm256_alignr_epi8::<4>(hi_to_lo, a),
        8 => _mm256_alignr_epi8::<8>(hi_to_lo, a),
        16 => hi_to_lo,
        28 => _mm256_srli_si256::<12>(hi_to_lo), // RSS for int32 (28 - 16)
        30 => _mm256_srli_si256::<14>(hi_to_lo), // RSS for int16 (30 - 16)
        _ => unreachable!("srli_si N out of the expected set {{0, 2, 4, 8, 16, 28, 30}}"),
    }
}

/// Unaligned 256-bit load from `src`. Ports `_mm256_loadu_si256` (unaligned per the plan's
/// alignment decision; upstream uses the aligned `_mm256_load_si256` at `impl:40`).
///
/// # Safety
/// Caller must guarantee AVX2 is available AND that `src` points to at least 32 readable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn loadu(src: *const u8) -> __m256i {
    _mm256_loadu_si256(src.cast::<__m256i>())
}

/// Unaligned 256-bit store to `dst`. Ports `_mm256_storeu_si256` (see [`loadu`] on the unaligned
/// choice; upstream uses the aligned `_mm256_store_si256` at `impl:44`).
///
/// # Safety
/// Caller must guarantee AVX2 is available AND that `dst` points to at least 32 writable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn storeu(dst: *mut u8, v: __m256i) {
    _mm256_storeu_si256(dst.cast::<__m256i>(), v);
}

/// Sign-extends the 16 packed `i16` lanes of `v` to `i32` and stores them contiguously to the 16
/// `i32` slots at `dst` (64 bytes). Widens each 128-bit half with `_mm256_cvtepi16_epi32` (the low
/// half via `_mm256_castsi256_si128`, the high half via `_mm256_extracti128_si256::<1>`) and stores
/// the two resulting 8×`i32` vectors.
///
/// # Safety
/// Caller must guarantee AVX2 is available AND that `dst` points to at least 16 writable `i32`
/// (64 bytes).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_widen_i16(dst: *mut i32, v: __m256i) {
    let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
    let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v));
    _mm256_storeu_si256(dst.cast::<__m256i>(), lo);
    _mm256_storeu_si256(dst.add(8).cast::<__m256i>(), hi);
}

// ---- Avx2I16: 16 × i16 -----------------------------------------------------------------------

/// AVX2 backend over 16 packed `i16` lanes (`impl:59-92`). The structural twin of
/// [`super::sse41::Sse41I16`], doubled to 256 bits. Constructed only through `SimdEngine`'s
/// x86_64 AVX2 dispatch, at runtime when AVX2 is detected.
pub(crate) struct Avx2I16;

/// Lane-typed `i16` add. Ports `_mm256_add_epi16` (`impl:69`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi16(a, b)
}

/// Lane-typed `i16` sub. Ports `_mm256_sub_epi16` (`impl:72`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sub16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi16(a, b)
}

/// Lane-typed `i16` signed min. Ports `_mm256_min_epi16` (`impl:75`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
// The `min` trait op is test-only: the DP fill maximizes score (so it uses `max`, never `min`),
// so this faithful-port helper is dead in non-test builds.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn min16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epi16(a, b)
}

/// Lane-typed `i16` signed max. Ports `_mm256_max_epi16` (`impl:78`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn max16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_max_epi16(a, b)
}

/// Broadcast `value` into all 16 `i16` lanes. Ports `_mm256_set1_epi16` (`impl:81`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn set1_16(value: i16) -> __m256i {
    _mm256_set1_epi16(value)
}

impl Simd for Avx2I16 {
    type Elem = i16;
    type Vec = __m256i;

    const LANES: usize = 16;
    const LOG_LANES: u32 = 4;
    const LSS: i32 = size_of::<i16>() as i32; // 2
    const RSS: i32 = 32 - size_of::<i16>() as i32; // 30
    const NEG_INF: i16 = NEG_INF_I16;

    #[inline(always)]
    fn splat(value: i16) -> __m256i {
        // SAFETY: only reached after `is_x86_feature_detected!("avx2")` (see module Safety note).
        unsafe { set1_16(value) }
    }

    #[inline(always)]
    fn add(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`. Non-saturating `_mm256_add_epi16` per the plan's Global Constraints.
        unsafe { add16(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`. Non-saturating `_mm256_sub_epi16`.
        unsafe { sub16(a, b) }
    }

    #[inline(always)]
    fn min(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { min16(a, b) }
    }

    #[inline(always)]
    fn max(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { max16(a, b) }
    }

    #[inline(always)]
    fn or(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { or_si(a, b) }
    }

    #[inline(always)]
    fn loadu(src: &[i16]) -> __m256i {
        debug_assert!(src.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` (and the trait's documented precondition)
        // guarantee `src` covers the 32 bytes read.
        unsafe { loadu(src.as_ptr().cast::<u8>()) }
    }

    #[inline(always)]
    fn storeu(v: __m256i, dst: &mut [i16]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `loadu`, mirrored for the 32-byte write.
        unsafe { storeu(dst.as_mut_ptr().cast::<u8>(), v) }
    }

    #[inline(always)]
    fn store_widened_i32(v: __m256i, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` guarantees `dst` covers the 16 `i32` written.
        unsafe { store_widen_i16(dst.as_mut_ptr(), v) }
    }

    #[inline(always)]
    fn slli<const N: i32>(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<N>(v) }
    }

    #[inline(always)]
    fn srli<const N: i32>(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<N>(v) }
    }

    /// Diagonal shift by `LSS = 2` bytes (one `i16` lane), the literal for this width.
    #[inline(always)]
    fn slli_one_lane(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<2>(v) }
    }

    /// Carry shift by `RSS = 30` bytes (isolate lane 15 into lane 0), the literal for this width.
    #[inline(always)]
    fn srli_top_lane(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<30>(v) }
    }

    #[inline(always)]
    fn horizontal_max(v: __m256i) -> i16 {
        let mut lanes = [0i16; 16];
        Self::storeu(v, &mut lanes);
        // Seed at 0 (the Smith-Waterman clamp), matching `_mmxxx_max_value` (`impl:240-250`).
        lanes.iter().fold(0i16, |acc, &x| acc.max(x))
    }

    /// Hand-unrolled 4-step ladder with byte-shifts `[2, 4, 8, 16]` (`impl:84-92`). Each step is the
    /// shared [`Simd::prefix_max_step`] with that step's literal shift constant. The `16` step is
    /// the one that exercises the cross-128-bit-lane path (`slli_si`'s `N >= 16` arm).
    #[inline(always)]
    fn prefix_max(v: __m256i, penalties: &[__m256i], masks: &[__m256i]) -> __m256i {
        debug_assert!(penalties.len() >= Self::LOG_LANES as usize);
        debug_assert!(masks.len() >= Self::LOG_LANES as usize);
        let mut a = v;
        a = Self::prefix_max_step::<2>(a, penalties[0], masks[0]);
        a = Self::prefix_max_step::<4>(a, penalties[1], masks[1]);
        a = Self::prefix_max_step::<8>(a, penalties[2], masks[2]);
        a = Self::prefix_max_step::<16>(a, penalties[3], masks[3]);
        a
    }
}

// ---- Avx2I32: 8 × i32 ------------------------------------------------------------------------

/// AVX2 backend over 8 packed `i32` lanes (`impl:94-127`). The structural twin of
/// [`super::sse41::Sse41I32`], doubled to 256 bits.
pub(crate) struct Avx2I32;

/// Lane-typed `i32` add. Ports `_mm256_add_epi32` (`impl:105`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

/// Lane-typed `i32` sub. Ports `_mm256_sub_epi32` (`impl:108`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sub32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi32(a, b)
}

/// Lane-typed `i32` signed min. Ports `_mm256_min_epi32` (`impl:111`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
// Test-only trait op; see the note on `min16` above.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn min32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epi32(a, b)
}

/// Lane-typed `i32` signed max. Ports `_mm256_max_epi32` (`impl:114`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn max32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_max_epi32(a, b)
}

/// Broadcast `value` into all 8 `i32` lanes. Ports `_mm256_set1_epi32` (`impl:117`).
///
/// # Safety
/// Caller must guarantee AVX2 is available (see the module-level Safety note).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn set1_32(value: i32) -> __m256i {
    _mm256_set1_epi32(value)
}

impl Simd for Avx2I32 {
    type Elem = i32;
    type Vec = __m256i;

    const LANES: usize = 8;
    const LOG_LANES: u32 = 3;
    const LSS: i32 = size_of::<i32>() as i32; // 4
    const RSS: i32 = 32 - size_of::<i32>() as i32; // 28
    const NEG_INF: i32 = NEG_INF_I32;

    #[inline(always)]
    fn splat(value: i32) -> __m256i {
        // SAFETY: only reached after `is_x86_feature_detected!("avx2")` (see module Safety note).
        unsafe { set1_32(value) }
    }

    #[inline(always)]
    fn add(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`. Non-saturating `_mm256_add_epi32` per the plan's Global Constraints.
        unsafe { add32(a, b) }
    }

    #[inline(always)]
    fn sub(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`. Non-saturating `_mm256_sub_epi32`.
        unsafe { sub32(a, b) }
    }

    #[inline(always)]
    fn min(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { min32(a, b) }
    }

    #[inline(always)]
    fn max(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { max32(a, b) }
    }

    #[inline(always)]
    fn or(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { or_si(a, b) }
    }

    #[inline(always)]
    fn loadu(src: &[i32]) -> __m256i {
        debug_assert!(src.len() >= Self::LANES);
        // SAFETY: see `splat`; the `debug_assert` (and the trait's documented precondition)
        // guarantee `src` covers the 32 bytes read.
        unsafe { loadu(src.as_ptr().cast::<u8>()) }
    }

    #[inline(always)]
    fn storeu(v: __m256i, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // SAFETY: see `loadu`, mirrored for the 32-byte write.
        unsafe { storeu(dst.as_mut_ptr().cast::<u8>(), v) }
    }

    #[inline(always)]
    fn store_widened_i32(v: __m256i, dst: &mut [i32]) {
        debug_assert!(dst.len() >= Self::LANES);
        // Elem is already `i32`; the "widen" is a plain 32-byte store.
        // SAFETY: see `loadu`, mirrored for the 32-byte write.
        unsafe { storeu(dst.as_mut_ptr().cast::<u8>(), v) }
    }

    #[inline(always)]
    fn slli<const N: i32>(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<N>(v) }
    }

    #[inline(always)]
    fn srli<const N: i32>(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<N>(v) }
    }

    /// Diagonal shift by `LSS = 4` bytes (one `i32` lane), the literal for this width.
    #[inline(always)]
    fn slli_one_lane(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { slli_si::<4>(v) }
    }

    /// Carry shift by `RSS = 28` bytes (isolate lane 7 into lane 0), the literal for this width.
    #[inline(always)]
    fn srli_top_lane(v: __m256i) -> __m256i {
        // SAFETY: see `splat`.
        unsafe { srli_si::<28>(v) }
    }

    #[inline(always)]
    fn horizontal_max(v: __m256i) -> i32 {
        let mut lanes = [0i32; 8];
        Self::storeu(v, &mut lanes);
        // Seed at 0 (the Smith-Waterman clamp), matching `_mmxxx_max_value` (`impl:240-250`).
        lanes.iter().fold(0i32, |acc, &x| acc.max(x))
    }

    /// Hand-unrolled 3-step ladder with byte-shifts `[4, 8, 16]` (`impl:120-127`). Each step is the
    /// shared [`Simd::prefix_max_step`] with that step's literal shift constant. The `16` step is
    /// the one that exercises the cross-128-bit-lane path (`slli_si`'s `N >= 16` arm).
    #[inline(always)]
    fn prefix_max(v: __m256i, penalties: &[__m256i], masks: &[__m256i]) -> __m256i {
        debug_assert!(penalties.len() >= Self::LOG_LANES as usize);
        debug_assert!(masks.len() >= Self::LOG_LANES as usize);
        let mut a = v;
        a = Self::prefix_max_step::<4>(a, penalties[0], masks[0]);
        a = Self::prefix_max_step::<8>(a, penalties[1], masks[1]);
        a = Self::prefix_max_step::<16>(a, penalties[2], masks[2]);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::simd::profile::{build_masks, build_penalties};

    /// True on any x86_64 host that exposes AVX2; the runtime gate every `target_feature` call
    /// below relies on. FALSE under Rosetta 2 (no AVX2) and on native arm64, so these tests
    /// no-op locally on the Apple-Silicon dev box and execute only on real AVX2 hardware / CI.
    fn avx2_available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// Extracts an `Avx2I16` register's 16 lanes into a plain array for comparison.
    fn unpack16(v: __m256i) -> [i16; 16] {
        let mut out = [0i16; 16];
        Avx2I16::storeu(v, &mut out);
        out
    }

    /// Extracts an `Avx2I32` register's 8 lanes into a plain array for comparison.
    fn unpack32(v: __m256i) -> [i32; 8] {
        let mut out = [0i32; 8];
        Avx2I32::storeu(v, &mut out);
        out
    }

    /// Independent scalar reference for the prefix-max recurrence: lane `j` is the max over all
    /// `k <= j` of `a[k] + (j - k) * penalty` (computed in `i32` to avoid intermediate overflow) —
    /// the closed form the shift-and-max ladder computes. Deriving it WITHOUT any intrinsic is
    /// exactly what makes it a valid oracle for the AVX2 ladder's cross-lane shift constants.
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

    /// Reference whole-register byte-shift-left (toward higher lane indices) over 16 `i16` lanes;
    /// `nbytes` a multiple of 2 (one `i16` = 2 bytes), so the shift is by `nbytes / 2` whole lanes
    /// with zero fill — the TRUE 256-bit semantics that cross the 128-bit-lane boundary (which a
    /// naive `_mm256_slli_si256` would NOT). This is what catches a wrong permute/alignr constant.
    fn shift_left_i16(a: &[i16; 16], nbytes: usize) -> [i16; 16] {
        let sh = nbytes / 2;
        let mut out = [0i16; 16];
        for (i, &x) in a.iter().enumerate() {
            if i + sh < 16 {
                out[i + sh] = x;
            }
        }
        out
    }

    /// Reference whole-register byte-shift-right (toward lower lane indices) over 16 `i16` lanes;
    /// see [`shift_left_i16`].
    fn shift_right_i16(a: &[i16; 16], nbytes: usize) -> [i16; 16] {
        let sh = nbytes / 2;
        let mut out = [0i16; 16];
        for (i, &x) in a.iter().enumerate() {
            if i >= sh {
                out[i - sh] = x;
            }
        }
        out
    }

    /// Reference whole-register byte-shift-left over 8 `i32` lanes; `nbytes` a multiple of 4.
    fn shift_left_i32(a: &[i32; 8], nbytes: usize) -> [i32; 8] {
        let sh = nbytes / 4;
        let mut out = [0i32; 8];
        for (i, &x) in a.iter().enumerate() {
            if i + sh < 8 {
                out[i + sh] = x;
            }
        }
        out
    }

    /// Reference whole-register byte-shift-right over 8 `i32` lanes; see [`shift_left_i32`].
    fn shift_right_i32(a: &[i32; 8], nbytes: usize) -> [i32; 8] {
        let sh = nbytes / 4;
        let mut out = [0i32; 8];
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
        if !avx2_available() {
            return;
        }
        let a = [3i16, -2, 5, 1, 0, 7, -4, 2, 9, -8, 11, -6, 13, -1, 15, -3];
        let b = [1i16, 1, -3, 4, -5, 2, 6, -1, 2, 2, -7, 8, -9, 3, 4, -2];
        let va = Avx2I16::loadu(&a);
        let vb = Avx2I16::loadu(&b);

        assert_eq!(unpack16(Avx2I16::splat(-9)), [-9i16; 16]);

        let mut exp_add = [0i16; 16];
        let mut exp_sub = [0i16; 16];
        let mut exp_min = [0i16; 16];
        let mut exp_max = [0i16; 16];
        let mut exp_or = [0i16; 16];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_add[i] = ai.wrapping_add(bi);
            exp_sub[i] = ai.wrapping_sub(bi);
            exp_min[i] = ai.min(bi);
            exp_max[i] = ai.max(bi);
            exp_or[i] = ai | bi;
        }
        assert_eq!(unpack16(Avx2I16::add(va, vb)), exp_add);
        assert_eq!(unpack16(Avx2I16::sub(va, vb)), exp_sub);
        assert_eq!(unpack16(Avx2I16::min(va, vb)), exp_min);
        assert_eq!(unpack16(Avx2I16::max(va, vb)), exp_max);
        assert_eq!(unpack16(Avx2I16::or(va, vb)), exp_or);
    }

    #[test]
    fn i16_add_sub_are_non_saturating() {
        if !avx2_available() {
            return;
        }
        // i16::MAX + 1 wraps to i16::MIN (modular `_mm256_add_epi16`, NOT saturating).
        let hi = Avx2I16::splat(i16::MAX);
        let lo = Avx2I16::splat(i16::MIN);
        let one = Avx2I16::splat(1);
        assert_eq!(unpack16(Avx2I16::add(hi, one)), [i16::MIN; 16]);
        assert_eq!(unpack16(Avx2I16::sub(lo, one)), [i16::MAX; 16]);
    }

    #[test]
    fn i16_loadu_storeu_round_trip() {
        if !avx2_available() {
            return;
        }
        let src = [
            10i16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let v = Avx2I16::loadu(&src);
        let mut dst = [0i16; 16];
        Avx2I16::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_slli_srli_cross_the_128_bit_lane_boundary() {
        if !avx2_available() {
            return;
        }
        // Lanes 0..8 live in the low 128-bit half, 8..16 in the high half; a shift of 8+ lanes
        // MUST carry across the boundary — exactly the cross-lane behavior a naive
        // `_mm256_slli_si256` gets wrong and the permute+alignr fixes.
        let a = [1i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let v = Avx2I16::loadu(&a);
        // slli: 2 bytes (1 lane, intra-half), 8 bytes (4 lanes), 16 bytes (8 lanes = whole half).
        assert_eq!(unpack16(Avx2I16::slli::<2>(v)), shift_left_i16(&a, 2));
        assert_eq!(unpack16(Avx2I16::slli::<4>(v)), shift_left_i16(&a, 4));
        assert_eq!(unpack16(Avx2I16::slli::<8>(v)), shift_left_i16(&a, 8));
        assert_eq!(unpack16(Avx2I16::slli::<16>(v)), shift_left_i16(&a, 16));
        // srli: small cross-lane cases plus RSS = 30 (isolate the single highest lane to lane 0).
        assert_eq!(unpack16(Avx2I16::srli::<2>(v)), shift_right_i16(&a, 2));
        assert_eq!(unpack16(Avx2I16::srli::<8>(v)), shift_right_i16(&a, 8));
        assert_eq!(unpack16(Avx2I16::srli::<16>(v)), shift_right_i16(&a, 16));
        assert_eq!(unpack16(Avx2I16::srli::<30>(v)), shift_right_i16(&a, 30));
    }

    #[test]
    fn i16_slli_one_lane_and_srli_top_lane_match_lss_rss() {
        if !avx2_available() {
            return;
        }
        let a = [1i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let v = Avx2I16::loadu(&a);
        // slli_one_lane = shift left by LSS = 2 bytes (1 i16 lane).
        assert_eq!(unpack16(Avx2I16::slli_one_lane(v)), shift_left_i16(&a, 2));
        // srli_top_lane = shift right by RSS = 30 bytes (lane 15 into lane 0).
        assert_eq!(unpack16(Avx2I16::srli_top_lane(v)), shift_right_i16(&a, 30));
    }

    #[test]
    fn i16_store_widened_i32_sign_extends_all_lanes() {
        if !avx2_available() {
            return;
        }
        let src = [
            -5i16,
            2,
            -9,
            11,
            i16::MIN,
            i16::MAX,
            0,
            -1,
            -4,
            2,
            -1,
            9,
            -8,
            21,
            -5,
            7,
        ];
        let v = Avx2I16::loadu(&src);
        let mut dst = [0i32; 16];
        Avx2I16::store_widened_i32(v, &mut dst);
        let expected: [i32; 16] = std::array::from_fn(|k| i32::from(src[k]));
        assert_eq!(dst, expected);
    }

    #[test]
    fn i32_store_widened_i32_is_a_plain_store() {
        if !avx2_available() {
            return;
        }
        let src = [-5i32, 123_456, i32::MIN, i32::MAX, 0, -1, 7, -100_000];
        let v = Avx2I32::loadu(&src);
        let mut dst = [0i32; 8];
        Avx2I32::store_widened_i32(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i16_horizontal_max_seeds_at_zero() {
        if !avx2_available() {
            return;
        }
        // All-negative reduces to 0 (SW clamp), not the largest (least-negative) lane.
        assert_eq!(Avx2I16::horizontal_max(Avx2I16::splat(-5)), 0);
        let mixed = Avx2I16::loadu(&[
            -5i16, -2, -9, -1, -3, -8, -7, -6, -4, -2, -1, -9, -8, -3, -5, -7,
        ]);
        assert_eq!(Avx2I16::horizontal_max(mixed), 0);
        // Winner in the HIGH 128-bit half, to prove the fold spans both halves.
        let positive =
            Avx2I16::loadu(&[-5i16, 2, -9, 1, -3, 8, -7, 6, -4, 2, -1, 9, -8, 21, -5, 7]);
        assert_eq!(Avx2I16::horizontal_max(positive), 21);
    }

    #[test]
    fn i16_prefix_max_matches_scalar_reference() {
        if !avx2_available() {
            return;
        }
        let penalty: i16 = -4;
        let penalties = build_penalties::<Avx2I16>(penalty);
        let masks = build_masks::<Avx2I16>(Avx2I16::NEG_INF);

        for a in [
            [3i16, -2, 5, 1, 0, 7, -4, 2, 9, -8, 11, -6, 13, -1, 15, -3],
            [0i16; 16],
            [
                16i16, 14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14,
            ],
            // Dominant lane 0 forces the *full* 4-step ladder: lane 15's winner is
            // `a[0] - 15*4 = 40` (distance 15), reachable only via all four byte-shifts
            // `[2, 4, 8, 16]` — and the `16` step is the CROSS-128-bit-lane one, so a wrong
            // permute/alignr constant leaves lanes 8..16 stuck at a nearer, smaller value.
            [100i16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            // Dominant lane 7 (top of the low half) propagating into lane 8 (bottom of the high
            // half) and beyond — a distance-8 jump `a[7] - 8*4 = 40` that must cross the boundary.
            [0i16, 1, 2, 3, 4, 5, 6, 72, 8, 9, 10, 11, 12, 13, 14, 15],
        ] {
            let v = Avx2I16::loadu(&a);
            let got = unpack16(Avx2I16::prefix_max(v, &penalties, &masks));
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
        if !avx2_available() {
            return;
        }
        let a = [3i32, -2, 500, -1, 9, -8, 700, -6];
        let b = [1i32, 1, -300, 4, -5, 2, 600, -1];
        let va = Avx2I32::loadu(&a);
        let vb = Avx2I32::loadu(&b);

        assert_eq!(unpack32(Avx2I32::splat(-9)), [-9i32; 8]);

        let mut exp_add = [0i32; 8];
        let mut exp_sub = [0i32; 8];
        let mut exp_min = [0i32; 8];
        let mut exp_max = [0i32; 8];
        let mut exp_or = [0i32; 8];
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            exp_add[i] = ai.wrapping_add(bi);
            exp_sub[i] = ai.wrapping_sub(bi);
            exp_min[i] = ai.min(bi);
            exp_max[i] = ai.max(bi);
            exp_or[i] = ai | bi;
        }
        assert_eq!(unpack32(Avx2I32::add(va, vb)), exp_add);
        assert_eq!(unpack32(Avx2I32::sub(va, vb)), exp_sub);
        assert_eq!(unpack32(Avx2I32::min(va, vb)), exp_min);
        assert_eq!(unpack32(Avx2I32::max(va, vb)), exp_max);
        assert_eq!(unpack32(Avx2I32::or(va, vb)), exp_or);
    }

    #[test]
    fn i32_add_sub_are_non_saturating() {
        if !avx2_available() {
            return;
        }
        let hi = Avx2I32::splat(i32::MAX);
        let lo = Avx2I32::splat(i32::MIN);
        let one = Avx2I32::splat(1);
        assert_eq!(unpack32(Avx2I32::add(hi, one)), [i32::MIN; 8]);
        assert_eq!(unpack32(Avx2I32::sub(lo, one)), [i32::MAX; 8]);
    }

    #[test]
    fn i32_loadu_storeu_round_trip() {
        if !avx2_available() {
            return;
        }
        let src = [100i32, 200, 300, 400, 500, 600, 700, 800];
        let v = Avx2I32::loadu(&src);
        let mut dst = [0i32; 8];
        Avx2I32::storeu(v, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn i32_slli_srli_cross_the_128_bit_lane_boundary() {
        if !avx2_available() {
            return;
        }
        // Lanes 0..4 in the low 128-bit half, 4..8 in the high half; a shift of 4+ lanes MUST
        // carry across the boundary.
        let a = [11i32, 22, 33, 44, 55, 66, 77, 88];
        let v = Avx2I32::loadu(&a);
        assert_eq!(unpack32(Avx2I32::slli::<4>(v)), shift_left_i32(&a, 4));
        assert_eq!(unpack32(Avx2I32::slli::<8>(v)), shift_left_i32(&a, 8));
        assert_eq!(unpack32(Avx2I32::slli::<16>(v)), shift_left_i32(&a, 16));
        assert_eq!(unpack32(Avx2I32::srli::<4>(v)), shift_right_i32(&a, 4));
        assert_eq!(unpack32(Avx2I32::srli::<8>(v)), shift_right_i32(&a, 8));
        assert_eq!(unpack32(Avx2I32::srli::<16>(v)), shift_right_i32(&a, 16));
        // RSS = 28: isolate the single highest lane down to lane 0.
        assert_eq!(unpack32(Avx2I32::srli::<28>(v)), shift_right_i32(&a, 28));
    }

    #[test]
    fn i32_slli_one_lane_and_srli_top_lane_match_lss_rss() {
        if !avx2_available() {
            return;
        }
        let a = [11i32, 22, 33, 44, 55, 66, 77, 88];
        let v = Avx2I32::loadu(&a);
        // slli_one_lane = shift left by LSS = 4 bytes (1 i32 lane).
        assert_eq!(unpack32(Avx2I32::slli_one_lane(v)), shift_left_i32(&a, 4));
        // srli_top_lane = shift right by RSS = 28 bytes (lane 7 into lane 0).
        assert_eq!(unpack32(Avx2I32::srli_top_lane(v)), shift_right_i32(&a, 28));
    }

    #[test]
    fn i32_horizontal_max_seeds_at_zero() {
        if !avx2_available() {
            return;
        }
        assert_eq!(Avx2I32::horizontal_max(Avx2I32::splat(-5)), 0);
        let mixed = Avx2I32::loadu(&[-5i32, -2, -9, -1, -4, -8, -7, -6]);
        assert_eq!(Avx2I32::horizontal_max(mixed), 0);
        // Winner in the HIGH 128-bit half.
        let positive = Avx2I32::loadu(&[-5i32, 42, -9, 11, -3, 8, -7, 96]);
        assert_eq!(Avx2I32::horizontal_max(positive), 96);
    }

    #[test]
    fn i32_prefix_max_matches_scalar_reference() {
        if !avx2_available() {
            return;
        }
        let penalty: i32 = -6;
        let penalties = build_penalties::<Avx2I32>(penalty);
        let masks = build_masks::<Avx2I32>(Avx2I32::NEG_INF);

        for a in [
            [3i32, -2, 5, 1, 9, -8, 7, 2],
            [0i32; 8],
            [16i32, 12, 8, 4, 0, -4, -8, -12],
            // Dominant lane 0 forces the full 3-step ladder: lane 7's winner is `a[0] - 7*6 = 58`
            // (distance 7), reachable only via all three byte-shifts `[4, 8, 16]` — and the `16`
            // step is the CROSS-128-bit-lane one, so a wrong permute/alignr constant leaves lanes
            // 4..8 stuck at a nearer, smaller value.
            [100i32, 1, 2, 3, 4, 5, 6, 7],
            // Dominant lane 3 (top of the low half) propagating into lane 4 (bottom of the high
            // half): a distance-4 jump `a[3] - 4*6 = 58` that must cross the boundary.
            [0i32, 1, 2, 82, 4, 5, 6, 7],
        ] {
            let v = Avx2I32::loadu(&a);
            let got = unpack32(Avx2I32::prefix_max(v, &penalties, &masks)).to_vec();
            let expected = scalar_prefix_max(&a, penalty);
            assert_eq!(got, expected, "prefix_max i32 mismatch for {a:?}");
        }
    }

    #[test]
    fn lane_constants_match_upstream() {
        assert_eq!(Avx2I16::LANES, 16);
        assert_eq!(Avx2I16::LOG_LANES, 4);
        assert_eq!(Avx2I16::LSS, 2);
        assert_eq!(Avx2I16::RSS, 30);
        assert_eq!(Avx2I16::NEG_INF, i16::MIN + 1024);
        assert_eq!(Avx2I32::LANES, 8);
        assert_eq!(Avx2I32::LOG_LANES, 3);
        assert_eq!(Avx2I32::LSS, 4);
        assert_eq!(Avx2I32::RSS, 28);
        assert_eq!(Avx2I32::NEG_INF, i32::MIN + 1024);
    }
}
