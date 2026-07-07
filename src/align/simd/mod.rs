//! SIMD-accelerated alignment engine.
//!
//! This module holds hand-tuned, per-ISA vectorized DP-fill kernels for
//! [`crate::align::AlignmentEngine`], dispatched at runtime: SSE4.1 and AVX2 on x86_64, NEON on
//! aarch64, each at the int16 and int32 tiers, with a scalar fallback. Every kernel is validated
//! bit-for-bit against the scalar [`crate::align::SisdEngine`] (its in-process oracle — see the
//! SIMD kernels plan's Global Constraints); [`SimdEngine`] returns exactly what `SisdEngine` would,
//! only faster. The kernels vectorize the DP fill and reuse the scalar backtrack, so the
//! accelerated path can never change the result.
//!
//! `unsafe` is confined to this module and its submodules: the crate root uses
//! `#![deny(unsafe_code)]` rather than `#![forbid(...)]` specifically so that this module can
//! reopen it with `#![allow(unsafe_code)]` below (a crate-level `forbid` cannot be relaxed by an
//! inner `allow` — rustc E0453). The `unsafe` here is exactly the per-ISA intrinsic calls, each
//! reached only after the matching runtime feature detection (see the module Safety note below).

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
mod avx2;
mod band;
mod fill;
mod lanes;
#[cfg(target_arch = "aarch64")]
mod neon;
mod profile;
#[cfg(target_arch = "x86_64")]
mod sse41;

pub use band::BandConfig;

use crate::align::backtrack::CellRead;
use crate::align::sisd::{ScalarInit, NEG_INF};
use crate::align::{Alignment, AlignmentEngine, AlignmentType, Scoring, SisdEngine};
use crate::graph::Graph;
use band::BandState;

/// The reused-across-`align`-calls **striped** SIMD scratch for one concrete register type `V`
/// (`__m128i`/`__m256i`/`int16x8_t`/`int32x4_t`): the striped char profile, the up-to-five striped
/// DP matrices, and the prefix-max ladder's masks/penalties. Every buffer is grow-only (only ever
/// resized *up*, never shrunk or reallocated smaller — mirroring [`crate::align::sisd`]'s own
/// `Realloc`); a smaller later alignment simply computes its offsets from its own dimensions into a
/// possibly-oversized buffer, and each fill fully re-writes (or `clear`+`resize`-refills) every cell
/// it later reads, so no stale value from a prior (larger) call is ever observed. This is the P2
/// "striped-buffer reuse" that removes the per-`align` allocation of these buffers, with zero change
/// to output.
///
/// The masks/penalties depend ONLY on the (fixed) scoring and the element width, so they are cached
/// (built once and reused) rather than rebuilt per call; [`StripedBuffers::cached_elem_width`]
/// records which element width they were built for, so a same-engine escalation switch (e.g. int16
/// → int32, which shares this same register type on x86) transparently rebuilds them exactly once
/// on the width change. The DP matrices/profile share one register type across widths and are fully
/// refilled per call, so no width tracking is needed for them.
// Only instantiated on targets with a vectorized backend wired in.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct StripedBuffers<V> {
    /// The striped char profile ([`profile::build_profile`]); rebuilt every call (depends on `seq`).
    profile: Vec<V>,
    /// Striped main DP matrix `H` (all gap modes).
    h: Vec<V>,
    /// Striped sequence-axis gap matrix `E` (affine/convex; unused for linear).
    e: Vec<V>,
    /// Striped graph-axis gap matrix `F` (affine/convex; unused for linear).
    f: Vec<V>,
    /// Striped second-affine-layer graph-axis gap matrix `O` (convex only).
    o: Vec<V>,
    /// Striped second-affine-layer sequence-axis gap matrix `Q` (convex only).
    q: Vec<V>,
    /// Cached prefix-max ladder masks ([`profile::build_masks`]).
    masks: Vec<V>,
    /// Cached prefix-max penalty ladder ([`profile::build_penalties`]): from `g` (linear), `e`
    /// (affine), or the first extend `e` (convex).
    penalties: Vec<V>,
    /// Cached SECOND prefix-max penalty ladder (convex only): from the second extend `c`.
    penalties_c: Vec<V>,
    /// The element width (`size_of::<S::Elem>()`, i.e. 2 for `i16` / 4 for `i32`) the cached
    /// `masks`/`penalties`/`penalties_c` were built for, or `0` when nothing has been cached yet.
    /// Guards against reusing an int16-shaped ladder for an int32 alignment (and vice versa) when
    /// both widths share one register type.
    cached_elem_width: usize,
}

// Manual `Default` (not derived) so it applies for register types `V` that do NOT implement
// `Default` (e.g. `__m128i`): every field is an empty `Vec`/zero, independent of `V`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<V> Default for StripedBuffers<V> {
    fn default() -> StripedBuffers<V> {
        StripedBuffers {
            profile: Vec::new(),
            h: Vec::new(),
            e: Vec::new(),
            f: Vec::new(),
            o: Vec::new(),
            q: Vec::new(),
            masks: Vec::new(),
            penalties: Vec::new(),
            penalties_c: Vec::new(),
            cached_elem_width: 0,
        }
    }
}

/// The per-ISA striped scratch stored on [`SimdEngine`], one variant per concrete register type
/// actually used on the host. [`SimdEngine`] is NOT generic over the lane backend `S` (it dispatches
/// at runtime), and `S::Vec` differs by ISA/width — this enum is the clean, `unsafe`-free bridge:
/// each variant owns a concretely-typed [`StripedBuffers`], and the dispatch site (where the
/// concrete `S` is known) selects the matching variant.
///
/// On x86_64 one `Vec<__m128i>` serves BOTH SSE4.1 widths (`Sse41I16::Vec == Sse41I32::Vec ==
/// __m128i`) and one `Vec<__m256i>` serves BOTH AVX2 widths, so a single variant per ISA suffices.
/// On aarch64 `NeonI16::Vec == int16x8_t` and `NeonI32::Vec == int32x4_t` are genuinely distinct
/// types, so NEON needs one variant per width. Because the host's ISA is fixed (a deterministic
/// [`detect_isa`] per CPU), only ONE variant is ever live on a given host; the lazy initialization
/// in the accessor methods below builds the matching one on first use.
// On any single build host only the variants reachable through that host's `#[cfg(target_arch)]`
// exist; `None` is the always-present initial state (set in `SimdEngine::new`).
#[derive(Default)]
enum StripedScratch {
    /// x86_64 SSE4.1 (128-bit `__m128i`), shared by the int16 and int32 SSE4.1 kernels.
    #[cfg(target_arch = "x86_64")]
    Sse41(StripedBuffers<core::arch::x86_64::__m128i>),
    /// x86_64 AVX2 (256-bit `__m256i`), shared by the int16 and int32 AVX2 kernels.
    #[cfg(target_arch = "x86_64")]
    Avx2(StripedBuffers<core::arch::x86_64::__m256i>),
    /// aarch64 NEON int16 (`int16x8_t`).
    #[cfg(target_arch = "aarch64")]
    NeonI16(StripedBuffers<core::arch::aarch64::int16x8_t>),
    /// aarch64 NEON int32 (`int32x4_t`).
    #[cfg(target_arch = "aarch64")]
    NeonI32(StripedBuffers<core::arch::aarch64::int32x4_t>),
    /// No striped scratch allocated yet (the initial state), or no vectorized backend on this host.
    #[default]
    None,
}

/// Which int-width kernel a given `(scoring, seq_len, node_count)` triple must use, based on the
/// worst-case DP-cell score that combination can reach.
///
/// Ports the escalation ladder in `spoa::SimdAlignmentEngine::Align`
/// (`simd_alignment_engine_implementation.hpp:664-672`): int16 lanes pack more parallelism per
/// register, so they're preferred whenever the worst case fits; a worse (more negative) case
/// escalates to int32, and a worst case that would overflow even `i32` has no safe vectorized
/// representation at all and must fall back to the scalar engine (upstream instead throws
/// `std::invalid_argument` there — this port has no fallible `align`, so it falls back to
/// [`SisdEngine`] instead, which is unconditionally correct).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Escalation {
    /// The worst case fits within the int16 DP-cell range; the int16 kernel may be used.
    Int16,
    /// The worst case overflows int16 but still fits within int32; the int32 kernel is required.
    Int32,
    /// The worst case would overflow even int32; no vectorized kernel is safe. Fall back to
    /// [`SisdEngine`].
    Fallback,
}

/// Computes the [`Escalation`] tier for aligning a `seq_len`-long sequence against a graph with
/// `node_count` nodes under `scoring`.
///
/// Ports `simd_alignment_engine_implementation.hpp:664-672` EXACTLY:
/// - The worst-case score is [`Scoring::worst_case_alignment_score`] evaluated at `(seq_len + 8,
///   node_count)` — **note the `+ 8`**, which reserves headroom for the up-to-8 padding lanes a
///   striped SIMD profile can introduce (each contributes a `padding_penalty` that can drive a
///   cell below the *unpadded* worst case, `impl:471-473,664-666`) — **and note the second
///   argument is the graph's node COUNT**, not a longest-path length (`impl:666`; mirrored by
///   `sisd.rs`'s own `SisdEngine::align` overflow guard, which uses the same `graph.nodes.len()`
///   at the unpadded `seq_len`).
/// - `worst_case < i32::MIN + 1024` selects [`Escalation::Fallback`].
/// - Otherwise, `worst_case < i16::MIN + 1024` selects [`Escalation::Int32`].
/// - Otherwise, [`Escalation::Int16`].
///
/// Comparisons are done in `i64` (matching [`Scoring::worst_case_alignment_score`]'s return type),
/// so the `i32`/`i16` bounds are widened via `i64::from` rather than compared in their native
/// width.
fn escalate(scoring: &Scoring, seq_len: usize, node_count: usize) -> Escalation {
    let worst_case = scoring.worst_case_alignment_score(seq_len as i64 + 8, node_count as i64);
    if worst_case < i64::from(i32::MIN) + 1024 {
        Escalation::Fallback
    } else if worst_case < i64::from(i16::MIN) + 1024 {
        Escalation::Int32
    } else {
        Escalation::Int16
    }
}

/// The instruction set selected at runtime for a vectorized kernel.
///
/// Mirrors the SIMD kernels plan's "Runtime dispatch, SISD fallback" constraint: prefer AVX2 over
/// SSE4.1 on x86_64 (a superset when available), NEON on aarch64 (baseline on that architecture),
/// or [`Isa::None`] when nothing usable was detected (which also falls back to [`SisdEngine`]).
// On any single build host only the variants reachable through that host's
// `#[cfg(target_arch = ...)]` branch in `detect_isa` are ever constructed: on aarch64, `Avx2`/
// `Sse41` are never produced (and vice versa on x86_64 for `Neon`), which `dead_code` flags
// per-variant even though `Isa` itself (and `detect_isa`) are live. `cfg_attr`-gated per target
// rather than a blanket `#[allow]`, so each variant still gets a real dead-code check on the
// targets where it IS constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Isa {
    /// x86_64 AVX2 (256-bit vectors).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Avx2,
    /// x86_64 SSE4.1 (128-bit vectors).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Sse41,
    /// aarch64 NEON (128-bit vectors).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Neon,
    /// No usable vectorized ISA was detected; kernels fall back to [`SisdEngine`].
    None,
}

/// Environment variable that pins [`detect_isa`] to a lower-tier ISA than the CPU's best, honored
/// **only as a downgrade** to an ISA the current CPU actually supports. It can never enable an ISA
/// the hardware lacks, preserving the soundness invariant that a `#[target_feature]` `run_*` wrapper
/// is reached only after its feature was detected. The only meaningful value is `sse41` (pins to
/// SSE4.1 on an AVX2-capable x86 CPU); any other value is ignored and normal detection proceeds.
///
/// Intended for A/B profiling (AVX2 vs SSE4.1 on one host) and for exercising the SSE4.1 path under
/// test on AVX2 hardware. It does not affect output — every ISA is bit-exact with [`SisdEngine`].
// Only read inside the `#[cfg(target_arch = "x86_64")]` arm of `detect_isa` (the sole ISA with a
// meaningful downgrade); the `cfg_attr` keeps it compiled everywhere for the cross-target unit test
// while allowing dead-code on architectures where the non-test build never references it.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const FORCE_ISA_ENV: &str = "SPOARS_FORCE_ISA";

/// Whether [`FORCE_ISA_ENV`]'s raw value (`None` when the variable is unset) requests suppressing
/// AVX2 in favor of SSE4.1. Case-insensitive match on `sse41`; every other value (including unset,
/// empty, or an unrecognized ISA name) is `false`, i.e. leaves normal detection in place. Factored
/// out as a pure function so the decision is unit-testable without an AVX2 CPU (the surrounding
/// `is_x86_feature_detected!` gate is not).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn should_force_sse41(value: Option<&str>) -> bool {
    value.is_some_and(|raw| raw.eq_ignore_ascii_case("sse41"))
}

/// Detects the best available vectorized [`Isa`] on the current CPU at runtime.
///
/// `is_x86_feature_detected!`/`is_aarch64_feature_detected!` are only defined by `std` on their
/// respective architectures, so each branch is behind a matching `#[cfg(target_arch = ...)]` —
/// this keeps the function compiling (and returning [`Isa::None`]) on every other target rather
/// than failing to compile at all.
///
/// Honors [`FORCE_ISA_ENV`] as a downgrade only (see its doc): `SPOARS_FORCE_ISA=sse41` pins to
/// SSE4.1 on an AVX2 CPU, which is why the AVX2 arm additionally checks it is not being suppressed.
fn detect_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        // Downgrade hook: only "sse41" is meaningful (suppress AVX2 in favor of SSE4.1). Reading it
        // here (once per `align`, not per DP cell) keeps it out of the vectorized hot loop.
        let force_sse41 = should_force_sse41(std::env::var(FORCE_ISA_ENV).ok().as_deref());
        if is_x86_feature_detected!("avx2") && !force_sse41 {
            return Isa::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return Isa::Sse41;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Isa::Neon;
        }
    }
    Isa::None
}

/// A SIMD-accelerated [`AlignmentEngine`].
///
/// [`SimdEngine::align`] performs the real [`Escalation`]/[`Isa`] routing (see
/// [`escalate`]/[`detect_isa`]); the SSE4.1 + **linear-gap** branch runs a real vectorized fill
/// ([`fill::fill_linear`], destriped into the shared scalar backtrack) for all three
/// [`AlignmentType`]s — Global/NW (SIMD kernels plan Task 7) and Local/SW + Overlap/OV (Task 8);
/// the SSE4.1 + **affine-gap** branch does the same via [`fill::fill_affine`] for all three types
/// (Task 9a: Global/NW; Task 9b: Local/SW + Overlap/OV); the SSE4.1 + **convex-gap** branch runs
/// [`fill::fill_convex`] for all three types too (Task 10a: Global/NW; Task 10b: Local/SW +
/// Overlap/OV) — completing the full SSE4.1 int16 engine (all 9 type x mode combinations) as of
/// Task 10. Task 11 then completed the SSE4.1 engine entirely: every one of those three
/// `align_simd_*` pipelines is generic over the lane backend, so the SAME code instantiated with
/// [`sse41::Sse41I32`] (instead of [`sse41::Sse41I16`]) serves the `Escalation::Int32` tier too —
/// no separate int32 implementation was needed. Task 12 then reused those same generic pipelines
/// for **aarch64 NEON**: the NEON branch instantiates them with `neon::NeonI16`/`neon::NeonI32`,
/// so on Apple-Silicon / Graviton the full 9-combo engine (int16 + int32) runs natively. Tasks
/// 14-15 then reused the same pipelines for **x86_64 AVX2** (`avx2::Avx2I16`/`avx2::Avx2I32`,
/// 256-bit vectors), execution-validated bit-for-bit against `SisdEngine` on real AVX2 hardware
/// (all 9 combos × int16/int32). Only [`Isa::None`] (no usable vectorized ISA
/// detected on this host) still delegates to an internal [`SisdEngine`], which is unconditionally
/// correct. All paths preserve the "must equal SISD" contract bit-for-bit. Task 16 wires this
/// engine into the CLI (`src/bin/spoars.rs`) as the default, with `SisdEngine` kept as a hidden
/// force-scalar escape hatch (`SPOARS_FORCE_SISD=1`).
pub struct SimdEngine {
    /// The alignment type this engine was built with, kept alongside `inner` (which also owns a
    /// copy) so [`SimdEngine::align`] can route on it without reaching into `inner`'s private state.
    // Read by the x86_64 SSE4.1 and aarch64 NEON routing branches; on any other target it is unread
    // until that target's vectorized fill lands.
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    alignment_type: AlignmentType,
    /// The scoring this engine was built with, kept alongside `inner` (which also owns a copy) so
    /// [`SimdEngine::align`] can evaluate [`escalate`] / the gap mode without reaching into
    /// `inner`'s private state.
    scoring: Scoring,
    inner: SisdEngine,
    /// Grow-only, reused-across-calls row-major DP scratch (boundary buffers, `sequence_profile`,
    /// `node_id_to_rank`) — the SIMD analog of [`SisdEngine`]'s own buffer fields (P2, first pass).
    /// Every vectorized `align` re-seeds this in place via
    /// [`crate::align::sisd::reseed_scalar_buffers`] instead of allocating (and zeroing) fresh
    /// `Vec`s, which is the largest per-`align` allocation the SIMD path was making.
    // Unused on targets without a vectorized backend wired in (same rationale as the fields above).
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    scratch: ScalarInit,
    /// Grow-only, reused-across-calls **striped** SIMD scratch (the ISA-register-typed profile, DP
    /// matrices, and cached masks/penalties) — the deferred second half of P2. Held as a per-ISA
    /// [`StripedScratch`] enum (one variant per concrete register type) rather than as `S`-generic
    /// fields, since [`SimdEngine`] dispatches to a concrete backend at runtime and is not itself
    /// generic over `S`. Lazily initialized to the matching variant on first `align` of that
    /// (ISA, width); see [`StripedBuffers`] for the grow-only/zero-output-change invariant.
    // Unused on targets without a vectorized backend wired in (same rationale as `scratch`).
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    striped: StripedScratch,
    /// Opt-in, heuristic abPOA-style band. `None` (the [`SimdEngine::new`] default) means the exact,
    /// spoa-bit-exact fill; `Some(cfg)` makes every [`SimdEngine::align`] build a per-call
    /// [`BandState`] and restrict the fill to that band. Banded results are **NOT** bit-exact with
    /// spoa — an alignment needing an indel wider than the band can be missed. See [`BandConfig`].
    band: Option<BandConfig>,
}

impl SimdEngine {
    /// Builds a [`SimdEngine`] for the given alignment type and scoring, mirroring
    /// [`SisdEngine::new`]'s signature.
    pub fn new(alignment_type: AlignmentType, scoring: Scoring) -> SimdEngine {
        SimdEngine {
            alignment_type,
            scoring,
            inner: SisdEngine::new(alignment_type, scoring),
            scratch: ScalarInit::default(),
            striped: StripedScratch::None,
            band: None,
        }
    }

    /// Builds a **banded** (opt-in, heuristic) engine. Unlike [`SimdEngine::new`] this is NOT
    /// bit-exact with spoa — it may miss alignments needing an indel wider than the band. See
    /// [`BandConfig`].
    pub fn banded(alignment_type: AlignmentType, scoring: Scoring, band: BandConfig) -> SimdEngine {
        let mut engine = SimdEngine::new(alignment_type, scoring);
        engine.band = Some(band);
        engine
    }

    /// Returns disjoint `&mut` handles to the row-major scratch and the SSE4.1 striped buffers
    /// (`Vec<__m128i>`, shared by the int16 and int32 SSE4.1 kernels), lazily switching
    /// [`SimdEngine::striped`] to the [`StripedScratch::Sse41`] variant on first use. Splitting the
    /// borrow here (both are disjoint fields of `self`) lets the generic `align_simd_*` pipeline
    /// take both without aliasing.
    #[cfg(target_arch = "x86_64")]
    fn sse41_scratch(
        &mut self,
    ) -> (
        &mut ScalarInit,
        &mut StripedBuffers<core::arch::x86_64::__m128i>,
    ) {
        if !matches!(self.striped, StripedScratch::Sse41(_)) {
            self.striped = StripedScratch::Sse41(StripedBuffers::default());
        }
        let striped = match &mut self.striped {
            StripedScratch::Sse41(buffers) => buffers,
            _ => unreachable!("just ensured the Sse41 variant"),
        };
        (&mut self.scratch, striped)
    }

    /// AVX2 counterpart of [`SimdEngine::sse41_scratch`] (`Vec<__m256i>`, shared by the int16 and
    /// int32 AVX2 kernels).
    #[cfg(target_arch = "x86_64")]
    fn avx2_scratch(
        &mut self,
    ) -> (
        &mut ScalarInit,
        &mut StripedBuffers<core::arch::x86_64::__m256i>,
    ) {
        if !matches!(self.striped, StripedScratch::Avx2(_)) {
            self.striped = StripedScratch::Avx2(StripedBuffers::default());
        }
        let striped = match &mut self.striped {
            StripedScratch::Avx2(buffers) => buffers,
            _ => unreachable!("just ensured the Avx2 variant"),
        };
        (&mut self.scratch, striped)
    }

    /// NEON int16 counterpart of [`SimdEngine::sse41_scratch`] (`Vec<int16x8_t>`). NEON's two widths
    /// use genuinely distinct register types, so each gets its own variant (unlike x86's shared
    /// register); switching width re-initializes to the other variant on next use.
    #[cfg(target_arch = "aarch64")]
    fn neon_i16_scratch(
        &mut self,
    ) -> (
        &mut ScalarInit,
        &mut StripedBuffers<core::arch::aarch64::int16x8_t>,
    ) {
        if !matches!(self.striped, StripedScratch::NeonI16(_)) {
            self.striped = StripedScratch::NeonI16(StripedBuffers::default());
        }
        let striped = match &mut self.striped {
            StripedScratch::NeonI16(buffers) => buffers,
            _ => unreachable!("just ensured the NeonI16 variant"),
        };
        (&mut self.scratch, striped)
    }

    /// NEON int32 counterpart of [`SimdEngine::neon_i16_scratch`] (`Vec<int32x4_t>`).
    #[cfg(target_arch = "aarch64")]
    fn neon_i32_scratch(
        &mut self,
    ) -> (
        &mut ScalarInit,
        &mut StripedBuffers<core::arch::aarch64::int32x4_t>,
    ) {
        if !matches!(self.striped, StripedScratch::NeonI32(_)) {
            self.striped = StripedScratch::NeonI32(StripedBuffers::default());
        }
        let striped = match &mut self.striped {
            StripedScratch::NeonI32(buffers) => buffers,
            _ => unreachable!("just ensured the NeonI32 variant"),
        };
        (&mut self.scratch, striped)
    }
}

/// Runs the vectorized **linear-gap** fill pipeline for `seq` against `graph`, returning the same
/// `(Alignment, i32)` a [`SisdEngine`] would (the SIMD kernels plan's bit-exactness contract).
/// Generic over the lane backend `S`, so it serves BOTH ISAs and BOTH width tiers unchanged:
/// [`sse41::Sse41I16`]/[`sse41::Sse41I32`] on x86_64 (SIMD kernels plan Tasks 7-8, 11) and
/// `neon::NeonI16`/`neon::NeonI32` on aarch64 (Task 12) — the SAME pipeline, only the `S`
/// instantiation differs.
///
/// The pipeline (SIMD kernels plan Tasks 7-8): [`profile::seed_scalar_buffers`] (row 0 / column
/// 0 / row-major profile — the C2 fix) → [`profile::build_profile`]/[`profile::build_masks`]/
/// [`profile::build_penalties`] → [`fill::fill_linear`] (striped interior, per-type max-tracking
/// — NW/SW/OV) → [`profile::destripe_interior`] the interior over the seeded `h` → the shared
/// [`crate::align::backtrack::backtrack_linear`] using the fill's `(max_i, max_j, max_score)`.
///
/// Only reached after the caller's runtime feature check (`is_x86_feature_detected!("sse4.1")` for
/// [`Isa::Sse41`], `is_aarch64_feature_detected!("neon")` for [`Isa::Neon`]) selected an ISA whose
/// `target_feature` code inside `S` is therefore sound (see each backend's module Safety note).
///
/// `band = Some(..)` activates the banded fill and its `is_banded && max_score == NEG_INF`
/// Global/Overlap sentinel guard (a banded run that reaches no valid end-to-end endpoint collapses
/// to an empty alignment); `None` reproduces the exact full-matrix path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn align_simd_linear<S>(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
    seeded: &mut ScalarInit,
    striped: &mut StripedBuffers<S::Vec>,
    band: Option<&mut BandState>,
) -> (Alignment, i32)
where
    S: lanes::Simd,
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    use crate::align::backtrack::backtrack_linear_impl;
    use crate::align::sisd::reseed_scalar_buffers;
    use profile::{build_masks, build_penalties, build_profile, ElemFromI32};

    reseed_scalar_buffers(seeded, alignment_type, scoring, seq, graph);
    build_profile::<S>(&mut striped.profile, graph, seq, scoring);
    // Masks/penalties depend only on the (fixed) scoring and the element width, so build them once
    // and reuse — rebuilding only if a same-engine escalation switched the element width.
    let elem_width = core::mem::size_of::<S::Elem>();
    if striped.cached_elem_width != elem_width {
        striped.masks = build_masks::<S>(S::NEG_INF);
        striped.penalties = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.g)));
        striped.cached_elem_width = elem_width;
    }

    let is_banded = band.is_some();
    let (max_i, max_j, max_score) = fill::fill_linear::<S>(
        graph,
        seq.len(),
        scoring,
        alignment_type,
        seeded,
        &striped.profile,
        &striped.masks,
        &striped.penalties,
        &mut striped.h,
        band,
    );

    // Band-aware Global/Overlap sentinel guard (design §Fill clip Global fix, MAJOR 5). When
    // banding leaves no sink with a reachable end-to-end (Global) / overlap (Overlap) endpoint,
    // `fill_*` returns `max_score == NEG_INF` per its "column L or sentinel" rule. Short-circuit to
    // the empty alignment the backtrack's `(0, 0)` path already yields, WITHOUT running the
    // backtrack — whose `debug_assert h.get == max_score` would pass vacuously on the sentinel and
    // then walk into out-of-band cells. No-op on the unbanded path: exact Global always reaches
    // column L and Local seeds `max_score = 0`, so `max_score` is never NEG_INF there.
    if is_banded && max_score == NEG_INF {
        return (Alignment::new(), NEG_INF);
    }

    // Prototype (option 1): skip the destripe; read the striped H directly via `StripedView`.
    let matrix_width_vecs = seq.len().div_ceil(S::LANES);
    let h_view = StripedView::<S> {
        boundary: &seeded.h,
        striped: &striped.h,
        width_scalar: seeded.matrix_width,
        width_vecs: matrix_width_vecs,
    };

    let alignment = backtrack_linear_impl(
        graph,
        &seeded.node_id_to_rank,
        &seeded.sequence_profile,
        &h_view,
        seeded.matrix_width,
        alignment_type,
        &scoring,
        max_i,
        max_j,
        max_score,
    );
    (alignment, max_score)
}

/// Runs the vectorized **affine-gap** fill pipeline for `seq` against `graph`, returning the same
/// `(Alignment, i32)` a [`SisdEngine`] would (the SIMD kernels plan's bit-exactness contract).
/// Wired for all three [`AlignmentType`]s — Global/NW (SIMD kernels plan Task 9a) and Local/SW +
/// Overlap/OV (Task 9b), whose per-type max-tracking branches in [`fill::fill_affine`] mirror
/// [`fill::fill_linear`]'s (proven in Task 8). Generic over the lane backend `S` exactly as
/// [`align_simd_linear`] is (SSE4.1 or NEON, int16 or int32).
///
/// The pipeline mirrors [`align_simd_linear`] but destripes all three of H, **E and F** over the
/// seeded buffers and backtracks via [`crate::align::backtrack::backtrack_affine`]. The prefix-max
/// penalty ladder is built from the affine EXTEND penalty `e` (not `g`, as linear uses), matching
/// upstream (`simd_alignment_engine_implementation.hpp:1111`).
///
/// Only reached after the caller's runtime ISA feature check selected an ISA whose `target_feature`
/// code inside `S` is therefore sound (see [`align_simd_linear`]).
///
/// `band = Some(..)` activates the banded fill and its `is_banded && max_score == NEG_INF`
/// Global/Overlap sentinel guard (a banded run that reaches no valid end-to-end endpoint collapses
/// to an empty alignment); `None` reproduces the exact full-matrix path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn align_simd_affine<S>(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
    seeded: &mut ScalarInit,
    striped: &mut StripedBuffers<S::Vec>,
    band: Option<&mut BandState>,
) -> (Alignment, i32)
where
    S: lanes::Simd,
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    use crate::align::backtrack::backtrack_affine_impl;
    use crate::align::sisd::reseed_scalar_buffers;
    use profile::{build_masks, build_penalties, build_profile, ElemFromI32};

    reseed_scalar_buffers(seeded, alignment_type, scoring, seq, graph);
    build_profile::<S>(&mut striped.profile, graph, seq, scoring);
    // Cached once per element width (see `align_simd_linear`); affine's ladder uses the extend `e`.
    let elem_width = core::mem::size_of::<S::Elem>();
    if striped.cached_elem_width != elem_width {
        striped.masks = build_masks::<S>(S::NEG_INF);
        striped.penalties = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.e)));
        striped.cached_elem_width = elem_width;
    }

    let is_banded = band.is_some();
    let (max_i, max_j, max_score) = fill::fill_affine::<S>(
        graph,
        seq.len(),
        scoring,
        alignment_type,
        seeded,
        &striped.profile,
        &striped.masks,
        &striped.penalties,
        &mut striped.h,
        &mut striped.e,
        &mut striped.f,
        band,
    );

    // Band-aware Global/Overlap sentinel guard — see `align_simd_linear` for the full rationale.
    if is_banded && max_score == NEG_INF {
        return (Alignment::new(), NEG_INF);
    }

    // Prototype (option 1): skip the destripe; read the striped H/E/F directly via `StripedView`.
    let matrix_width_vecs = seq.len().div_ceil(S::LANES);
    let width_scalar = seeded.matrix_width;
    let h_view = StripedView::<S> {
        boundary: &seeded.h,
        striped: &striped.h,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };
    let e_view = StripedView::<S> {
        boundary: &seeded.e,
        striped: &striped.e,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };
    let f_view = StripedView::<S> {
        boundary: &seeded.f,
        striped: &striped.f,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };

    let alignment = backtrack_affine_impl(
        graph,
        &seeded.node_id_to_rank,
        &seeded.sequence_profile,
        &h_view,
        &e_view,
        &f_view,
        seeded.matrix_width,
        alignment_type,
        &scoring,
        max_i,
        max_j,
        max_score,
    );
    (alignment, max_score)
}

/// A [`CellRead`] view over a striped fill matrix, so the convex backtrack can index the striped
/// `H`/`E`/`F`/`O`/`Q` directly instead of destriping the whole interior first (prototype for the
/// "skip the full-matrix destripe" optimization). Interior cells (`i >= 1`, `j >= 1`) are read from
/// the striped buffer with the same lane mapping [`profile::destripe_interior`] uses; boundary cells
/// (`i == 0` or `j == 0`) are read from `boundary`, the scalar-seeded row-major buffer whose row 0
/// and column 0 [`reseed_scalar_buffers`] already fills.
struct StripedView<'a, S: lanes::Simd>
where
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    /// Scalar-seeded row-major buffer; only its row 0 and column 0 are read.
    boundary: &'a [i32],
    /// Full striped matrix, graph row `i`'s block at `[i * width_vecs ..]` (row 0 is the boundary).
    striped: &'a [S::Vec],
    /// Row-major width, `seq_len + 1`.
    width_scalar: usize,
    /// Striped width in vectors, `ceil(seq_len / LANES)`.
    width_vecs: usize,
}

impl<S: lanes::Simd> CellRead for StripedView<'_, S>
where
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> i32 {
        if i == 0 || j == 0 {
            return self.boundary[i * self.width_scalar + j];
        }
        let lanes = <S as lanes::Simd>::LANES;
        let pos = j - 1; // 0-based query position
        let seg = pos / lanes;
        let lane = pos % lanes;
        // Extract one lane of the striped cell, widened to i32. `LANES <= 32` for every backend
        // (max 16 for AVX2 int16), so a fixed 32-wide stack buffer avoids a per-cell heap alloc.
        debug_assert!(lanes <= 32);
        let mut buf = [<S::Elem as profile::ElemFromI32>::from_i32(0); 32];
        <S as lanes::Simd>::storeu(self.striped[i * self.width_vecs + seg], &mut buf[..lanes]);
        <S::Elem as profile::ElemToI32>::to_i32(buf[lane])
    }
}

/// Runs the vectorized **convex-gap** fill pipeline for `seq` against `graph`, returning the same
/// `(Alignment, i32)` a [`SisdEngine`] would (the SIMD kernels plan's bit-exactness contract).
/// Wired for all three [`AlignmentType`]s — Global/NW (SIMD kernels plan Task 10a) and Local/SW +
/// Overlap/OV (Task 10b), whose per-type max-tracking branches in [`fill::fill_convex`] mirror
/// [`fill::fill_linear`]/[`fill::fill_affine`]'s (proven in Tasks 8 and 9b). Generic over the lane
/// backend `S` exactly as [`align_simd_linear`] is (SSE4.1 or NEON, int16 or int32).
///
/// The pipeline mirrors [`align_simd_affine`] but adds the SECOND affine function's matrices; the
/// prototype reads all five of H, E, F, **O and Q** striped (via [`StripedView`]) instead of
/// destriping. Two prefix-max penalty ladders are built — one from the first extend `e` (for the
/// `E` ladder) and one from the second extend `c` (for the `Q` ladder), matching upstream
/// (`simd_alignment_engine_implementation.hpp:1559-1565`).
///
/// Only reached after the caller's runtime ISA feature check selected an ISA whose `target_feature`
/// code inside `S` is therefore sound (see [`align_simd_linear`]).
///
/// `band = Some(..)` activates the banded fill and its `is_banded && max_score == NEG_INF`
/// Global/Overlap sentinel guard (a banded run that reaches no valid end-to-end endpoint collapses
/// to an empty alignment); `None` reproduces the exact full-matrix path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn align_simd_convex<S>(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
    seeded: &mut ScalarInit,
    striped: &mut StripedBuffers<S::Vec>,
    band: Option<&mut BandState>,
) -> (Alignment, i32)
where
    S: lanes::Simd,
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    use crate::align::backtrack::backtrack_convex_impl;
    use crate::align::sisd::reseed_scalar_buffers;
    use profile::{build_masks, build_penalties, build_profile, ElemFromI32};

    reseed_scalar_buffers(seeded, alignment_type, scoring, seq, graph);
    build_profile::<S>(&mut striped.profile, graph, seq, scoring);
    // Cached once per element width (see `align_simd_linear`). Two ladders: the first affine's E
    // uses the extend `e`, the second affine's Q uses `c`.
    let elem_width = core::mem::size_of::<S::Elem>();
    if striped.cached_elem_width != elem_width {
        striped.masks = build_masks::<S>(S::NEG_INF);
        striped.penalties = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.e)));
        striped.penalties_c = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.c)));
        striped.cached_elem_width = elem_width;
    }

    let is_banded = band.is_some();
    let (max_i, max_j, max_score) = fill::fill_convex::<S>(
        graph,
        seq.len(),
        scoring,
        alignment_type,
        seeded,
        &striped.profile,
        &striped.masks,
        &striped.penalties,
        &striped.penalties_c,
        &mut striped.h,
        &mut striped.e,
        &mut striped.f,
        &mut striped.o,
        &mut striped.q,
        band,
    );

    // Band-aware Global/Overlap sentinel guard — see `align_simd_linear` for the full rationale.
    if is_banded && max_score == NEG_INF {
        return (Alignment::new(), NEG_INF);
    }

    // Prototype (option 1): skip the full-matrix destripe. Row 0 / column 0 are already the
    // scalar boundary in the seeded buffers (`reseed_scalar_buffers`); the interior stays striped
    // and is read on demand through `StripedView`, so the backtrack touches only the cells along
    // its (short) path instead of paying an O(rows*cols) transpose per matrix (x5 for convex).
    let matrix_width_vecs = seq.len().div_ceil(S::LANES);
    let width_scalar = seeded.matrix_width;
    let h_view = StripedView::<S> {
        boundary: &seeded.h,
        striped: &striped.h,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };
    let e_view = StripedView::<S> {
        boundary: &seeded.e,
        striped: &striped.e,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };
    let f_view = StripedView::<S> {
        boundary: &seeded.f,
        striped: &striped.f,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };
    let o_view = StripedView::<S> {
        boundary: &seeded.o,
        striped: &striped.o,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };
    let q_view = StripedView::<S> {
        boundary: &seeded.q,
        striped: &striped.q,
        width_scalar,
        width_vecs: matrix_width_vecs,
    };

    let alignment = backtrack_convex_impl(
        graph,
        &seeded.node_id_to_rank,
        &seeded.sequence_profile,
        &h_view,
        &e_view,
        &f_view,
        &o_view,
        &q_view,
        seeded.matrix_width,
        alignment_type,
        &scoring,
        max_i,
        max_j,
        max_score,
    );
    (alignment, max_score)
}

// ---- Per-ISA `#[target_feature]` entry wrappers ---------------------------------------------
//
// Each wrapper carries its ISA's `#[target_feature]` and calls straight into the generic
// `align_simd_*` pipeline. Because that pipeline — `align_simd_*` → `fill_*` → the `Simd` trait
// ops → the same-feature intrinsic islands (`add16` etc.) — is `#[inline]`/`#[inline(always)]`
// throughout, the ENTIRE tree inlines into this one feature-enabled function, and the intrinsic
// islands (same feature) then inline into that feature-enabled context. This reproduces
// minimap2's "whole translation unit compiled with `-mavx2`" trick: the hot DP loop compiles to
// inline vector instructions instead of one non-inlined `call` per vector op. Without the
// wrapper the plain generic pipeline lacks the target feature, so a `#[target_feature]` op helper
// cannot be inlined into it (a target-feature fn never inlines into a caller lacking the feature)
// and every vector op in the hot loop became a non-inlined call — ~4x slower than scalar on AVX2.
//
// # Safety
//
// Every wrapper is an `unsafe fn` solely because `#[target_feature]` requires it; the single
// precondition is that the running CPU actually has the named feature. [`SimdEngine::align`]
// reaches a given wrapper only through the [`Isa`] arm that [`detect_isa`] selected, and
// `detect_isa` returns `Isa::Avx2`/`Isa::Sse41`/`Isa::Neon` only after the matching
// `is_x86_feature_detected!("avx2")` / `is_x86_feature_detected!("sse4.1")` /
// `is_aarch64_feature_detected!("neon")` returned true. So the feature is guaranteed present at
// every call site (each `unsafe { run_*::<_>(...) }` below documents this same invariant).
macro_rules! define_simd_runners {
    ($arch:literal, $feature:literal, $linear:ident, $affine:ident, $convex:ident) => {
        #[cfg(target_arch = $arch)]
        #[target_feature(enable = $feature)]
        unsafe fn $linear<S>(
            alignment_type: AlignmentType,
            scoring: Scoring,
            seq: &[u8],
            graph: &Graph,
            seeded: &mut ScalarInit,
            striped: &mut StripedBuffers<S::Vec>,
            band: Option<&mut BandState>,
        ) -> (Alignment, i32)
        where
            S: lanes::Simd,
            S::Elem: profile::ElemFromI32 + profile::ElemToI32,
        {
            align_simd_linear::<S>(alignment_type, scoring, seq, graph, seeded, striped, band)
        }

        #[cfg(target_arch = $arch)]
        #[target_feature(enable = $feature)]
        unsafe fn $affine<S>(
            alignment_type: AlignmentType,
            scoring: Scoring,
            seq: &[u8],
            graph: &Graph,
            seeded: &mut ScalarInit,
            striped: &mut StripedBuffers<S::Vec>,
            band: Option<&mut BandState>,
        ) -> (Alignment, i32)
        where
            S: lanes::Simd,
            S::Elem: profile::ElemFromI32 + profile::ElemToI32,
        {
            align_simd_affine::<S>(alignment_type, scoring, seq, graph, seeded, striped, band)
        }

        #[cfg(target_arch = $arch)]
        #[target_feature(enable = $feature)]
        unsafe fn $convex<S>(
            alignment_type: AlignmentType,
            scoring: Scoring,
            seq: &[u8],
            graph: &Graph,
            seeded: &mut ScalarInit,
            striped: &mut StripedBuffers<S::Vec>,
            band: Option<&mut BandState>,
        ) -> (Alignment, i32)
        where
            S: lanes::Simd,
            S::Elem: profile::ElemFromI32 + profile::ElemToI32,
        {
            align_simd_convex::<S>(alignment_type, scoring, seq, graph, seeded, striped, band)
        }
    };
}

define_simd_runners!(
    "x86_64",
    "avx2",
    run_avx2_linear,
    run_avx2_affine,
    run_avx2_convex
);
define_simd_runners!(
    "x86_64",
    "sse4.1",
    run_sse41_linear,
    run_sse41_affine,
    run_sse41_convex
);
// NEON is architecturally baseline on aarch64, so its intrinsic islands already inline freely
// (this is why NEON never hit the x86 cliff); the wrapper is added for symmetry so the dispatch
// is uniform across all three ISAs.
define_simd_runners!(
    "aarch64",
    "neon",
    run_neon_linear,
    run_neon_affine,
    run_neon_convex
);

impl AlignmentEngine for SimdEngine {
    /// Aligns `seq` against `graph`.
    ///
    /// Ports `spoa::SimdAlignmentEngine::Align`'s routing
    /// (`simd_alignment_engine_implementation.hpp:653-672`): an empty graph or empty sequence
    /// short-circuits to `(Alignment::new(), 0)` before any kernel selection (this also sidesteps
    /// divide-by-`LANES` / empty-rank indexing a real kernel would otherwise have to guard); then
    /// [`escalate`] picks the int16/int32/fallback tier and [`detect_isa`] picks the ISA. Every
    /// real ISA runs a genuine vectorized kernel through the SAME generic `align_simd_*` pipelines:
    /// AVX2 (`avx2::Avx2I16`/`avx2::Avx2I32`) and SSE4.1
    /// ([`sse41::Sse41I16`]/[`sse41::Sse41I32`]) on x86_64, NEON (`neon::NeonI16`/`neon::NeonI32`)
    /// on aarch64, at both the int16 and int32 tiers. Only [`Escalation::Fallback`] (scores that
    /// would overflow even `i32`) and [`Isa::None`] (no usable vectorized ISA detected) delegate to
    /// the internal [`SisdEngine`]. Whichever path is taken, the result is bit-identical to
    /// `SisdEngine::align` by construction (every kernel is validated against it).
    fn align(&mut self, seq: &[u8], graph: &Graph) -> (Alignment, i32) {
        if graph.nodes.is_empty() || seq.is_empty() {
            return (Alignment::new(), 0);
        }

        let escalation = escalate(&self.scoring, seq.len(), graph.nodes.len());
        let isa = detect_isa();

        // Snapshot the `Copy` routing inputs up front so the per-ISA arms below can take a `&mut`
        // borrow of `self`'s scratch fields (via the `*_scratch` accessors) without also borrowing
        // `self` for these reads.
        let alignment_type = self.alignment_type;
        let scoring = self.scoring;

        // Build the per-call band once (if this is a banded engine) and thread `as_mut()` into the
        // single tier×ISA×gap-mode arm that `escalate`/`detect_isa` selected below. `escalate` is
        // static (one tier picked up front, no runtime int16→int32 retry), so exactly one `run_*`
        // call site executes per `align`; `Option::as_mut` reborrows, so it is fine that every arm
        // names `band_state.as_mut()` — only the taken arm evaluates it, and `band_state` outlives
        // the whole `match`. The rank map is built from the graph's always-current `rank_to_node`
        // (NOT `self.scratch.node_id_to_rank`, which is not seeded until inside the runner and is
        // stale here); it is exactly the inverse ranking the fills index `BandState` by via
        // `seeded.node_id_to_rank`, since both derive from the same topological order.
        let mut band_state = self.band.map(|cfg| {
            let mut node_id_to_rank = vec![0u32; graph.nodes.len()];
            for (rank, &nid) in graph.rank_to_node.iter().enumerate() {
                node_id_to_rank[nid.0 as usize] = rank as u32;
            }
            BandState::new(graph, &node_id_to_rank, seq.len(), cfg)
        });

        match escalation {
            // Worst case overflows even i32: no vectorized kernel is safe at any ISA. Note: when
            // `self.band` is `Some`, this silently returns the EXACT (unbanded) `SisdEngine` result
            // instead of a banded one — safe because exact >= banded (never wrong, just not sped up).
            Escalation::Fallback => self.inner.align(seq, graph),
            Escalation::Int32 => match isa {
                Isa::Avx2 => {
                    // The AVX2 int32 branch (SIMD kernels plan Task 14): the SAME generic
                    // `align_simd_*` pipelines as the SSE4.1/NEON tiers, instantiated with
                    // `Avx2I32` (8 x i32 lanes, 256-bit). Selected in preference to SSE4.1 when
                    // `detect_isa` finds AVX2. On any non-x86_64 target this arm is otherwise
                    // unreachable (`detect_isa` never returns `Isa::Avx2` there), but the fallback
                    // keeps the match exhaustive and the return type consistent.
                    #[cfg(target_arch = "x86_64")]
                    {
                        use crate::align::GapMode;
                        use avx2::Avx2I32;
                        let (scratch, striped) = self.avx2_scratch();
                        match scoring.gap_mode() {
                            GapMode::Linear => unsafe {
                                run_avx2_linear::<Avx2I32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Affine => unsafe {
                                run_avx2_affine::<Avx2I32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Convex => unsafe {
                                run_avx2_convex::<Avx2I32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
                Isa::Sse41 => {
                    // The int32 vectorized branches (SIMD kernels plan Task 11): the SAME
                    // `align_sse41_*` pipelines as the int16 tier below, instantiated with
                    // `Sse41I32` (4 x i32 lanes) instead of `Sse41I16` (8 x i16 lanes) — used
                    // whenever `escalate` finds the worst-case score too negative for int16 but
                    // still within int32's range. On any non-x86_64 target this arm is otherwise
                    // unreachable (`detect_isa` never returns `Isa::Sse41` there), but the fallback
                    // keeps the match exhaustive and the function's return type consistent.
                    #[cfg(target_arch = "x86_64")]
                    {
                        use crate::align::GapMode;
                        use sse41::Sse41I32;
                        let (scratch, striped) = self.sse41_scratch();
                        match scoring.gap_mode() {
                            GapMode::Linear => unsafe {
                                run_sse41_linear::<Sse41I32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Affine => unsafe {
                                run_sse41_affine::<Sse41I32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Convex => unsafe {
                                run_sse41_convex::<Sse41I32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
                Isa::Neon => {
                    // The aarch64 NEON int32 branch (SIMD kernels plan Task 12): the SAME
                    // `align_simd_*` pipelines as the SSE4.1 tier, instantiated with `NeonI32`
                    // (4 x i32 lanes). On any non-aarch64 target this arm is otherwise unreachable
                    // (`detect_isa` never returns `Isa::Neon` there), but the fallback keeps the
                    // match exhaustive and the function's return type consistent.
                    #[cfg(target_arch = "aarch64")]
                    {
                        use crate::align::GapMode;
                        use neon::NeonI32;
                        let (scratch, striped) = self.neon_i32_scratch();
                        match scoring.gap_mode() {
                            GapMode::Linear => unsafe {
                                run_neon_linear::<NeonI32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Affine => unsafe {
                                run_neon_affine::<NeonI32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Convex => unsafe {
                                run_neon_convex::<NeonI32>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
                // No usable vectorized ISA on this host: delegate to the scalar `SisdEngine`. Note:
                // when `self.band` is `Some`, this silently returns the EXACT (unbanded) result
                // instead of a banded one — safe because exact >= banded (never wrong, just not
                // sped up).
                Isa::None => self.inner.align(seq, graph),
            },
            Escalation::Int16 => match isa {
                Isa::Avx2 => {
                    // The AVX2 int16 branch (SIMD kernels plan Task 14): the SAME generic
                    // `align_simd_*` pipelines as the SSE4.1/NEON tiers, instantiated with
                    // `Avx2I16` (16 x i16 lanes, 256-bit) — the widest x86_64 kernel, selected in
                    // preference to SSE4.1 when `detect_isa` finds AVX2. On any non-x86_64 target
                    // this arm is otherwise unreachable, but the fallback keeps the match
                    // exhaustive and the return type consistent.
                    #[cfg(target_arch = "x86_64")]
                    {
                        use crate::align::GapMode;
                        use avx2::Avx2I16;
                        let (scratch, striped) = self.avx2_scratch();
                        match scoring.gap_mode() {
                            GapMode::Linear => unsafe {
                                run_avx2_linear::<Avx2I16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Affine => unsafe {
                                run_avx2_affine::<Avx2I16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Convex => unsafe {
                                run_avx2_convex::<Avx2I16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
                Isa::Sse41 => {
                    // The real vectorized branches: int16 SSE4.1 handles linear-gap NW/SW/OV (SIMD
                    // kernels plan Tasks 7-8), affine-gap NW/SW/OV (Tasks 9a-9b), and convex-gap
                    // NW/SW/OV (Tasks 10a-10b) here — completing the full SSE4.1 int16 engine (all
                    // 9 type x mode combinations). On any non-x86_64 target this arm is otherwise
                    // unreachable (`detect_isa` never returns `Isa::Sse41` there), but the fallback
                    // keeps the match exhaustive and the function's return type consistent.
                    #[cfg(target_arch = "x86_64")]
                    {
                        use crate::align::GapMode;
                        use sse41::Sse41I16;
                        let (scratch, striped) = self.sse41_scratch();
                        match scoring.gap_mode() {
                            GapMode::Linear => unsafe {
                                run_sse41_linear::<Sse41I16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Affine => unsafe {
                                run_sse41_affine::<Sse41I16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Convex => unsafe {
                                run_sse41_convex::<Sse41I16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
                Isa::Neon => {
                    // The aarch64 NEON int16 branch (SIMD kernels plan Task 12): the SAME
                    // `align_simd_*` pipelines as the SSE4.1 tier, instantiated with `NeonI16`
                    // (8 x i16 lanes) — the native aarch64 vectorized path (no Rosetta). On any
                    // non-aarch64 target this arm is otherwise unreachable, but the fallback keeps
                    // the match exhaustive and the return type consistent.
                    #[cfg(target_arch = "aarch64")]
                    {
                        use crate::align::GapMode;
                        use neon::NeonI16;
                        let (scratch, striped) = self.neon_i16_scratch();
                        match scoring.gap_mode() {
                            GapMode::Linear => unsafe {
                                run_neon_linear::<NeonI16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Affine => unsafe {
                                run_neon_affine::<NeonI16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                            GapMode::Convex => unsafe {
                                run_neon_convex::<NeonI16>(
                                    alignment_type,
                                    scoring,
                                    seq,
                                    graph,
                                    scratch,
                                    striped,
                                    band_state.as_mut(),
                                )
                            },
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
                // No usable vectorized ISA on this host: delegate to the scalar `SisdEngine`. Note:
                // when `self.band` is `Some`, this silently returns the EXACT (unbanded) result
                // instead of a banded one — safe because exact >= banded (never wrong, just not
                // sped up).
                Isa::None => self.inner.align(seq, graph),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    /// Builds a small linear graph (a single sequence with no branches) from `seed`, mirroring the
    /// `sisd.rs` test helpers.
    fn linear_graph(seed: &[u8]) -> Graph {
        let mut graph = Graph::new();
        graph.add_alignment_weight(&[], seed, 1).unwrap();
        graph
    }

    /// Asserts a freshly-built [`SimdEngine`] and [`SisdEngine`] (identical `alignment_type`/
    /// `scoring`) return the exact same `(Alignment, i32)` for `seq`/`graph`.
    fn assert_matches_sisd(
        alignment_type: AlignmentType,
        scoring: Scoring,
        seq: &[u8],
        graph: &Graph,
    ) {
        let mut simd_engine = SimdEngine::new(alignment_type, scoring);
        let mut sisd_engine = SisdEngine::new(alignment_type, scoring);

        let simd_result = simd_engine.align(seq, graph);
        let sisd_result = sisd_engine.align(seq, graph);

        assert_eq!(simd_result, sisd_result);
    }

    /// Locks in the API shape and the "must equal SISD" contract: a freshly-built [`SimdEngine`]
    /// must return the exact same `(Alignment, i32)` as a [`SisdEngine`] built with identical
    /// parameters, for the same input. This is trivially true today (pure delegation), but real
    /// kernels landing in later tasks must keep it true.
    #[test]
    fn simd_engine_matches_sisd_engine_on_tiny_input() {
        let alignment_type = AlignmentType::Global;
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let seq = b"ACGT";
        let graph = Graph::new();

        let mut simd_engine = SimdEngine::new(alignment_type, scoring);
        let mut sisd_engine = SisdEngine::new(alignment_type, scoring);

        let simd_result = simd_engine.align(seq, &graph);
        let sisd_result = sisd_engine.align(seq, &graph);

        assert_eq!(simd_result, sisd_result);
    }

    /// Default-ish scoring, over a handful of alignment types: exercises the (very likely) int16
    /// branch.
    #[test]
    fn simd_engine_matches_sisd_engine_with_default_scoring() {
        let graph = linear_graph(b"ACGTACGTAC");
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();

        for alignment_type in [
            AlignmentType::Local,
            AlignmentType::Global,
            AlignmentType::Overlap,
        ] {
            assert_matches_sisd(alignment_type, scoring, b"ACGTTCGTAC", &graph);
        }
    }

    /// A mid-range affine scoring set, still comfortably within the int16 tier.
    #[test]
    fn simd_engine_matches_sisd_engine_with_mid_range_affine_scoring() {
        let graph = linear_graph(b"GATTACAGATTACA");
        let scoring = Scoring::new(3, -3, -5, -2, -5, -2).unwrap();

        assert_matches_sisd(AlignmentType::Local, scoring, b"GATTACAGATTAA", &graph);
    }

    /// A large-penalty scoring set (`i8::MIN`-adjacent gap penalties over a moderately sized
    /// graph/sequence) whose worst-case score overflows int16, forcing the int32 branch. Confirms
    /// `escalate` actually selects [`Escalation::Int32`] here (not just that the delegation
    /// happens to match, which it always would).
    #[test]
    fn simd_engine_matches_sisd_engine_when_escalation_forces_int32_branch() {
        let seq_len = 300usize;
        let node_count = 300usize;
        let scoring = Scoring::new(127, -128, -128, -128, -128, -128).unwrap();

        assert_eq!(escalate(&scoring, seq_len, node_count), Escalation::Int32);

        let seed = vec![b'A'; node_count];
        let graph = linear_graph(&seed);
        let seq = vec![b'C'; seq_len];

        assert_matches_sisd(AlignmentType::Global, scoring, &seq, &graph);
    }

    /// A worst case that overflows even int32 falls back to [`Escalation::Fallback`] without
    /// panicking. Constructing an actual graph/sequence large enough to organically reach this
    /// tier (worst case beyond roughly ±2 billion) would require on the order of tens of millions
    /// of graph nodes/sequence bases, which isn't practical for a unit test; instead this asserts
    /// the branch predicate directly, per the task brief's fallback guidance.
    #[test]
    fn escalation_predicate_selects_fallback_before_i32_overflow() {
        let scoring = Scoring::new(127, -128, -128, -128, -128, -128).unwrap();
        let huge_node_count = 20_000_000usize;
        let huge_seq_len = 20_000_000usize;

        assert_eq!(
            escalate(&scoring, huge_seq_len, huge_node_count),
            Escalation::Fallback
        );
    }

    /// An empty graph returns `(vec![], 0)` and does not panic, regardless of sequence content.
    #[test]
    fn empty_graph_returns_empty_alignment_and_zero_score() {
        let alignment_type = AlignmentType::Global;
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let graph = Graph::new();

        let mut engine = SimdEngine::new(alignment_type, scoring);
        let (alignment, score) = engine.align(b"ACGT", &graph);

        assert_eq!(alignment, Vec::new());
        assert_eq!(score, 0);
    }

    /// An empty sequence returns `(vec![], 0)` and does not panic, regardless of graph content.
    #[test]
    fn empty_sequence_returns_empty_alignment_and_zero_score() {
        let alignment_type = AlignmentType::Global;
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        let graph = linear_graph(b"ACGT");

        let mut engine = SimdEngine::new(alignment_type, scoring);
        let (alignment, score) = engine.align(b"", &graph);

        assert_eq!(alignment, Vec::new());
        assert_eq!(score, 0);
    }

    /// `detect_isa` always returns one of the defined variants (trivially true by exhaustive
    /// match, but this also documents that no branch is expected to panic on the test host) and,
    /// whichever ISA it picks, `align` still matches [`SisdEngine`] end to end (every branch
    /// currently delegates).
    #[test]
    fn detect_isa_returns_a_defined_variant_and_align_still_matches_sisd() {
        let isa = detect_isa();
        assert!(matches!(
            isa,
            Isa::Avx2 | Isa::Sse41 | Isa::Neon | Isa::None
        ));

        let graph = linear_graph(b"ACGTACGTAC");
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        assert_matches_sisd(AlignmentType::Global, scoring, b"ACGTTCGTAC", &graph);
    }

    /// The `SPOARS_FORCE_ISA` downgrade decision: only the literal `sse41` (any case) requests
    /// suppressing AVX2; unset/empty/other values leave normal detection in place. Tests the pure
    /// helper directly so the decision is covered without an AVX2 CPU.
    #[test]
    fn should_force_sse41_only_matches_the_sse41_token() {
        assert!(should_force_sse41(Some("sse41")));
        assert!(should_force_sse41(Some("SSE41")));
        assert!(should_force_sse41(Some("Sse41")));
        assert!(!should_force_sse41(None));
        assert!(!should_force_sse41(Some("")));
        assert!(!should_force_sse41(Some("avx2")));
        assert!(!should_force_sse41(Some("sse4.1")));
        assert!(!should_force_sse41(Some("neon")));
    }

    /// Task 9 — the `align_simd_*` band guard. When the banded fill finds no reachable Global
    /// endpoint (`max_score == NEG_INF`), the pipeline must short-circuit to an empty alignment +
    /// `NEG_INF` WITHOUT running the backtrack (whose `debug_assert h.get == max_score` would pass
    /// vacuously on the sentinel, then walk into out-of-band cells). `align`'s public path does not
    /// yet plumb a band (it always passes `None`), so the guard is validated at the
    /// `align_simd_linear` entry point with the same hand-built all-out-of-band band as the fill
    /// test `banded_global_returns_sentinel_when_column_l_out_of_band`.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn banded_global_guard_returns_empty_alignment_and_neg_inf() {
        #[cfg(target_arch = "aarch64")]
        type TestSimd = crate::align::simd::neon::NeonI16;
        #[cfg(target_arch = "x86_64")]
        type TestSimd = crate::align::simd::sse41::Sse41I16;

        let seq = b"ACGTTGCAGATCCGTAAGCTTACGGATCAGTTCAGGATCACGTTGCAA";
        let scoring = Scoring::new(5, -4, -8, -6, -10, -4).unwrap();
        // A short (4-node) graph against the long query keeps the adaptive band near the left edge,
        // so no sink reaches column L (mirrors the fill test of the same name).
        let graph = linear_graph(b"ACGT");
        let n = graph.num_nodes();

        let mut seeded = ScalarInit::default();
        let mut striped =
            StripedBuffers::<<TestSimd as crate::align::simd::lanes::Simd>::Vec>::default();
        // anchor = L - R = 0 for every node: no sink reaches column L, so the fill returns NEG_INF.
        let mut band = BandState {
            r: vec![seq.len() as u32; n],
            best_col: vec![0; n],
            w: 1,
        };

        let (alignment, score) = align_simd_linear::<TestSimd>(
            AlignmentType::Global,
            scoring,
            seq,
            &graph,
            &mut seeded,
            &mut striped,
            Some(&mut band),
        );

        assert!(
            alignment.is_empty(),
            "guard must return the empty alignment"
        );
        assert_eq!(
            score, NEG_INF,
            "guard must return the NEG_INF sentinel score"
        );
    }

    /// Task 10 (Gate B) — end-to-end proof that `SimdEngine::banded` + the `align()` band dispatch
    /// are wired correctly: over a small family of near-identical reads with the default band, the
    /// banded engine reproduces the exact (`SimdEngine::new`) engine's `(Alignment, score)` for
    /// every read. No near-identical read needs an out-of-band indel, so the in-band optimum equals
    /// the exact optimum. Uses the consumer's convex scoring (`Scoring::spoa_default`).
    #[test]
    fn banded_engine_matches_exact_on_near_identical_family() {
        let alignment_type = AlignmentType::Global;
        let scoring = Scoring::spoa_default(); // convex — the consumer's path
        let family: [&[u8]; 4] = [
            b"ACGTACGTACGTACGTACGT",
            b"ACGTACGTATGTACGTACGT", // one substitution
            b"ACGTACGTACGTACCTACGT", // one substitution
            b"ACGTACGAACGTACGTACGT", // one substitution
        ];

        // Build the shared graph once with an exact engine.
        let mut graph = Graph::new();
        let mut builder = SimdEngine::new(alignment_type, scoring);
        for read in family {
            crate::align::align_and_add(&mut graph, &mut builder, read, 1).unwrap();
        }

        let mut exact = SimdEngine::new(alignment_type, scoring);
        let mut banded = SimdEngine::banded(alignment_type, scoring, BandConfig::default());
        for read in family {
            assert_eq!(
                banded.align(read, &graph),
                exact.align(read, &graph),
                "banded must equal exact for in-band near-identical read {read:?}"
            );
        }
    }

    /// Task 10 (Gate B) — pins the documented heuristic contract: a query with an indel run WIDER
    /// than a tiny band (`base = 2, frac = 0.0`) is missed by the banded engine (its score differs
    /// from — and is no better than — the exact score), yet the banded engine still returns a
    /// STRUCTURALLY VALID alignment (no panic, every emitted node/query index is `-1` or in range).
    #[test]
    fn banded_engine_documents_large_indel_miss() {
        let alignment_type = AlignmentType::Global;
        let scoring = Scoring::spoa_default();
        let graph = linear_graph(b"ACGTACGTACGTACGTACGT"); // 20 bp

        // Same base with a 12-base run inserted in the middle — far wider than a w=2 band.
        let query = b"ACGTACGTACTTTTTTTTTTTTGTACGTACGT";

        let mut exact = SimdEngine::new(alignment_type, scoring);
        let mut banded =
            SimdEngine::banded(alignment_type, scoring, BandConfig { base: 2, frac: 0.0 });

        let (_exact_alignment, exact_score) = exact.align(query, &graph);
        let (banded_alignment, banded_score) = banded.align(query, &graph);

        // The tiny band cannot represent the wide indel, so it cannot reach the exact optimum.
        assert_ne!(
            banded_score, exact_score,
            "a w=2 band must miss the wide-indel optimum (documented heuristic miss)"
        );
        assert!(
            banded_score <= exact_score,
            "a banded search is a subset of the exact search, so it can never beat it"
        );

        // ...but the returned alignment must still be structurally valid (no garbage indices).
        for &(node_idx, query_idx) in &banded_alignment {
            assert!(
                node_idx == -1 || (node_idx >= 0 && (node_idx as usize) < graph.num_nodes()),
                "node index {node_idx} out of range"
            );
            assert!(
                query_idx == -1 || (query_idx >= 0 && (query_idx as usize) < query.len()),
                "query index {query_idx} out of range"
            );
        }
    }
}
