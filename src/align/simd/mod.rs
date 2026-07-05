//! SIMD-accelerated alignment engine.
//!
//! This module will eventually hold hand-tuned, per-ISA (SSE4.1 / NEON / AVX2) vectorized DP-fill
//! kernels for [`crate::align::AlignmentEngine`], dispatched at runtime and validated bit-for-bit
//! against the scalar [`crate::align::SisdEngine`] (which is their in-process oracle — see the
//! SIMD kernels plan's Global Constraints). For now [`SimdEngine`] delegates `align` entirely to
//! an internal `SisdEngine`, so it is correct (if not yet fast) from this first commit; later
//! tasks replace the delegation with real vectorized kernels one gap-mode/ISA at a time.
//!
//! `unsafe` is confined to this module (and its submodules, once they exist): the crate root uses
//! `#![deny(unsafe_code)]` rather than `#![forbid(...)]` specifically so that this module can
//! reopen it with `#![allow(unsafe_code)]` below (a crate-level `forbid` cannot be relaxed by an
//! inner `allow` — rustc E0453). No `unsafe` is used yet; the allow is here in advance for the
//! hand-written intrinsics later tasks add.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
mod avx2;
mod fill;
mod lanes;
#[cfg(target_arch = "aarch64")]
mod neon;
mod profile;
#[cfg(target_arch = "x86_64")]
mod sse41;

use crate::align::{Alignment, AlignmentEngine, AlignmentType, Scoring, SisdEngine};
use crate::graph::Graph;

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

/// Detects the best available vectorized [`Isa`] on the current CPU at runtime.
///
/// `is_x86_feature_detected!`/`is_aarch64_feature_detected!` are only defined by `std` on their
/// respective architectures, so each branch is behind a matching `#[cfg(target_arch = ...)]` —
/// this keeps the function compiling (and returning [`Isa::None`]) on every other target rather
/// than failing to compile at all.
fn detect_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
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
}

impl SimdEngine {
    /// Builds a [`SimdEngine`] for the given alignment type and scoring, mirroring
    /// [`SisdEngine::new`]'s signature.
    pub fn new(alignment_type: AlignmentType, scoring: Scoring) -> SimdEngine {
        SimdEngine {
            alignment_type,
            scoring,
            inner: SisdEngine::new(alignment_type, scoring),
        }
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
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn align_simd_linear<S>(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
) -> (Alignment, i32)
where
    S: lanes::Simd,
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    use crate::align::backtrack::backtrack_linear;
    use profile::{
        build_masks, build_penalties, build_profile, destripe_interior, seed_scalar_buffers,
        ElemFromI32,
    };

    let mut seeded = seed_scalar_buffers(graph, seq, scoring, alignment_type);
    let simd_profile = build_profile::<S>(graph, seq, scoring);
    let masks = build_masks::<S>(S::NEG_INF);
    let penalties = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.g)));

    let (striped_h, max_i, max_j, max_score) = fill::fill_linear::<S>(
        graph,
        seq.len(),
        scoring,
        alignment_type,
        &seeded,
        &simd_profile,
        &masks,
        &penalties,
    );

    // Destripe only the interior rows (rows 1..); row 0 of `striped_h` is the boundary already in
    // `seeded.h`, which `destripe_interior` never touches.
    let matrix_width_vecs = seq.len().div_ceil(S::LANES);
    destripe_interior::<S>(
        &mut seeded.h,
        &striped_h[matrix_width_vecs..],
        matrix_width_vecs,
        seq.len(),
    );

    let alignment = backtrack_linear(
        graph,
        &seeded.node_id_to_rank,
        &seeded.sequence_profile,
        &seeded.h,
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
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn align_simd_affine<S>(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
) -> (Alignment, i32)
where
    S: lanes::Simd,
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    use crate::align::backtrack::backtrack_affine;
    use profile::{
        build_masks, build_penalties, build_profile, destripe_interior, seed_scalar_buffers,
        ElemFromI32,
    };

    let mut seeded = seed_scalar_buffers(graph, seq, scoring, alignment_type);
    let simd_profile = build_profile::<S>(graph, seq, scoring);
    let masks = build_masks::<S>(S::NEG_INF);
    let penalties = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.e)));

    let (striped_h, striped_e, striped_f, max_i, max_j, max_score) = fill::fill_affine::<S>(
        graph,
        seq.len(),
        scoring,
        alignment_type,
        &seeded,
        &simd_profile,
        &masks,
        &penalties,
    );

    // Destripe only the interior rows (rows 1..); row 0 is the boundary already in the seeded
    // buffers, which `destripe_interior` never touches.
    let matrix_width_vecs = seq.len().div_ceil(S::LANES);
    destripe_interior::<S>(
        &mut seeded.h,
        &striped_h[matrix_width_vecs..],
        matrix_width_vecs,
        seq.len(),
    );
    destripe_interior::<S>(
        &mut seeded.e,
        &striped_e[matrix_width_vecs..],
        matrix_width_vecs,
        seq.len(),
    );
    destripe_interior::<S>(
        &mut seeded.f,
        &striped_f[matrix_width_vecs..],
        matrix_width_vecs,
        seq.len(),
    );

    let alignment = backtrack_affine(
        graph,
        &seeded.node_id_to_rank,
        &seeded.sequence_profile,
        &seeded.h,
        &seeded.e,
        &seeded.f,
        seeded.matrix_width,
        alignment_type,
        &scoring,
        max_i,
        max_j,
        max_score,
    );
    (alignment, max_score)
}

/// Runs the vectorized **convex-gap** fill pipeline for `seq` against `graph`, returning the same
/// `(Alignment, i32)` a [`SisdEngine`] would (the SIMD kernels plan's bit-exactness contract).
/// Wired for all three [`AlignmentType`]s — Global/NW (SIMD kernels plan Task 10a) and Local/SW +
/// Overlap/OV (Task 10b), whose per-type max-tracking branches in [`fill::fill_convex`] mirror
/// [`fill::fill_linear`]/[`fill::fill_affine`]'s (proven in Tasks 8 and 9b). Generic over the lane
/// backend `S` exactly as [`align_simd_linear`] is (SSE4.1 or NEON, int16 or int32).
///
/// The pipeline mirrors [`align_simd_affine`] but adds the SECOND affine function's matrices: it
/// destripes all five of H, E, F, **O and Q** over the seeded buffers and backtracks via
/// [`crate::align::backtrack::backtrack_convex`]. Two prefix-max penalty ladders are built — one
/// from the first extend `e` (for the `E` ladder) and one from the second extend `c` (for the `Q`
/// ladder), matching upstream (`simd_alignment_engine_implementation.hpp:1559-1565`).
///
/// Only reached after the caller's runtime ISA feature check selected an ISA whose `target_feature`
/// code inside `S` is therefore sound (see [`align_simd_linear`]).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn align_simd_convex<S>(
    alignment_type: AlignmentType,
    scoring: Scoring,
    seq: &[u8],
    graph: &Graph,
) -> (Alignment, i32)
where
    S: lanes::Simd,
    S::Elem: profile::ElemFromI32 + profile::ElemToI32,
{
    use crate::align::backtrack::backtrack_convex;
    use profile::{
        build_masks, build_penalties, build_profile, destripe_interior, seed_scalar_buffers,
        ElemFromI32,
    };

    let mut seeded = seed_scalar_buffers(graph, seq, scoring, alignment_type);
    let simd_profile = build_profile::<S>(graph, seq, scoring);
    let masks = build_masks::<S>(S::NEG_INF);
    // Two ladders: the first affine's E uses the extend `e`, the second affine's Q uses `c`.
    let penalties_e = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.e)));
    let penalties_c = build_penalties::<S>(S::Elem::from_i32(i32::from(scoring.c)));

    let (striped_h, striped_e, striped_f, striped_o, striped_q, max_i, max_j, max_score) =
        fill::fill_convex::<S>(
            graph,
            seq.len(),
            scoring,
            alignment_type,
            &seeded,
            &simd_profile,
            &masks,
            &penalties_e,
            &penalties_c,
        );

    // Destripe only the interior rows (rows 1..); row 0 is the boundary already in the seeded
    // buffers, which `destripe_interior` never touches.
    let matrix_width_vecs = seq.len().div_ceil(S::LANES);
    for (dst, striped) in [
        (&mut seeded.h, &striped_h),
        (&mut seeded.e, &striped_e),
        (&mut seeded.f, &striped_f),
        (&mut seeded.o, &striped_o),
        (&mut seeded.q, &striped_q),
    ] {
        destripe_interior::<S>(
            dst,
            &striped[matrix_width_vecs..],
            matrix_width_vecs,
            seq.len(),
        );
    }

    let alignment = backtrack_convex(
        graph,
        &seeded.node_id_to_rank,
        &seeded.sequence_profile,
        &seeded.h,
        &seeded.e,
        &seeded.f,
        &seeded.o,
        &seeded.q,
        seeded.matrix_width,
        alignment_type,
        &scoring,
        max_i,
        max_j,
        max_score,
    );
    (alignment, max_score)
}

impl AlignmentEngine for SimdEngine {
    /// Aligns `seq` against `graph`.
    ///
    /// Ports `spoa::SimdAlignmentEngine::Align`'s routing
    /// (`simd_alignment_engine_implementation.hpp:653-672`): an empty graph or empty sequence
    /// short-circuits to `(Alignment::new(), 0)` before any kernel selection (this also sidesteps
    /// divide-by-`LANES` / empty-rank indexing a real kernel would otherwise have to guard); then
    /// [`escalate`] picks the int16/int32/fallback tier and [`detect_isa`] picks the ISA. The
    /// SSE4.1 (x86_64) and NEON (aarch64) branches of both tiers run a real vectorized kernel via
    /// the SAME generic `align_simd_*` pipelines (SIMD kernels plan Tasks 7-12): SSE4.1 with
    /// [`sse41::Sse41I16`]/[`sse41::Sse41I32`], NEON with `neon::NeonI16`/`neon::NeonI32`. The
    /// remaining `(Escalation, Isa)` combinations (AVX2, no ISA) still delegate to the internal
    /// [`SisdEngine`] for now (later tasks replace each remaining `// TODO(SIMD Task N+)` marker
    /// with a real `fill_*::<Kernel>` call), so this is correct, if not yet fast on those ISAs, by
    /// construction: it always returns exactly what `SisdEngine::align` would.
    fn align(&mut self, seq: &[u8], graph: &Graph) -> (Alignment, i32) {
        if graph.nodes.is_empty() || seq.is_empty() {
            return (Alignment::new(), 0);
        }

        let escalation = escalate(&self.scoring, seq.len(), graph.nodes.len());
        let isa = detect_isa();

        match escalation {
            // Worst case overflows even i32: no vectorized kernel is safe at any ISA.
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
                        match self.scoring.gap_mode() {
                            GapMode::Linear => align_simd_linear::<Avx2I32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Affine => align_simd_affine::<Avx2I32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Convex => align_simd_convex::<Avx2I32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
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
                        match self.scoring.gap_mode() {
                            GapMode::Linear => align_simd_linear::<Sse41I32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Affine => align_simd_affine::<Sse41I32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Convex => align_simd_convex::<Sse41I32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
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
                        match self.scoring.gap_mode() {
                            GapMode::Linear => align_simd_linear::<NeonI32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Affine => align_simd_affine::<NeonI32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Convex => align_simd_convex::<NeonI32>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
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
                        match self.scoring.gap_mode() {
                            GapMode::Linear => align_simd_linear::<Avx2I16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Affine => align_simd_affine::<Avx2I16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Convex => align_simd_convex::<Avx2I16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
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
                        match self.scoring.gap_mode() {
                            GapMode::Linear => align_simd_linear::<Sse41I16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Affine => align_simd_affine::<Sse41I16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Convex => align_simd_convex::<Sse41I16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
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
                        match self.scoring.gap_mode() {
                            GapMode::Linear => align_simd_linear::<NeonI16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Affine => align_simd_affine::<NeonI16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                            GapMode::Convex => align_simd_convex::<NeonI16>(
                                self.alignment_type,
                                self.scoring,
                                seq,
                                graph,
                            ),
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.inner.align(seq, graph)
                    }
                }
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
}
