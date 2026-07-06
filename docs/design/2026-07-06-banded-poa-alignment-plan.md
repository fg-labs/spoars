# Banded POA Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, heuristic, abPOA-style adaptive-banded alignment mode to `SimdEngine` that computes only a per-node query-column window of the DP matrix, trading optimality for speed on near-identical short-read families.

**Architecture:** A new `band.rs` module owns all band geometry (config, remaining-path `R` pass, per-node `[beg,end)` window, segment range, `best_col` propagation) as pure, unit-testable functions. The existing striped `fill_{linear,affine,convex}` gain an optional per-row clip that iterates `beg_sn..end_sn`, closes the *horizontal* band-edge carries to `NEG_INF`, seeds the *diagonal* carry from the predecessor buffer, and uses saturating penalty adds so interior out-of-band sentinels can never wrap and win a `max`. `SimdEngine` gains `band: Option<BandConfig>` and a `banded()` constructor; `align()` dispatches `None` → the untouched exact pipeline, `Some` → the banded pipeline with a `max_score == NEG_INF` guard. The scalar `SisdEngine` and `SimdEngine::new` are never touched, so bit-exactness with spoa is preserved on the default path.

**Tech Stack:** Rust (edition per `rust-toolchain.toml`), `std::arch` SIMD intrinsics (SSE4.1/AVX2/NEON) behind the per-ISA `Simd` trait in `src/align/simd/lanes.rs`, `proptest` for property tests, `criterion` for benches. All work is in the `nh_banded-poa` worktree at the bench scratchpad path; source paths below are repo-relative.

## Global Constraints

- **Design source of truth:** `docs/design/2026-07-06-banded-poa-alignment-design.md`. Every task implements a part of it; when in doubt, that spec (including its two-pass "Adversarial review" trail) wins.
- **Default path is sacred:** `SimdEngine::new`, `SisdEngine`, and every existing parity test (`simd_parity`, `engine_parity`, `cli_parity`, lib unit tests, graph/proptests) must stay green and bit-exact after every task. Banding is reached **only** through `SimdEngine::banded`.
- **Rust conventions (Fulcrum):** `rustfmt` max_width=100; `clippy --all-features --all-targets -- -D warnings` clean; `forbid(unsafe_code)` stays at crate level except inside the already-`unsafe` per-ISA backends; prefer borrows and `&[T]`; side-effecting fns return `Result` where applicable; doc comments on all new public items and non-trivial private ones (explain *why*).
- **Determinism across ISAs:** any band-affecting reduction (`best_col`, `R` tie-breaks) must be `LANES`-independent — reuse the `index_of` flat-scan convention (`fill.rs:87`, "lowest column achieving the max"), never a lane-order argmax.
- **No AI attribution** in any commit message. Conventional-commit prefixes (`feat:`, `test:`, `refactor:`, `docs:`). Sign commits (`git commit -S`).
- **Sentinel constant is shared:** `NEG_INF = i16::MIN + 1024` / `i32::MIN + 1024` (`lanes.rs:182,188`) is depended on by `seed_striped_row0` and `escalate`. Do **not** redefine it per-tier.
- **TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, run the full suite, commit.

**Commands** (run from the worktree root):
- Build/lint: `cargo build` · `cargo clippy --all-features --all-targets -- -D warnings` · `cargo fmt --check`
- Full suite: `cargo test`
- One test: `cargo test --lib <name> -- --exact` or `cargo test --test <suite> <name>`
- Force scalar oracle: `SPOARS_FORCE_SISD=1 cargo test`

## Current-code anchors (read before starting)

- `src/align/simd/lanes.rs` — `trait Simd` (`:31`): `Elem`, `Vec`, `LANES`, `LOG_LANES`, `LSS`, `RSS`, `NEG_INF`, and methods `splat/add/sub/min/max/or/loadu/storeu/store_widened_i32/slli/srli/slli_one_lane/srli_top_lane/horizontal_max/prefix_max/prefix_max_step`. `add`/`sub` are **wrapping** (`:60-68`). Eight impls: scalar `i16`/`i32` (oracle, `LANES=1`, `:196`,`:304`), plus `Sse41I16/Sse41I32` (`sse41.rs`), `Avx2I16/Avx2I32` (`avx2.rs`), `NeonI16/NeonI32` (`neon.rs`).
- `src/align/simd/fill.rs` — `fill_linear` (`:158`), `fill_affine` (`:361`), `fill_convex` (`:572`), each `<S: Simd>(graph, seq_len, scoring, alignment_type, seeded: &ScalarInit, profile, masks, penalties, striped_h: &mut Vec<S::Vec>) -> (max_i, max_j, max_score)`. Helpers: `value_at` (`:44`), `row_max` (`:69`), `index_of` (`:87`), `seed_striped_row0` (`:117`). Per-row loop over `graph.rank_to_node` starts `:219`. Diagonal carry `x` seeded from `first_column(pred_i)` (`:232`,`:249`); horizontal carry from `first_column(i)+g` (`:263`); `carry_mask = masks[LOG_LANES]` (`:182`).
- `src/align/simd/mod.rs` — `struct SimdEngine { alignment_type, scoring, inner, scratch, striped }` (`:286`); `new` (`:329`); per-ISA scratch splitters (`:344`+); `align_simd_linear/affine/convex` (`:437/516/654`); `escalate` (`:171`); `impl AlignmentEngine for SimdEngine::align` dispatch (`:872`, routes on `escalate` tier × `detect_isa` × `GapMode`).
- `src/align/backtrack.rs` — `trait CellRead { fn get(&self, i, j) -> i32 }`, `RowMajor`, `StripedView` (PR #11); `backtrack_{linear,affine,convex}_impl<V: CellRead>`.
- `src/graph.rs` — `Graph { nodes, edges, rank_to_node }`, `Node { code, inedges, outedges, ... }`, `Edge { tail, head, weight }`, `NodeId(u32)`, `EdgeId(u32)`.

---

### Task 1: `BandConfig` + saturating `width()`

Pure config type and half-width computation. No fill interaction yet. Covers spec §API and MAJOR 7 (usize saturating width).

**Files:**
- Create: `src/align/simd/band.rs`
- Modify: `src/align/simd/mod.rs` (add `mod band;` and re-export)

**Interfaces:**
- Produces: `pub struct BandConfig { pub base: u32, pub frac: f32 }`; `impl BandConfig { pub fn width(&self, query_len: usize) -> usize }`; `impl Default for BandConfig` (abPOA defaults `base=10, frac=0.01`).

- [ ] **Step 1: Write the failing test**

Add to a new `band.rs`, in a `#[cfg(test)] mod tests`:

```rust
#[test]
fn width_is_base_plus_rounded_fraction() {
    let cfg = BandConfig { base: 10, frac: 0.01 };
    assert_eq!(cfg.width(0), 10); // base only
    assert_eq!(cfg.width(235), 12); // 10 + round(2.35) = 10 + 2
    assert_eq!(cfg.width(1000), 20); // 10 + round(10.0)
}

#[test]
fn width_saturates_and_never_panics() {
    // Huge/degenerate configs must clamp, not overflow or panic (MAJOR 7).
    let huge = BandConfig { base: u32::MAX, frac: f32::MAX };
    let _ = huge.width(usize::MAX); // must not panic
    let neg = BandConfig { base: 5, frac: -1.0 };
    assert_eq!(neg.width(100), 5); // negative fraction floors to 0 contribution
    let nan = BandConfig { base: 7, frac: f32::NAN };
    assert_eq!(nan.width(100), 7); // NaN -> 0 contribution
}

#[test]
fn default_is_abpoa() {
    assert_eq!(BandConfig::default().base, 10);
    assert!((BandConfig::default().frac - 0.01).abs() < 1e-9);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib band::tests -- --exact`
Expected: FAIL — `BandConfig` not found (module not declared / type missing).

- [ ] **Step 3: Write minimal implementation**

`src/align/simd/band.rs` (top of file):

```rust
//! Band geometry for the opt-in, heuristic abPOA-style banded alignment mode.
//!
//! Everything here is pure and `LANES`-independent so it can be unit-tested without a
//! SIMD backend and produces identical bands on every ISA. See
//! `docs/design/2026-07-06-banded-poa-alignment-design.md`.

/// Adaptive-band configuration (abPOA-style). APPROXIMATE: banded alignment may miss the
/// optimal path when it needs an indel larger than the band. `SimdEngine::new` stays exact
/// (bit-exact with spoa); use this only when the speed/accuracy trade-off is acceptable
/// (near-identical reads).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandConfig {
    /// Constant half-width added to every band, in query columns.
    pub base: u32,
    /// Fraction of the query length added to the half-width (`round(frac * L)`).
    pub frac: f32,
}

impl Default for BandConfig {
    fn default() -> Self {
        BandConfig { base: 10, frac: 0.01 }
    }
}

impl BandConfig {
    /// Per-align half-width `w = base + round(frac * L)`, computed in `usize` and **saturating**
    /// so no config can overflow or panic. Negative/NaN `frac` contributes 0. A width `>= L`
    /// means "no effective band" (used only by the smoke test); production values are small.
    pub fn width(&self, query_len: usize) -> usize {
        let frac_cols = (f64::from(self.frac) * query_len as f64).round();
        // A negative or NaN product yields 0 columns; a huge product saturates at usize::MAX.
        let frac_cols = if frac_cols.is_finite() && frac_cols > 0.0 {
            frac_cols as usize // saturating float->int cast (Rust: clamps, NaN->0)
        } else {
            0
        };
        (self.base as usize).saturating_add(frac_cols)
    }
}
```

In `src/align/simd/mod.rs`, near the other `mod` declarations, add `mod band;` and `pub use band::BandConfig;`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib band::tests -- --exact`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/band.rs src/align/simd/mod.rs
git commit -S -m "feat(simd): add BandConfig with saturating width()"
```

---

### Task 2: Saturating `adds`/`subs` on the `Simd` trait

Add saturating add/sub to the lane abstraction with impls on all eight backends. int16 uses native saturating intrinsics; **x86 int32 has no native saturating add/sub and must be emulated**; NEON int32 is native. Covers spec §Safety model MAJOR 2 (pass 2).

**Files:**
- Modify: `src/align/simd/lanes.rs` (trait + 2 scalar impls + tests)
- Modify: `src/align/simd/sse41.rs`, `src/align/simd/avx2.rs`, `src/align/simd/neon.rs` (backend impls)

**Interfaces:**
- Produces: `fn adds(a: Self::Vec, b: Self::Vec) -> Self::Vec` and `fn subs(a: Self::Vec, b: Self::Vec) -> Self::Vec` on `trait Simd` — lane-wise **saturating** add/sub.

- [ ] **Step 1: Write the failing test**

Add to `lanes.rs` tests (mirroring the existing `i16_add_sub_are_non_saturating` at `:421`), asserting the new methods saturate for both scalar tiers:

```rust
#[test]
fn i16_saturating_add_sub_clamp_at_bounds() {
    // adds/subs must clamp, unlike add/sub which wrap.
    assert_eq!(<i16 as Simd>::adds(i16::MAX, 1), i16::MAX);
    assert_eq!(<i16 as Simd>::subs(i16::MIN, 1), i16::MIN);
    // NEG_INF + a large negative penalty must stay clamped, never wrap positive.
    let neg = <i16 as Simd>::NEG_INF;
    assert_eq!(<i16 as Simd>::adds(neg, -128), i16::MIN.max(neg.saturating_add(-128)));
    assert!(<i16 as Simd>::adds(neg, -128) < 0); // the property the band relies on
}

#[test]
fn i32_saturating_add_sub_clamp_at_bounds() {
    assert_eq!(<i32 as Simd>::adds(i32::MAX, 1), i32::MAX);
    assert_eq!(<i32 as Simd>::subs(i32::MIN, 1), i32::MIN);
    let neg = <i32 as Simd>::NEG_INF;
    // 9 successive -128 adds must not wrap the int32 sentinel positive.
    let mut v = neg;
    for _ in 0..9 {
        v = <i32 as Simd>::adds(v, -128);
    }
    assert!(v < 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib lanes::tests::i16_saturating -- --exact`
Expected: FAIL — no method `adds` on the trait.

- [ ] **Step 3: Write minimal implementation**

In `trait Simd` (`lanes.rs`, after `sub` at `:68`):

```rust
    /// Lane-wise **saturating** addition — clamps at the element bound instead of wrapping.
    /// Required by the banded fill: interior out-of-band `NEG_INF` sentinels can be penalized
    /// across many rows, and a wrapping add would eventually flip a sentinel positive and win a
    /// `max()`. The exact (non-banded) path never uses this (it relies on the +1024 headroom).
    fn adds(a: Self::Vec, b: Self::Vec) -> Self::Vec;

    /// Lane-wise **saturating** subtraction; see [`Simd::adds`].
    fn subs(a: Self::Vec, b: Self::Vec) -> Self::Vec;
```

Scalar impls (`lanes.rs`, in the `i16` block near `:213` and the `i32` block near `:319`):

```rust
    // i16 block:
    fn adds(a: i16, b: i16) -> i16 { a.saturating_add(b) }
    fn subs(a: i16, b: i16) -> i16 { a.saturating_sub(b) }
    // i32 block:
    fn adds(a: i32, b: i32) -> i32 { a.saturating_add(b) }
    fn subs(a: i32, b: i32) -> i32 { a.saturating_sub(b) }
```

`sse41.rs` — int16 native, int32 emulated (no `_mm_adds_epi32`). In the `Sse41I16` impl:

```rust
    #[target_feature(enable = "sse4.1")]
    fn adds(a: __m128i, b: __m128i) -> __m128i { unsafe { _mm_adds_epi16(a, b) } }
    #[target_feature(enable = "sse4.1")]
    fn subs(a: __m128i, b: __m128i) -> __m128i { unsafe { _mm_subs_epi16(a, b) } }
```

In the `Sse41I32` impl (emulate signed saturating add/sub via 64-bit widen or compare/blend; the compare/blend form below avoids widening and works on SSE4.1):

```rust
    // Signed 32-bit saturating add: r = a + b; on positive overflow clamp to INT_MAX, on
    // negative overflow clamp to INT_MIN. Overflow occurs iff a and b share a sign and r differs.
    #[target_feature(enable = "sse4.1")]
    fn adds(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let r = _mm_add_epi32(a, b);
            let sat = _mm_add_epi32(
                _mm_srli_epi32::<31>(a),
                _mm_set1_epi32(i32::MAX), // a>=0 -> INT_MAX, a<0 -> INT_MIN (INT_MAX+1)
            );
            // overflow = (a ^ r) & (b ^ r) < 0  (sign bit set)
            let overflow = _mm_and_si128(_mm_xor_si128(a, r), _mm_xor_si128(b, r));
            _mm_blendv_epi8(r, sat, overflow)
        }
    }
    #[target_feature(enable = "sse4.1")]
    fn subs(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let r = _mm_sub_epi32(a, b);
            let sat = _mm_add_epi32(_mm_srli_epi32::<31>(a), _mm_set1_epi32(i32::MAX));
            // overflow = (a ^ b) & (a ^ r) < 0
            let overflow = _mm_and_si128(_mm_xor_si128(a, b), _mm_xor_si128(a, r));
            _mm_blendv_epi8(r, sat, overflow)
        }
    }
```

`avx2.rs` — same pattern with `_mm256_*` (`_mm256_adds_epi16`/`_mm256_subs_epi16` native; int32 emulated via `_mm256_add_epi32`/`_mm256_xor_si256`/`_mm256_and_si256`/`_mm256_srli_epi32::<31>`/`_mm256_set1_epi32`/`_mm256_blendv_epi8`).

`neon.rs` — all native: `vqaddq_s16`/`vqsubq_s16` and `vqaddq_s32`/`vqsubq_s32`.

> **Implementer note:** the int32 emulation is the one non-obvious piece. Verify it against a scalar reference in Step 4's test (the `i32` scalar impl uses `saturating_add`, so the parity harness in later tasks will cross-check the vector form against it). If the compare/blend form is hard to get right, the 64-bit-widen fallback (`_mm_add_epi64` on sign-extended halves, then clamp to `[i32::MIN,i32::MAX]` and pack) is equivalent — pick whichever you can prove equal to `saturating_add` per lane.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib lanes::tests -- --exact` then the full `cargo test`.
Expected: the two new tests PASS; all existing tests still PASS (no default-path behavior changed — `adds`/`subs` are new, unused so far). Clippy clean.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/lanes.rs src/align/simd/sse41.rs src/align/simd/avx2.rs src/align/simd/neon.rs
git commit -S -m "feat(simd): add saturating adds/subs to the Simd lane trait"
```

---

### Task 3: Remaining-path `R` reverse pass + `anchor()`

Pure graph pass, precomputable before the fill. Covers spec §Band computation.1 and MINOR 8 (documented node-count bias).

**Files:**
- Modify: `src/align/simd/band.rs`

**Interfaces:**
- Consumes: `Graph`, `rank_to_node`, `node_id_to_rank` (from `ScalarInit`).
- Produces: `pub(crate) fn remaining_path(graph: &Graph, node_id_to_rank: &[u32]) -> Vec<u32>` returning `R` indexed by **rank** (`R[rank]`); `pub(crate) fn anchor(r_len: u32, query_len: usize) -> usize` = `clamp(L - R, 0, L)`.

- [ ] **Step 1: Write the failing test**

```rust
// Helper: build a linear chain graph A->B->C (3 nodes) using the crate's Graph API,
// return (graph, node_id_to_rank). Reuse whatever test builder exists in graph.rs tests;
// if none is exported, construct via Graph::new + add_alignment as other simd tests do.
#[test]
fn remaining_path_counts_heaviest_successor_chain() {
    let (graph, n2r) = linear_chain_3(); // ranks 0,1,2 in topo order
    let r = remaining_path(&graph, &n2r);
    // sink has R=0; each predecessor is 1 + successor's R.
    assert_eq!(r[2], 0);
    assert_eq!(r[1], 1);
    assert_eq!(r[0], 2);
}

#[test]
fn anchor_clamps_to_query_bounds() {
    assert_eq!(anchor(0, 235), 235); // sink -> end of query
    assert_eq!(anchor(2, 235), 233);
    assert_eq!(anchor(1000, 235), 0); // R > L clamps to 0
}

#[test]
fn remaining_path_tie_breaks_by_lowest_rank() {
    // A node with two equal-weight out-edges must pick the successor with the LOWEST rank,
    // so R is deterministic across runs/ISAs.
    let (graph, n2r) = diamond_equal_weights();
    let r = remaining_path(&graph, &n2r);
    // Assert the chosen branch is the lower-ranked one (exact values per the fixture).
    assert_eq!(r, expected_r_for_diamond());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib band::tests::remaining_path -- --exact`
Expected: FAIL — `remaining_path` not defined.

- [ ] **Step 3: Write minimal implementation**

```rust
use crate::graph::{EdgeId, Graph};

/// Remaining heaviest-support path length per rank: `R[sink]=0`, `R[n]=1+R[s*]` where `s*` is the
/// successor reached by `n`'s heaviest out-edge (max weight; ties broken by **lowest rank** for
/// cross-run/ISA determinism). Computed in reverse topological order (`rank_to_node` reversed),
/// which is well-founded on the DAG. NOTE (heuristic, MINOR 8): this counts *nodes* on the heaviest
/// path as a proxy for query *columns* remaining; indels between that path and the query bias the
/// derived anchor — a documented tradeoff, not a bug.
pub(crate) fn remaining_path(graph: &Graph, node_id_to_rank: &[u32]) -> Vec<u32> {
    let n = graph.rank_to_node.len();
    let mut r = vec![0u32; n];
    for &node_id in graph.rank_to_node.iter().rev() {
        let node = &graph.nodes[node_id.0 as usize];
        let rank = node_id_to_rank[node_id.0 as usize] as usize;
        let mut best: Option<(i64, usize, u32)> = None; // (weight desc, rank asc, r_of_succ)
        for &edge_id in &node.outedges {
            let edge = &graph.edges[edge_id.0 as usize];
            let succ_rank = node_id_to_rank[edge.head.0 as usize] as usize;
            let key = (i64::from(edge.weight), succ_rank);
            let better = match best {
                None => true,
                Some((w, rk, _)) => key.0 > w || (key.0 == w && succ_rank < rk),
            };
            if better {
                best = Some((key.0, succ_rank, r[succ_rank]));
            }
        }
        r[rank] = best.map_or(0, |(_, _, succ_r)| 1 + succ_r);
    }
    r
}

/// Query-column anchor for a node: `clamp(L - R, 0, L)`.
pub(crate) fn anchor(r_len: u32, query_len: usize) -> usize {
    query_len.saturating_sub(r_len as usize).min(query_len)
}
```

> **Implementer note:** confirm the field names `Node.outedges`, `Edge.head`, `Edge.weight`, `EdgeId(u32)`, `NodeId(u32)` against `src/graph.rs`; adjust the accessor spelling if the real API differs (e.g. `weight()` method vs field). Reuse an existing graph test-builder helper for the fixtures — do **not** commit fixture data files (generate in-test).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib band::tests -- --exact`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/band.rs
git commit -S -m "feat(simd): compute remaining-path R and query anchor for banding"
```

---

### Task 4: Per-node window `[beg,end)`, segment range, and `best_col` propagation

Pure geometry given anchor + predecessors' `best_col`. Covers spec §Band computation.2/.3, §best_col propagation, MAJOR 6 (half-open), MINOR 6 (empty-band guard), MINOR 5 (determinism).

**Files:**
- Modify: `src/align/simd/band.rs`

**Interfaces:**
- Consumes: `anchor` (usize), predecessor `best_col` values, `w` (usize), `L` (usize), `lanes` (usize), `matrix_width_vecs` (usize).
- Produces:
  - `pub(crate) fn node_window(anchor: usize, mstart: usize, mend: usize, w: usize, query_len: usize) -> (usize, usize)` → `(beg, end)` half-open.
  - `pub(crate) fn segment_range(beg: usize, end: usize, lanes: usize, matrix_width_vecs: usize) -> (usize, usize)` → `(beg_sn, end_sn)` half-open, clamped, **non-empty**.
  - A `BandState` scratch struct holding `r: Vec<u32>`, `best_col: Vec<u32>` (indexed by rank), and `w: usize`, plus a constructor `BandState::new(graph, node_id_to_rank, query_len, cfg)`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn node_window_is_union_of_anchor_and_predecessors_widened() {
    // beg = max(0, min(Mstart, anchor) - w); end = min(L, max(Mend, anchor) + w + 1)
    let (beg, end) = node_window(/*anchor*/100, /*mstart*/90, /*mend*/110, /*w*/12, /*L*/235);
    assert_eq!(beg, 90 - 12);
    assert_eq!(end, 110 + 12 + 1);
    // clamps at 0 and L
    let (b0, e0) = node_window(5, 5, 5, 12, 235);
    assert_eq!(b0, 0);
    let (_, eL) = node_window(230, 230, 230, 12, 235);
    assert_eq!(eL, 235);
}

#[test]
fn segment_range_half_open_no_off_by_one() {
    // L % LANES == 0 boundary (MAJOR 6): end==L must give end_sn==matrix_width_vecs, not +1.
    let lanes = 8;
    let mwv = 240usize.div_ceil(lanes); // 30
    let (bs, es) = segment_range(0, 240, lanes, mwv);
    assert_eq!((bs, es), (0, 30));
    // L % LANES == 1 boundary
    let mwv2 = 241usize.div_ceil(lanes); // 31
    let (_, es2) = segment_range(0, 241, lanes, mwv2);
    assert_eq!(es2, 31);
    // interior band -> floored beg, ceil end
    let (bs3, es3) = segment_range(20, 60, 8, 30);
    assert_eq!((bs3, es3), (2, 8)); // 20/8=2 ; ceil(60/8)=8
}

#[test]
fn segment_range_never_empty() {
    // MINOR 6: an empty [beg,beg) window must widen to a single non-empty segment.
    let (bs, es) = segment_range(240, 240, 8, 30);
    assert!(es > bs);
    assert!(es <= 30);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib band::tests::node_window -- --exact` and `...segment_range...`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Write minimal implementation**

```rust
/// Half-open query-column window for a node: union of its anchor with its predecessors' best
/// columns `[Mstart, Mend]`, widened by `w` on each side and clamped to `[0, L]`. A source node
/// passes `mstart = mend = anchor`.
pub(crate) fn node_window(
    anchor: usize,
    mstart: usize,
    mend: usize,
    w: usize,
    query_len: usize,
) -> (usize, usize) {
    let lo = mstart.min(anchor);
    let hi = mend.max(anchor);
    let beg = lo.saturating_sub(w);
    let end = (hi + w + 1).min(query_len);
    (beg, end)
}

/// Segment (vector-lane) range `[beg_sn, end_sn)` covering `[beg, end)` query columns, clamped to
/// the row block and guaranteed **non-empty** (MINOR 6). `beg_sn` floors, `end_sn` ceils — so the
/// effective computed band is `[beg_sn*lanes, end_sn*lanes)`; the left-edge carry closure therefore
/// happens at `beg_sn*lanes`, which unit tests must target.
pub(crate) fn segment_range(
    beg: usize,
    end: usize,
    lanes: usize,
    matrix_width_vecs: usize,
) -> (usize, usize) {
    let beg_sn = (beg / lanes).min(matrix_width_vecs.saturating_sub(1));
    let mut end_sn = end.div_ceil(lanes).min(matrix_width_vecs);
    if end_sn <= beg_sn {
        end_sn = (beg_sn + 1).min(matrix_width_vecs);
    }
    (beg_sn, end_sn)
}
```

Add `BandState`:

```rust
/// Per-align band scratch: precomputed `R` (by rank), the half-width `w`, and a `best_col` buffer
/// filled incrementally as the fill reaches each row. `best_col[rank]` is set to the query column of
/// that row's max via the `LANES`-independent `index_of` flat-scan (MINOR 5 determinism), by the fill.
pub(crate) struct BandState {
    pub(crate) r: Vec<u32>,
    pub(crate) best_col: Vec<u32>,
    pub(crate) w: usize,
}

impl BandState {
    pub(crate) fn new(
        graph: &Graph,
        node_id_to_rank: &[u32],
        query_len: usize,
        cfg: BandConfig,
    ) -> BandState {
        let r = remaining_path(graph, node_id_to_rank);
        BandState {
            best_col: vec![0; r.len()],
            r,
            w: cfg.width(query_len),
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib band::tests -- --exact`
Expected: PASS (all band unit tests).

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/band.rs
git commit -S -m "feat(simd): per-node band window, segment range, and BandState scratch"
```

---

### Task 5: Thread an optional band through the fills and `align_simd_*` (no behavior change)

Pure plumbing: add `band: Option<&mut BandState>` to `fill_*` and `align_simd_*`, defaulting `None` at every call site. `None` reproduces current behavior **exactly** — this task's whole point is that the full parity suite stays green.

**Files:**
- Modify: `src/align/simd/fill.rs` (signatures of `fill_linear/affine/convex`)
- Modify: `src/align/simd/mod.rs` (signatures of `align_simd_linear/affine/convex` + their fill calls)

**Interfaces:**
- Produces (new trailing param on each): `fill_linear<S>(.., striped_h: &mut Vec<S::Vec>, band: Option<&mut BandState>) -> (usize, usize, i32)` (same for affine/convex); `align_simd_linear<S>(alignment_type, scoring, seq, graph, seeded, striped, band: Option<&mut BandState>) -> (Alignment, i32)` (same for affine/convex).

- [ ] **Step 1: Write the failing test**

No new test — this task is defined by the *existing* suite staying green. Write a one-line sentinel test to make the intent explicit:

```rust
// in mod.rs tests
#[test]
fn banded_signature_none_matches_exact() {
    // Compiles only if align_simd_* accept Option<&mut BandState>; behavior is covered by the
    // full parity suite (unchanged when band is None).
    // (No assertion body needed beyond the suite; this documents the plumbing contract.)
}
```

- [ ] **Step 2: Run to verify current state**

Run: `cargo build`
Expected: FAILS to compile once you start editing call sites mid-way; the gate is that after Step 3 `cargo test` is fully green.

- [ ] **Step 3: Implement the plumbing**

In `fill.rs`, add `band: Option<&mut BandState>` as the last parameter of each `fill_*`. Do **not** use it yet — bind `let _ = &band;` at the top so clippy doesn't warn, or `#[allow(unused_variables)]`. Keep the loop iterating `0..matrix_width_vecs` exactly as today.

In `mod.rs`, add `band: Option<&mut BandState>` as the last parameter of each `align_simd_*`, and pass it through to the corresponding `fill_*` call. At the six `align()` dispatch call sites (`:903`+), pass `None` for now.

Import `use crate::align::simd::band::BandState;` where needed.

- [ ] **Step 4: Run the full suite to verify no behavior change**

Run: `cargo test` then `SPOARS_FORCE_SISD=1 cargo test` then `cargo clippy --all-features --all-targets -- -D warnings`
Expected: ALL green, bit-exact (this is the contract: `None` == today).

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/fill.rs src/align/simd/mod.rs
git commit -S -m "refactor(simd): thread optional BandState through fills (no-op when None)"
```

---

### Task 6: Banded clip in `fill_linear` (SW/OV) with correct carry seeding + saturating adds

The first real banded fill. Implements the per-row clip, the **horizontal** carry closure to `NEG_INF`, the **diagonal** carry seeded from the predecessor buffer, saturating adds, and band-aware SW/OV max-tracking. Global endpoint handling is Task 9. Covers spec §Fill clip (pass-2 corrected) and PRIMARY test gate §1.

**Files:**
- Modify: `src/align/simd/fill.rs` (`fill_linear`)
- Test: `src/align/simd/fill.rs` `#[cfg(test)]` (per-cell exact oracle at `beg_sn>=1`)

**Interfaces:**
- Consumes: `BandState` (Task 4), `node_window`/`segment_range` (Task 4), `Simd::adds` (Task 2), `index_of` (`fill.rs:87`).
- Produces: `fill_linear` computes exact in-band `H` when `band = Some(..)` and the optimal path stays in-band.

- [ ] **Step 1: Write the failing test**

The PRIMARY gate. Build a graph + query long enough that a deliberately narrow band forces `beg_sn >= 1` on deep rows, compute the exact matrix (band `None`), then the banded matrix, and assert **every in-band cell** matches:

```rust
#[test]
fn banded_linear_in_band_cells_match_exact_at_beg_sn_ge_1() {
    // Near-identical reads so the optimal path hugs the diagonal and stays in-band.
    let (graph, seeded, /*profile,masks,penalties setup*/ ..) = build_linear_fixture(/*len*/ 64);
    let query = fixture_query();

    // Exact:
    let mut h_exact: Vec<Scalar::Vec> = Vec::new();
    let (_, _, _) = fill_linear::<Backend>(&graph, query.len(), scoring, AlignmentType::Local,
        &seeded, &profile, &masks, &penalties, &mut h_exact, None);

    // Banded with a narrow w that guarantees beg_sn>=1 on deep rows:
    let mut band = BandState::new(&graph, &seeded.node_id_to_rank, query.len(),
        BandConfig { base: 2, frac: 0.0 });
    let mut h_band: Vec<Backend::Vec> = Vec::new();
    fill_linear::<Backend>(&graph, query.len(), scoring, AlignmentType::Local,
        &seeded, &profile, &masks, &penalties, &mut h_band, Some(&mut band));

    // Recompute the per-row [beg,end) the band used and assert in-band equality; also assert at
    // least one row actually had beg_sn>=1 (the gate must EXERCISE the risky seeding).
    assert!(any_row_had_beg_sn_ge_1(&band, &graph, &seeded, query.len(), Backend::LANES));
    for_each_in_band_cell(&band, &graph, &seeded, query.len(), Backend::LANES, |i, j| {
        assert_eq!(cell(&h_band, i, j), cell(&h_exact, i, j), "mismatch at ({i},{j})");
    });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib fill::tests::banded_linear -- --exact`
Expected: FAIL — either compile error (band unused) or mismatch (clip not implemented).

- [ ] **Step 3: Implement the clip**

In `fill_linear`, when `band = Some(state)`, inside the `for &node_id in &graph.rank_to_node` loop:

1. Compute this row's window before the segment loops:

```rust
let (beg_sn, end_sn, beg_col) = if let Some(state) = band.as_deref() {
    let rank = node_id_to_rank[node_id.0 as usize] as usize;
    let anchor = band::anchor(state.r[rank], seq_len);
    // Mstart/Mend from predecessors' best_col (source node uses anchor).
    let (mstart, mend) = if node.inedges.is_empty() {
        (anchor, anchor)
    } else {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for &e in &node.inedges {
            let pr = node_id_to_rank[graph.edges[e.0 as usize].tail.0 as usize] as usize;
            let bc = state.best_col[pr] as usize;
            lo = lo.min(bc);
            hi = hi.max(bc);
        }
        (lo, hi)
    };
    let (beg, end) = band::node_window(anchor, mstart, mend, state.w, seq_len);
    let (bs, es) = band::segment_range(beg, end, lanes, matrix_width_vecs);
    (bs, es, bs * lanes)
} else {
    (0, matrix_width_vecs, 0)
};
```

2. **Diagonal + vertical pass** — iterate `beg_sn..end_sn` instead of `0..matrix_width_vecs`. Seed the diagonal carry from the predecessor buffer at `beg_sn-1` (or `NEG_INF` when `beg_sn==0`, preserving today's `first_column` seed only for the unbanded/`beg_sn==0` case):

```rust
let mut x = if beg_sn == 0 {
    S::srli_top_lane(S::splat(S::Elem::from_i32(first_column(pred_i))))
} else {
    S::srli_top_lane(striped_h[pred_base + (beg_sn - 1)]) // real pred value, or NEG_INF if OOB
};
for j in beg_sn..end_sn {
    let h_pred = striped_h[pred_base + j];
    let t1 = S::srli_top_lane(h_pred);
    let diag = S::or(S::slli_one_lane(h_pred), x);
    x = t1;
    let value = S::max(
        S::adds(diag, profile[profile_base + j]),   // saturating (Task 2)
        S::adds(h_pred, g_vec),
    );
    striped_h[row_base + j] = value;
}
```

Apply the identical `beg_sn-1` diagonal seed to the **additional-predecessors** loop.

3. **Horizontal (prefix_max) pass** — close the horizontal carry to `NEG_INF` at `beg_sn` (today's seed is `first_column(i)+g`; under banding the left neighbor is uncomputed, so seed the sentinel), iterate `beg_sn..end_sn`, use `adds`:

```rust
let mut x = S::splat(S::NEG_INF); // horizontal edge closure (ksw2-style)
for j in beg_sn..end_sn {
    let mut hv = striped_h[row_base + j];
    hv = S::max(hv, S::or(x, carry_mask));
    hv = S::prefix_max(hv, penalties, masks);
    x = S::srli_top_lane(S::adds(hv, g_vec));
    if alignment_type == AlignmentType::Local {
        hv = S::max(hv, zeroes);
    }
    striped_h[row_base + j] = hv;
    score = S::max(score, hv);
}
```

4. **best_col + band-aware max-tracking** — after the row, for SW/OV reduce over `beg_sn..end_sn` only, and record `best_col[rank]` via the `LANES`-independent `index_of` flat-scan restricted to the in-band segments:

```rust
if let Some(state) = band.as_deref_mut() {
    let rank = node_id_to_rank[node_id.0 as usize] as usize;
    let row_best = S::horizontal_max(score).to_i32();
    // index_of over the in-band slice -> lowest column achieving row_best (determinism, MINOR 5)
    let col = index_of::<S>(&striped_h[row_base + beg_sn .. row_base + end_sn],
                            end_sn - beg_sn, row_best);
    state.best_col[rank] = (beg_col as i32 + col).max(0) as u32;
}
```

> Keep the unbanded path (`None`) byte-for-byte as today (guard the new blocks behind `band.is_some()` / the `(0, matrix_width_vecs, 0)` fallback so the emitted code for `None` is identical in behavior).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib fill::tests::banded_linear -- --exact`, then full `cargo test`, then clippy.
Expected: the gate PASSES (in-band cells match exact; at least one row had `beg_sn>=1`); all existing tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/fill.rs
git commit -S -m "feat(simd): banded clip for fill_linear (SW/OV) with correct carry seeding"
```

---

### Task 7: Banded clip in `fill_affine`

Same transformation for the affine fill: adds the E gap-run carry (horizontal → `NEG_INF`) and the vertical F/O matrices (read directly from the predecessor buffer — **no carry to close**, handled by the sentinel). Covers spec §Fill clip for affine.

**Files:**
- Modify: `src/align/simd/fill.rs` (`fill_affine`)
- Test: `src/align/simd/fill.rs` (affine variant of the Task-6 gate)

**Interfaces:**
- Consumes: same as Task 6.
- Produces: `fill_affine` exact in-band under `Some(band)`.

- [ ] **Step 1: Write the failing test**

Mirror the Task-6 gate for `AlignmentType::Local` **and** add an affine-specific assertion over the E/F cells:

```rust
#[test]
fn banded_affine_in_band_cells_match_exact_at_beg_sn_ge_1() {
    // Same structure as banded_linear_..., but call fill_affine and assert H AND the E/F gap
    // matrices match exact for every in-band cell; assert a row with beg_sn>=1 exists.
    // ... (build fixture, exact fill, banded fill with BandConfig{base:2,frac:0.0}) ...
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib fill::tests::banded_affine -- --exact`
Expected: FAIL — clip not applied to affine.

- [ ] **Step 3: Implement the clip**

Apply the Task-6 pattern to `fill_affine`:
- Compute `(beg_sn, end_sn, beg_col)` identically (factor the block from Task 6 into a small helper `fn row_band<S>(band, node, graph, node_id_to_rank, seq_len, lanes, matrix_width_vecs) -> (usize, usize, usize)` in `fill.rs` and call it from both fills — DRY).
- Diagonal H carry: seed from `striped_h[pred_base + beg_sn - 1]` (or `first_column` when `beg_sn==0`).
- Horizontal E carry: seed to `NEG_INF`, iterate `beg_sn..end_sn`, use `adds`/`subs` on every penalty add/sub.
- Vertical F/O: iterate `beg_sn..end_sn`, read `striped_f[pred_base + j]` directly — out-of-band predecessor segments already hold `NEG_INF` (no seeding). Use `adds` for the `F + gap` extension.
- best_col + SW/OV max-tracking over `beg_sn..end_sn` exactly as Task 6.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib fill::tests::banded_affine -- --exact`, then full `cargo test`, clippy.
Expected: PASS; existing green.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/fill.rs
git commit -S -m "feat(simd): banded clip for fill_affine"
```

---

### Task 8: Banded clip in `fill_convex` (primary consumer path)

The convex fill has two gap matrices (E/Q and their carries) plus `y`; it is the fgumi consumer's actual path. Covers spec §Fill clip for convex.

**Files:**
- Modify: `src/align/simd/fill.rs` (`fill_convex`)
- Test: `src/align/simd/fill.rs` (convex variant of the gate)

**Interfaces:**
- Consumes: same as Task 6/7.
- Produces: `fill_convex` exact in-band under `Some(band)`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn banded_convex_in_band_cells_match_exact_at_beg_sn_ge_1() {
    // Same as banded_affine_..., but fill_convex and assert H, E, Q, O, and the second-gap
    // matrices match exact for every in-band cell; assert a beg_sn>=1 row exists.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib fill::tests::banded_convex -- --exact`
Expected: FAIL.

- [ ] **Step 3: Implement the clip**

Apply the same pattern to `fill_convex`:
- Reuse `row_band<S>` (Task 7) for `(beg_sn, end_sn, beg_col)`.
- Diagonal H carry: `striped_h[pred_base + beg_sn - 1]` seed.
- Both horizontal gap-run carries (E and Q): seed to `NEG_INF`.
- The convex `y` carry: seed to `NEG_INF` (it is a horizontal gap-run carry).
- Vertical matrices (F/O second gap): read predecessor buffer directly, sentinel handles OOB.
- Every penalty add/sub → `adds`/`subs`.
- best_col + SW/OV tracking over `beg_sn..end_sn`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib fill::tests::banded_convex -- --exact`, then full `cargo test`, clippy.
Expected: PASS; existing green.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/fill.rs
git commit -S -m "feat(simd): banded clip for fill_convex"
```

---

### Task 9: Band-aware Global endpoint + `NEG_INF` guard (Global & Overlap)

Global is end-to-end in the query: read `H[sink, L]` (or the sentinel), max across sinks, and guard `max_score == NEG_INF`. Do **not** scan for a best-in-band cell (that returns a partial-query score). Extend the guard to Overlap. Covers spec §Fill clip Global fix + MAJOR 3/5 and MINOR 4 (pass 2).

**Files:**
- Modify: `src/align/simd/fill.rs` (Global/Overlap max-tracking in all three fills)
- Modify: `src/align/simd/mod.rs` (`align_simd_*` / `align()` — `NEG_INF` guard)
- Test: `src/align/simd/fill.rs` and/or `mod.rs`

**Interfaces:**
- Produces: banded Global returns the true end-to-end score when column `L` is in a sink's band; returns a sentinel (guarded by the caller) when no sink reaches `L`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn banded_global_reads_column_l_or_sentinel_not_partial() {
    // A sink whose band excludes column L must NOT contribute a partial-query score.
    // Build a graph where a narrow band pushes column L out of the sink's window, and assert the
    // banded Global score is NEG_INF (guarded), never a higher partial score.
    let mut band = BandState::new(&graph, &n2r, query.len(), BandConfig { base: 1, frac: 0.0 });
    let (_, _, max_score) = fill_convex::<Backend>(..., AlignmentType::Global, ..., Some(&mut band));
    assert_eq!(max_score, NEG_INF); // no in-band column-L endpoint -> sentinel, not garbage
}

#[test]
fn banded_global_matches_exact_when_l_in_band() {
    // Wide-enough band: banded Global score == exact Global score, and the endpoint is column L.
}
```

Add a `mod.rs` integration test asserting `align()` surfaces the guard (returns a valid empty/fallback alignment rather than a garbage backtrack) when the score is `NEG_INF`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib fill::tests::banded_global -- --exact`
Expected: FAIL — Global still reads the last-segment lane unconditionally.

- [ ] **Step 3: Implement**

In each fill's Global/Overlap max-tracking branch (`fill.rs:279`+):
- **Global:** for each sink row, read the cell at column `L` (`value_at::<S>(striped_h[row_base + last_seg], last_column_id)` where `last_seg = (seq_len-1)/lanes`), but **only if `L` is within `[beg_sn*lanes, end_sn*lanes)`**; otherwise treat it as `NEG_INF`. Track the max across sinks with `max_j = seq_len`. Under banding, if no sink's band contains `L`, `max_score` stays `NEG_INF`.
- **Overlap:** reduce over `beg_sn..end_sn` (Task 6) and if the result is `NEG_INF` leave it — the caller guards it.

In `mod.rs`, in `align()` (or each `align_simd_*` before backtrack) when `band.is_some()`: if `max_score == NEG_INF`, short-circuit to a valid fallback (empty alignment + `NEG_INF` score, or the documented error return) instead of running the backtrack. Keep the unbanded path unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib fill::tests::banded_global -- --exact`, then full `cargo test`, clippy.
Expected: PASS; existing green.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/fill.rs src/align/simd/mod.rs
git commit -S -m "feat(simd): band-aware Global endpoint + NEG_INF guard for Global/Overlap"
```

---

### Task 10: `SimdEngine::banded` public API + `align()` dispatch

Wire the band into the engine: `band: Option<BandConfig>` field, `banded()` constructor, and `align()` building a `BandState` per call and passing it into the fills. Covers spec §API and §File structure (`mod.rs`).

**Files:**
- Modify: `src/align/simd/mod.rs`
- Test: `tests/` (integration) or `mod.rs` tests

**Interfaces:**
- Produces: `pub fn SimdEngine::banded(alignment_type: AlignmentType, scoring: Scoring, band: BandConfig) -> SimdEngine`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn banded_engine_matches_exact_on_near_identical_family() {
    // Small family of near-identical reads: banded consensus/score == unbanded (in-band optimum).
    let mut exact = SimdEngine::new(AlignmentType::Global, scoring);
    let mut banded = SimdEngine::banded(AlignmentType::Global, scoring, BandConfig::default());
    // Build the same graph in both, align the same reads, assert identical (Alignment, score).
}

#[test]
fn banded_engine_documents_large_indel_miss() {
    // A read with an indel run wider than w: banded result differs from exact (contract pinned).
    let mut exact = SimdEngine::new(AlignmentType::Global, scoring);
    let mut banded = SimdEngine::banded(AlignmentType::Global, scoring,
        BandConfig { base: 2, frac: 0.0 });
    // Assert the scores differ (the documented heuristic miss), and banded still returns a valid
    // structural alignment (no panic, in-band).
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib mod::tests::banded_engine -- --exact` (or the integration test name)
Expected: FAIL — `banded` not defined.

- [ ] **Step 3: Implement**

- Add `band: Option<BandConfig>` to `struct SimdEngine`; set `None` in `new` (`:329`).
- Add the constructor:

```rust
/// Builds a **banded** (opt-in, heuristic) engine. See [`BandConfig`]; unlike [`SimdEngine::new`]
/// this is NOT bit-exact with spoa — it may miss alignments needing an indel wider than the band.
pub fn banded(alignment_type: AlignmentType, scoring: Scoring, band: BandConfig) -> SimdEngine {
    let mut engine = SimdEngine::new(alignment_type, scoring);
    engine.band = Some(band);
    engine
}
```

- In `align()` (`:872`), after seeding scratch and knowing `seq.len()`, build `let mut band_state = self.band.map(|cfg| BandState::new(graph, &self.scratch.node_id_to_rank, seq.len(), cfg));` and pass `band_state.as_mut()` into each `align_simd_*` call (all six tier×gapmode branches). `None` engines pass `None` → exact path verbatim.

> **Escalation note:** the band routes through both int16 and int32 tiers unchanged (the fills are `S`-generic; `adds`/`subs` exist on every backend). `BandState` is tier-independent, so build it once and reuse across a same-`align` escalation retry.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`, then `SPOARS_FORCE_SISD=1 cargo test` (SISD path must be unaffected — no `banded` on `SisdEngine`), then clippy + `cargo fmt --check`.
Expected: PASS; all existing green.

- [ ] **Step 5: Commit**

```bash
git add src/align/simd/mod.rs
git commit -S -m "feat(simd): SimdEngine::banded constructor and align() band dispatch"
```

---

### Task 11: Property tests — accuracy, no-panic, in-band traceback

Pin the heuristic contract and the safety guarantees. Covers spec §Testing.3/.5/.6 and MINOR 9.

**Files:**
- Create: `tests/banded_parity.rs` (integration proptests)

**Interfaces:**
- Consumes: `SimdEngine::banded`, `SimdEngine::new`, `BandConfig`.

- [ ] **Step 1: Write the failing tests**

```rust
proptest! {
    // Accuracy: near-identical families (sub + small indel <= w) -> banded == exact.
    #[test]
    fn banded_equals_exact_when_indels_within_band(seed in any::<u64>()) {
        let (reads, w_ok_cfg) = gen_family_small_indels(seed); // max indel run <= w
        // Build both engines, align all reads, assert identical consensus + score.
    }

    // Saturation: a band edge drifting across many rows must never beat the exact score.
    #[test]
    fn banded_never_beats_exact(seed in any::<u64>()) {
        let reads = gen_family(seed);
        // For Global: banded_score <= exact_score always (sentinel can't wrap positive).
    }

    // No-panic + in-band traceback: random graphs x bands never index OOB and every backtracked
    // cell is in-band; result is a structurally valid alignment.
    #[test]
    fn banded_never_panics_and_stays_in_band(seed in any::<u64>()) {
        let (graph_reads, cfg) = gen_random(seed);
        // align under banded engine; assert Ok/valid alignment, and (via an instrumented
        // CellRead or post-hoc check) every traceback step is within the row's band.
    }
}
```

- [ ] **Step 2: Run to verify they fail (or reveal real bugs)**

Run: `cargo test --test banded_parity`
Expected: FAIL if any invariant is violated (a real bug to fix in Tasks 6-10), else compile-fail until helpers exist.

- [ ] **Step 3: Implement helpers + fix any surfaced bug**

Write the `gen_*` corpus generators (fixed-seed xorshift, reuse `benches/common/mod.rs` style; generate programmatically, no committed fixtures). If a property fails, fix the underlying fill/dispatch (do not weaken the property).

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --test banded_parity`, then full `cargo test`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/banded_parity.rs
git commit -S -m "test(simd): property tests for banded accuracy, saturation, in-band traceback"
```

---

### Task 12: Benchmarks — banded vs exact + cells-computed ratio

Add banded variants to the criterion suite so the speedup is measured, not assumed, alongside a cells-computed ratio. Covers spec §Performance.

**Files:**
- Modify: `benches/poa.rs`
- Modify: `benches/README.md` (document the banded group + how to read the ratio)

**Interfaces:**
- Consumes: `SimdEngine::banded`, `BandConfig`.

- [ ] **Step 1: Add the banded bench group**

In `benches/poa.rs`, add a `build_family_banded` (and `align_one_banded`) group mirroring the existing exact groups over `{3,4,6,10}×235 + 50×1000` (Global/convex), using `SimdEngine::banded(.., BandConfig::default())`. Emit the cells-computed ratio as a `println!`/custom measurement (banded segments summed vs `matrix_height * matrix_width_vecs`), since criterion times wall-clock only.

- [ ] **Step 2: Run the benches (smoke)**

Run: `cargo bench --bench poa -- build_family_banded --warm-up-time 1 --measurement-time 2 --sample-size 20`
Expected: completes; banded is faster than exact at 235 bp (spec predicts ~2–4× on the family build) and the cells ratio is well below 1.

- [ ] **Step 3: Document**

Update `benches/README.md` with the banded group, the expected ~2–4× (measured, not guaranteed), and that banded is heuristic (cite the design doc).

- [ ] **Step 4: Verify build**

Run: `cargo build --benches` and `cargo clippy --all-features --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add benches/poa.rs benches/README.md
git commit -S -m "bench(simd): banded vs exact family build with cells-computed ratio"
```

---

## Self-Review

**Spec coverage:**
- §API (`BandConfig`, `banded()`, `band: Option`) → Tasks 1, 10. ✓
- §Safety model saturating adds (MAJOR 2/4) → Task 2 (primitive) + Tasks 6-8 (applied). ✓
- §Safety model carry closure, pass-2 diagonal-vs-horizontal (FATAL) → Tasks 6-8. ✓
- §Band computation R/anchor (MINOR 8) → Task 3; window/segment/empty-band (MAJOR 6, MINOR 6) → Task 4. ✓
- §best_col determinism (MINOR 5) → Task 4 (struct) + Tasks 6-8 (`index_of` flat-scan). ✓
- §Fill clip Global endpoint + guard (MAJOR 3/5, MINOR 4) → Task 9. ✓
- §Backtrack (relies on PR #11 `StripedView` sentinel, MINOR 9) → no code change; pinned by Task 11. ✓
- §Testing.1 primary gate → Tasks 6-8; .3 saturation, .5 accuracy, .6 no-panic/in-band → Task 11. ✓
- §Performance → Task 12. ✓
- **Open (deferred, per spec §Risks):** multi-sink Global anchor beyond the `NEG_INF` guard — Task 9 implements the interim guard only; the true per-sink anchor is out of scope and flagged in the spec. Noted, not a gap.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — each task repeats its own code. Fixture builders (`build_linear_fixture`, `gen_family`, etc.) are named and described but their bodies defer to existing test/bench helpers; the implementer must wire them to the real `Graph`/`SamBuilder`-style builders in the repo (flagged in-task). This is the one area needing on-the-spot adaptation to the actual test scaffolding.

**Type consistency:** `BandConfig`/`BandState`/`remaining_path`/`anchor`/`node_window`/`segment_range` names are used identically across Tasks 1-10. `adds`/`subs` trait methods (Task 2) match their call sites (Tasks 6-8). `fill_*` and `align_simd_*` trailing `band` param (Task 5) matches all later usage. `best_col` indexed by rank throughout.

## Execution Handoff

Plan complete and saved to `docs/design/2026-07-06-banded-poa-alignment-plan.md` (in `docs/design/`, not `docs/superpowers/plans/`, because the latter is gitignored here). Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task with a two-stage review between tasks; fast iteration and each task's parity gate is checked before the next starts.
2. **Inline Execution** — execute the tasks in this session with checkpoints for review.

Which approach?
