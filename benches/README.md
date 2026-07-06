# spoars benchmarks & profiling

Exploratory microbenchmarks + a sampling-profiler workflow, aimed at the regime the
fgumi `pairhmm-consensus-proto` consumer drives: **Global (NW) alignment + spoa's default
convex gaps**, small UMI families (3–10 reads) of short (~235 bp) molecules, consumed via
the graph accessors (`column_members` / `msa_columns` / `sequence_path_iter`).

## Microbenchmarks (criterion)

```bash
cargo bench --bench poa                          # full suite
cargo bench --bench poa -- build_family          # one group
cargo bench --bench poa -- --warm-up-time 1 --measurement-time 2 --sample-size 20   # quick
```

Groups (all sweep `{3,4,6,10}×235` + a `50×1000` scaling point):

| group | what it isolates |
|---|---|
| `build_family` | end-to-end `Graph::new` + `align_and_add` loop, **exact** engine — what the consumer feels per family |
| `build_family_banded` | same, with the opt-in `SimdEngine::banded(.., BandConfig::default())` — see below |
| `align_one` | one `engine.align` into an (n−1)-read graph, **exact** engine — align cost alone |
| `align_one_banded` | same, banded engine |
| `add_alignment` | graph merge + per-add `topological_sort` — mutation cost alone |
| `accessors` | `column_members` / `msa_columns` / `sequence_path_iter` — the read-out path |
| `sisd_vs_simd` | same build under both engines — does SIMD pay off at 235 bp? |

Corpus is generated in `benches/common/mod.rs` (fixed-seed xorshift64; 1% substitution +
0.5% indel so the convex-gap/backtrack paths are exercised, not just the diagonal). No
committed fixtures. `build_family` and `build_family_banded` (likewise `align_one` /
`align_one_banded`) consume the *same* corpus per point, so `cargo bench --bench poa --
build_family` runs both exact and banded groups back-to-back for a direct, apples-to-apples
comparison.

## Banded vs exact — measured, not assumed

`SimdEngine::banded(.., BandConfig::default())` is the opt-in, heuristic abPOA-style adaptive
band (see `docs/design/2026-07-06-banded-poa-alignment-design.md`): it can miss the optimal
alignment when a read needs an indel wider than the band, so it trades a small accuracy risk
for a smaller DP search area. It is **not** a drop-in replacement for the exact
`SimdEngine::new` — use it only when that trade-off is acceptable (near-identical reads,
which is the common case for a UMI family).

Run both groups together:

```bash
cargo bench --bench poa -- build_family --warm-up-time 1 --measurement-time 3 --sample-size 30
```

**Measured wall-clock (Apple M-series, NEON, release, `sample-size 30`/`20`, this run's raw
output redirected to a file per the project's long-running-output convention):**

| point | `build_family` (exact) | `build_family_banded` | speedup | `align_one` (exact) | `align_one_banded` | speedup |
|---|---|---|---|---|---|---|
| 3×235 | 291.67 µs | 139.00 µs | **2.10×** | 112.56 µs | 33.998 µs | **3.31×** |
| 4×235 | 408.16 µs | 176.74 µs | **2.31×** | 112.87 µs | 35.001 µs | **3.23×** |
| 6×235 | 657.16 µs | 262.52 µs | **2.50×** | 117.30 µs | 35.950 µs | **3.26×** |
| 10×235 | 1.1578 ms | 428.57 µs | **2.70×** | 120.57 µs | 36.472 µs | **3.31×** |
| 50×1000 | 126.10 ms | 25.058 ms | **5.03×** | 2.9999 ms | 477.27 µs | **6.29×** |

At the fgumi-regime sizes (≤10×235) the `build_family` speedup (2.1–2.7×) sits at the lower
end of the design's ~2–4× hypothesis, while the align-cost-only `align_one` speedup
(3.2–3.3×) lands squarely in the middle of it — the gap between the two is exactly the fixed
per-family overhead (graph mutation, scalar boundary init, allocation) that banding does not
shrink, as documented in "First findings" below. At the `50×1000` scaling point both groups
comfortably exceed the hypothesis (5.0× / 6.3×), consistent with the band covering a much
smaller fraction of a wider matrix (see the cells ratio below).

**Cells-computed ratio (analytical, not measured).** Criterion only times wall-clock, so the
*why* behind the speedup — how much less DP-cell area the banded fill actually scans — is
reported separately via `analytical_cells_ratio` in `benches/poa.rs`, printed once per point
when `build_family_banded` runs (`cargo bench -- build_family_banded`). It is **analytical**:
computed from public band geometry (`BandConfig::width`) and the built graph's node count
(`rows × min(2w+1, L+1)` vs `rows × (L+1)`), not from instrumenting the hot fill path with
per-cell counters — deliberately avoided (see task brief) to keep the fill loop clean. It is a
coarse *lower bound* on cells scanned (it ignores per-row anchor drift near the query's edges
and vector-lane quantization in `segment_range`, both of which only narrow the true window
further):

| point | cells ratio (banded/exact) | fewer cells |
|---|---|---|
| 3×235 | 0.1061 | 9.4× |
| 4×235 | 0.1058 | 9.5× |
| 6×235 | 0.1057 | 9.5× |
| 10×235 | 0.1058 | 9.5× |
| 50×1000 | 0.0409 | 24.4× |

The cells ratio (9.5–24×) is far more favorable than the measured wall-clock speedup
(2.1–6.3×) — expected, since wall-clock also carries fixed per-align costs (graph mutation,
scalar boundary init/reseed, allocation) that scale with the *number* of aligns, not the
*area* of the DP matrix, so they don't shrink when the band narrows. This is the same fixed-
overhead effect "First findings" below documents for the exact engine's own scalar-boundary-
init cost.

## Profiling (macOS, no sudo)

`examples/profile_family.rs` mirrors the consumer exactly (Global/convex, small family,
rebuilt each pass). Build **with a dSYM** so frames symbolicate:

```bash
CARGO_PROFILE_RELEASE_DEBUG=2 CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed \
  cargo build --release --example profile_family
```

Then pick a profiler:

```bash
# Interactive flamegraph (Firefox Profiler UI):
samply record --save-only -o profiles/simd.json.gz -- ./target/release/examples/profile_family 9000 6 235
samply load profiles/simd.json.gz          # re-open the saved capture

# Text call tree, symbolicated, no browser (Apple's sampler):
./target/release/examples/profile_family 40000 6 235 & \
  sample $! 18 1 -f profiles/sample_simd.txt; wait

# CPU/mem/IO over time:
tricorder --trace ./target/release/examples/profile_family 9000 6 235

# Scalar path, for comparison (any of the above):
SPOARS_FORCE_SISD=1 <cmd> ...
```

## First findings (Apple M-series, Global/convex, 6×235)

- **The align inner loop is ~everything.** `SimdEngine::align` self-time is **~85%** of
  samples. PR6's `#[inline(always)]` folds the convex fill *and* destripe into `align`, so
  they show up as `align`'s own self-time rather than separate frames.

- **Drill-in split** (temporarily `#[inline(never)]` on `fill_convex` / `destripe_interior`
  / `build_profile`, reverted after measuring). Self-time over the whole family build:

  | bucket | share |
  |---|---|
  | `fill_convex` — the vectorized DP recurrence | **70.3%** |
  | `destripe_interior` — striped → row-major transpose | **18.9%** |
  | graph mutation + alloc churn (`topological_sort`/`add_edge`/`add_alignment`/malloc) | 4.1% |
  | backtrack (scalar) | 2.2% |
  | scalar boundary init reused by the SIMD path | 1.7% |
  | `build_profile` + row-0 seed | 0.5% |

  **The headline: destripe is ~19% of total** — the #2 hotpath after the fill itself, and
  the most attractive concrete target. `destripe_interior` transposes the *entire* interior
  matrix (`O(rows × cols)`) to row-major purely so the shared scalar backtrack can index it,
  but the backtrack only walks a single `O(path)` route. That's ~19% of runtime spent
  materializing cells the backtrack never reads. Teaching the backtrack to index the striped
  `H` directly (or destriping lazily along the backtrack path) could reclaim most of it, and
  is bit-exact-safe: it changes *how* cells are read, not their values.
- **SIMD already pays off at 235 bp:** ~**3.2×** over scalar (`build_family`: 6×235 ≈
  788 µs SIMD vs 2.40 ms scalar). Per-align fixed overhead does *not* eat the win at this
  size — the earlier hypothesis that setup/topo-sort dominates at small N was **refuted**.
- **Everything else is small:** graph mutation (`topological_sort` + `add_edge` +
  `add_alignment`) ≈ 2%, scalar boundary init reused by the SIMD path
  (`SisdEngine::initialize` / `boundary_column_value` / `reseed_scalar_buffers`) ≈ 4–5%
  (a *fixed* per-align cost, proportionally larger for short reads), striped
  `build_profile` ≈ 1%, allocation growth ≈ <1%. The consumer's accessor read-out
  (`column_members` ≈ 34 µs, `sequence_path_iter` ≈ 17 µs, `msa_columns` ≈ 0.2 µs) is
  negligible against a ~1.4 ms family build.

## Prototype: option 1 (striped-aware convex backtrack) — validated

Implemented the "skip the full-matrix destripe" idea for the convex path: the shared
backtrack now reads its H/E/F/O/Q matrices through a `CellRead` view
(`src/align/backtrack.rs`), so the SIMD engine feeds it a `StripedView`
(`src/align/simd/mod.rs`) that indexes the striped fill directly along the backtrack path
instead of destriping all five interiors first (`align_simd_convex` no longer calls
`destripe_interior`). The scalar/`SisdEngine` path is unchanged (it feeds `RowMajor` views).

**Result (Apple M-series, NEON):**

| bench | baseline (destripe) | prototype (striped) | change |
|---|---|---|---|
| `build_family/6x235` | 786 µs | 605 µs | **−23%** |
| `build_family/10x235` | 1.39 ms | 1.06 ms | **−23%** |
| `build_family/50x1000` | 154 ms | 118 ms | **−24%** |
| `align_one/*` | — | — | **−22% to −24.5%** |
| SIMD-vs-scalar (6x235) | ~3.2× | ~4.0× | — |

- **Bit-exact:** the full suite passes unchanged — 99 lib + 16 `simd_parity`
  (incl. `simd_convex_matches_sisd` and the int16/int32 capstones) + 12 `engine_parity` +
  14 `cli_parity` + graph/proptest, i.e. SIMD == SISD == C++ oracle across all modes.
- **Where the time went:** a confirmation profile shows `destripe_interior` at **0%**
  (was ~19%); the striped read adds only ~2% to `backtrack_convex_impl` (it's `O(path)`,
  not `O(rows·cols)`). The net ~22–24% slightly exceeds destripe's own ~19% self-time
  because dropping the full-interior transpose also relieves cache/bandwidth pressure on
  the surrounding code. The `SisdEngine` path is flat (±2% noise), confirming the scalar
  engine was untouched.

**Scope / notes:** all three gap modes (`linear`/`affine`/`convex`) now read striped and
skip the destripe; the shared `backtrack_*` functions became generic over a `CellRead` view
(`RowMajor` for the scalar path, `StripedView` for SIMD). The convex win is the largest
(5 matrices destriped → 0); linear (1) and affine (3) get the same treatment with smaller
absolute destripe savings. `destripe_interior` is retained as the tested reference for the
striped→row-major mapping `StripedView` mirrors. The per-cell lane extract in
`StripedView::get` is a store-then-index; already negligible (~2%) but a single-lane extract
intrinsic would trim it. Perf measured on NEON (convex); x86 (AVX2/SSE4.1), where destripe
was historically hotter, should see at least as much — confirm on an x86 host, but CI
validates x86 *correctness* (the parity suite runs on `ubuntu-latest`).

**Takeaway:** for this consumer, the next real win is in the convex-gap fill recurrence
(algorithmic — e.g. banding — or micro-opt), not in graph mutation, profile build, or the
accessors. The ~5% fixed scalar-boundary-init cost is a smaller, self-contained secondary
target that would help short-read families specifically.
