# Banded POA Alignment (opt-in, abPOA-style) — Design

**Status:** Design approved; spec revised across **two** adversarial (Fable) review passes that found real soundness holes — see "Adversarial review" at the end for the finding→resolution trail. Pass 2 corrected the band-edge carry model (diagonal vs horizontal) and the saturating-add availability gap. Stacks on `nh_striped-backtrack` (PR #11).

**Goal:** Add an **opt-in, heuristic, adaptive-banded** alignment mode to `SimdEngine` that computes only a per-node query-column window of the DP matrix, trading optimality for speed on similar-sequence inputs (the fgumi `pairhmm-consensus-proto` UMI-consensus workload: small families of near-identical short reads, Global + convex).

**Non-goals:** Changing the default engine's behavior; exact/optimal banded alignment (that would require an A\*/POASTA-style engine — explicitly out of scope); banding the scalar `SisdEngine`.

---

## Background

- **SPOA does not band.** The library spoars reimplements offers only full-matrix `kNW`/`kSW`/`kOV` (verified: 0 hits for "band" in `~/work/git/spoa`). So banding is a **new mode with no bit-exact-with-spoa counterpart** — it produces results spoa never would. spoars' identity is bit-exactness, so banding must be explicitly opt-in and fenced off from the default.
- **abPOA** (Gao et al., Bioinformatics 2021) is the sole graph-banding prior art and is explicitly heuristic. It is our model: allocate the full matrix, initialize out-of-band cells to a min-sentinel, compute only a per-node `[beg,end]` window, and let the sentinel lose every `max`.
- Cross-tool survey: ksw2/minimap2 (static `|i−j|≤w` + separate z-drop; **closes the band's left edge with a min-sentinel, does NOT inject a boundary value** — the pattern we follow), BWA-MEM (`±w` envelope + local-only lossless shrink + `max_off` doubling retry), block-aligner (adaptive block, edge-gradient shift, ×2 grow/shrink), WFA2 (score-order wavefront; never prune the sink diagonal).

## Decisions (locked)

1. **Opt-in heuristic band, abPOA-style.** May miss alignments whose optimal path needs an indel larger than the band. Bit-exact full-DP stays the untouched default.
2. **Centering: abPOA union model** — `L − R` anchor (R = heaviest-read-support remaining path length) **unioned with predecessors' best-cell columns**; width `w = base + round(frac·L)`.
3. **API:** `SimdEngine::banded(alignment_type, scoring, BandConfig)`; internal `band: Option<BandConfig>`. `SimdEngine::new` is unchanged and remains the exact path.
4. **Scope:** all three alignment types (Global/Local/Overlap) and all three gap modes (linear/affine/convex). The band clip is shared across all six; per-type max-tracking becomes band-aware.

---

## Design

### Safety model (revised — this is the crux)

Out-of-band cells are **never allocated away**: like abPOA, the striped buffers stay full-width and out-of-band cells hold a **saturating min-sentinel**. Safety rests on *sentinel domination*, precisely stated:

> Every DP read of an out-of-band cell returns the sentinel `NEG_INF`; adds against it **saturate** (stay `NEG_INF`, never wrap); therefore it loses every `max()` and can never satisfy a backtrack equality test. The band's *union* with predecessors only widens where real values are computed — it makes no claim that reads stay inside computed cells (they don't, and that's fine).

This replaces the earlier (incorrect) "unioning predecessors guarantees reads land in computed cells" invariant — the union makes bands *larger*, so reads of a predecessor's out-of-band cells are expected; the sentinel is what makes them safe. Two things make the sentinel actually dominate (both are new requirements, not free):

- **Saturating penalty adds on the banded path.** The exact path's escalation headroom (`escalate()` reserves ~1024 and pads lanes) was derived assuming `NEG_INF` lives *only* in the boundary row/column/padding, where it accrues a bounded handful of adds. Banding creates *interior* `NEG_INF` cells that a drifting band edge can penalize across many rows (`NEG_INF + k·g`); with `g` down to −128, ~8 adds wrap int16 to a large positive and win the max — a silent wrong answer. **Fix:** the banded fill uses saturating adds so a sentinel cell stays `NEG_INF` regardless of add depth.
  - **Availability gap (must be budgeted).** The `Simd` trait today exposes *only* wrapping `add`/`sub` (`lanes.rs:60-68`; the exact path relies on the +1024 headroom, deliberately **not** saturation), and no backend uses `adds`/`vqadd`. Saturating variants are therefore **new trait methods requiring impls on all five backends**. The int16 tier is easy (`_mm_adds_epi16`/`_mm256_adds_epi16`/`vqaddq_s16` all exist). The int32 **escalation** tier is not: **x86 has no `_mm_adds_epi32`/`_mm256_adds_epi32`/`_mm_subs_epi32`** — saturating 32-bit add/sub must be *emulated* (compare/blend or 64-bit widen); NEON has `vqaddq_s32`/`vqsubq_s32`. And int32 genuinely needs it: `NEG_INF = i32::MIN + 1024` carries the *same* 1024 absolute headroom as int16, so a sentinel penalized by `g=−128` wraps after ~9 adds in int32 too. A "re-derived headroom bound" is **not** a viable escape: over hundreds of drifting rows the accumulated `k·|g|` exceeds even the int16 range, and changing `NEG_INF` for int32 only would break the shared constant that `seed_striped_row0` and `escalate()`'s thresholds depend on. Both the affine/convex carry `sub` (`S::sub(e_row, g_minus_e)`) and the fill adds are affected. *Verified sound:* saturation never masks escalation — `escalate()` guarantees real DP values stay far below the saturation threshold, so on real values a saturating add is bit-identical to wrapping; it only pins sentinel-range values.
- **Band-edge carry closure — horizontal carries only.** See fill clip below (pass 2 corrected an over-broad "close every carry" rule that also closed the *diagonal* carry and broke left-edge exactness).

### API

```rust
/// Adaptive-band configuration (abPOA-style). APPROXIMATE: banded alignment may miss the optimal
/// path when it needs an indel larger than the band. `SimdEngine::new` stays exact (bit-exact with
/// spoa); use this only when the speed/accuracy trade-off is acceptable (near-identical reads).
#[derive(Debug, Clone, Copy)]
pub struct BandConfig { pub base: u32, pub frac: f32 } // default abPOA: base=10, frac=0.01

impl BandConfig {
    /// Per-align half-width, computed in usize and SATURATING (never overflows). A width `>= L`
    /// means "no effective band" (used by the smoke test); production values are small.
    fn width(&self, query_len: usize) -> usize {
        (self.base as usize).saturating_add((self.frac as f64 * query_len as f64).round() as usize)
    }
}

impl SimdEngine {
    pub fn new(alignment_type: AlignmentType, scoring: Scoring) -> SimdEngine; // exact (unchanged)
    pub fn banded(alignment_type: AlignmentType, scoring: Scoring, band: BandConfig) -> SimdEngine;
}
```

`SimdEngine` gains `band: Option<BandConfig>`. `align()` dispatches: `None` → existing exact pipeline verbatim; `Some(cfg)` → banded pipeline. Same `AlignmentEngine` trait, so `align_and_add` and the consumer call site work unchanged. **`w` is always computed in `usize`, saturating** — there is no `u32::MAX` sentinel arithmetic.

### Band computation (`src/align/simd/band.rs`, new — isolated & unit-testable)

Per `align(seq, graph)` with `L = seq.len()`, `w = cfg.width(L)`:

1. **Remaining-path length `R`** — one reverse pass over `rank_to_node` (reversed topological order; well-founded on a DAG): `R[sink]=0`; `R[n]=1+R[s*]`, `s*` = heaviest-read-support successor (max outedge weight, ties by lowest rank for determinism). Anchor `anchor(n) = clamp(L − R[n], 0, L)`. **Caveat (heuristic):** `R` counts *nodes* on the heaviest path as a proxy for *query columns remaining*; indels between that path and the query bias the anchor (documented tradeoff, not a bug).
2. **Per-node window `[beg, end)`** (half-open), computed as each row is reached:
   - `Mstart, Mend` = min/max over `n`'s predecessors of each predecessor's best-cell query column (`best_col[rank]`, propagated forward). Source node (no predecessors) uses `anchor(n)`.
   - `beg = max(0, min(Mstart, anchor(n)) − w)`, `end = min(L, max(Mend, anchor(n)) + w + 1)` (half-open upper bound).
3. **Segment range** `[beg_sn, end_sn) = [beg / LANES, end.div_ceil(LANES))`, clamped `end_sn = min(end_sn, matrix_width_vecs)`. Iteration is `beg_sn..end_sn` (half-open) — this avoids the off-by-one that `..=end/LANES` hits when `L % LANES == 0` (which would index one segment past the row block). **Empty-band guard:** a pathological config (e.g. `base=0, frac=0`) or a sink with `anchor=L` can make `beg = L`, giving `beg_sn == end_sn` — an empty range that leaves the whole row at `NEG_INF` and silently drops it. Enforce a minimum non-empty window (`end_sn > beg_sn`, widening `end_sn` by one segment if needed) or reject such configs. Note also `beg_sn = beg/LANES` **floors** to the segment boundary, so the effective computed band is `[beg_sn·LANES, end_sn·LANES)` and the left-edge carry closure happens at `beg_sn·LANES` (not `beg`); unit tests must target the *segment* boundary.

### Fill clip (`src/align/simd/fill.rs`)

Each `fill_{linear,affine,convex}` takes the per-row window and iterates `beg_sn..end_sn` instead of `0..matrix_width_vecs`. Out-of-band segments keep their `NEG_INF` init.

**Band-edge carry closure (primary hazard, corrected — pass 2).** Each banded row has **two distinct lane-0 carries that must be treated oppositely**; the pass-1 rule "close every band-edge carry to `NEG_INF`" was wrong because it conflated them:

- **Horizontal carry — close to `NEG_INF`.** The prefix/gap-run carry `x` (`fill.rs:263,464,697` and the affine/convex E/Q open carries) represents *this row's own* running H/E/Q at the column just left of the current segment (`beg−1`). In a banded row that column is genuinely **never computed**, and it is *not* the boundary column (column 0) — so injecting any finite value fabricates a horizontal gap from column 0 with the skipped-segment penalties missing (over-estimation). **Seed this carry to `NEG_INF` (edge closure)**, exactly as ksw2 does. `srli_top_lane(NEG_INF_splat)` correctly closes lane 0.
- **Diagonal carry — seed from the predecessor, do NOT close.** The diagonal carry `x` (`fill.rs:232,249` first predecessor; the additional-predecessor loop; the affine/convex H-diagonal) brings the **predecessor row's** value at column `beg−1` into lane 0 (`x = srli_top_lane(splat(first_column(pred)))`, updated `x = srli_top_lane(h_pred)`). Because the union model makes adjacent rows' bands overlap (the anchor drifts ~1 column/row), `(pred, beg−1)` is normally **inside the predecessor's band and holds a real value** — it is the *match/diagonal* transition `H_pred[beg−1] + s`, the dominant term for near-identical reads. Closing it to `NEG_INF` would discard that term, make left-edge in-band cells differ from exact, and **fail the §1 per-cell exact-oracle gate by construction.** **Fix:** seed the diagonal carry from the predecessor buffer's top lane at segment `beg_sn−1` (`srli_top_lane(striped_h[pred_base + beg_sn − 1])`); because buffers are full-width and resize-refilled to `NEG_INF`, this *naturally* yields `NEG_INF` only when that predecessor cell is itself out of band — the correct behavior with no special case.

**State that has no carry to close.** The vertical F/O matrices are read directly from the predecessor buffer (`striped_f[pred_base + j]`, `fill.rs:435,662`), so out-of-band values are already handled by the buffer sentinel — there is nothing to seed. Note only **convex** has a `y` carry; affine does not. The band is thus entered from above/diagonal (predecessors) with real predecessor values, but never by a *horizontal* gap crossing the skipped region.

**Band-aware max-tracking (per type).**
- **SW/OV:** reduce over `beg_sn..end_sn` only (not the whole row). **Overlap guard (pass 2):** an Overlap sink whose window contains no in-band overlap endpoint yields a `NEG_INF` row score; extend the `max_score == NEG_INF` guard (below) to Overlap so a sentinel score never leaks to the caller as a real result.
- **Global (NW):** the exact path reads the *last* segment's `last_column_id` lane unconditionally to get the endpoint at column `L`. With a band, a sink row's window may not reach column `L`; that last-segment cell is then an uncomputed `NEG_INF`, giving `max_score = NEG_INF` and a `max_j = L` that sends the backtrack into sentinel cells (and the `debug_assert h.get(max_i,max_j)==max_score` passes vacuously, `NEG_INF==NEG_INF`, masking it). **Fix (pass 2 — precise wording):** Global is end-to-end in the query, so the *only* valid endpoint is column `L`. Read `H[sink, L]` for each sink — a real value if `L` is in that sink's band, else the resize-refilled `NEG_INF` — take the max across sinks, then guard `max_score == NEG_INF`. Do **not** "scan for the best in-band cell of the sink row": that would pick a partial-query column and silently return a *higher* score for an alignment that does not consume the whole query — a semantically wrong Global score. The rule is strictly "column `L` or sentinel." **Multi-sink caveat:** POA graphs have multiple sink nodes (reads ending at different graph positions); giving every sink `R=0 ⇒ anchor=L` forces early-ending sinks' true endpoints out of band. The `NEG_INF` guard only catches the case where *no* sink reaches `L` — if the true optimum ends at an early sink out of band but another sink reaches `L` sub-optimally, the guard does **not** fire and a silently-lower score is returned (within the heuristic contract, but not "safe"). The anchor for a non-primary sink must reflect where its supporting reads actually end — **open question**, see Risks.

### Backtrack

Reads via PR #11's `StripedView`, which returns the saturating `NEG_INF` sentinel for out-of-band cells. The sentinel cannot satisfy a tie-break equality (`h_ij == h.get(pred,·) + penalty`) because it saturates below any reachable real value by more than any single penalty — so the backtrack never steps out of the computed region. This is now an explicit guarantee (sentinel + saturation), not an inherited assumption, and is pinned by an **in-band-traceback proptest** (below).

### `best_col` propagation

The fill already computes each row's max for per-type tracking; banding additionally records that max's query column into `best_col[rank]`. A successor's `Mstart/Mend` is the min/max of `best_col` over its predecessors (one write per row; min/max over in-edges).

**Determinism (pass 2).** `best_col`'s "the max's query column" **must** reuse the existing `index_of` flat-scan convention (lowest column achieving the max, `fill.rs:87-102`), *not* a within-vector argmax. A lane-order argmax is `LANES`-dependent and would differ between AVX2 (16 lanes) and SSE4.1 (8 lanes), making bands — and thus heuristic results — vary by host ISA and breaking the crate's cross-ISA determinism (and any cross-ISA test). The `R` reverse-pass tie-break is already pinned (lowest rank); min/max unions are order-free.

---

## Correctness & testing

Banding is heuristic, so the differential oracle is not a direct pass/fail. Layered strategy:

1. **PRIMARY gate — per-cell exact oracle at `beg_sn ≥ 1`.** Compute the full exact matrix (existing engine), then run the banded fill with a band **deliberately placed so `beg_sn ≥ 1`** and assert its in-band `H/E/F/O/Q` cells equal the exact cells. This is the only test that exercises the corrected band-edge carry seeding — the thing most likely to be wrong. (The earlier "infinite band ⇒ exact" idea is demoted: with `w ≥ L`, `beg = 0` on every row, so `beg_sn = 0` always and the seeding branch never runs — it is a **smoke test only**, blind to the real hazard.)
2. **Left-edge carry unit tests.** Hand-built small graphs with the band starting mid-row, asserting first-in-band-segment values match the full DP — the targeted twin of gate 1.
3. **Saturation test.** A band edge that drifts across ≥ ⌈headroom/|min gap|⌉ rows must NOT wrap the sentinel to a positive value that wins; assert the banded result never beats the exact score.
4. **Global endpoint / multi-sink test.** A graph with an early-ending sink under a narrow band must not return a `NEG_INF` score or empty/garbage alignment; assert the fallback fires and the result is a valid in-band alignment.
5. **Accuracy property tests.** Near-identical read families (sub + indel, fixed seed): banded consensus/MSA == unbanded when max indel run ≤ `w`; characterize divergence vs `w`/indel; an explicit test *demonstrating* the documented failure (indel > `w` ⇒ missed) so the contract is pinned.
6. **No-panic + in-band-traceback proptests.** Random graphs × bands: never index outside a row block; every backtracked cell is in-band; always a structurally valid alignment.

## Performance

At 235 bp with `w ≈ 12`, ~25 of ~235 columns computed → up to ~8–9× fewer fill cells (minus the reverse-`R` pass and per-node band bookkeeping); larger for longer reads. Measured by adding banded variants to the criterion `build_family`/`align_one` groups vs the exact baseline (Global/convex, {3,4,6,10}×235 + 50×1000), reporting cells-computed ratio alongside wall-clock.

## File structure

- **`src/align/simd/band.rs`** (new): `BandConfig` (+ saturating `width`), `R` reverse pass, per-node `[beg,end)` computation, segment-range helper (half-open + clamp), `best_col` propagation. Unit-tested here.
- **`src/align/simd/mod.rs`**: `SimdEngine.band` + `banded()` ctor; `align()` dispatch + `max_score == NEG_INF` guard; thread the band into the fills.
- **`src/align/simd/fill.rs`**: per-row clip; `NEG_INF` band-edge carry closure; saturating adds on the banded path; band-aware max-tracking incl. Global in-band endpoint scan.
- **Backtrack: no logic change** (relies on the saturating sentinel from `StripedView`).
- **`benches/poa.rs`**: banded variants.

## Risks & open questions

- **Band-edge carry seeding** (primary) — corrected to `NEG_INF` closure; gate 1/2 exist because getting this wrong is silent.
- **Saturating adds** — must cover every add that can touch a sentinel across all three gap modes; if a single path uses a wrapping add, the sentinel can win. Auditable + tested (gate 3).
- **Global multi-sink anchor** — `R=0 ⇒ anchor=L` is wrong for sinks whose reads end early; needs a per-sink anchor from actual read-end positions. **Open.** Interim: guard `NEG_INF` max and fall back; may over-widen or reject some multi-sink graphs until resolved.
- **`R` node-count vs query-offset bias** — heuristic anchor drift on indel-heavy graphs (acknowledged tradeoff).
- **Local/Overlap centering** — endpoints not fixed at `(sink, L)`; leans on predecessor propagation; validate accuracy separately from Global.
- **Memory** — like abPOA, buffers stay full-width (only compute is banded); per-band allocation is a later optimization, out of scope.
- **`max_off` doubling retry** (BWA/block-aligner) — deferred; plain heuristic band without retry, per the locked stance.

## Stacking

Branch `nh_banded-poa` off `nh_striped-backtrack` (PR #11); depends on PR #11's `CellRead`/`StripedView`.

---

## Adversarial review (Fable) — findings & resolutions

A Fable sub-agent adversarially reviewed the first draft against the real fill/backtrack/graph source. Findings and how this revision resolves each:

- **FATAL 1 — dependency-safety invariant backwards.** Union widens bands, so reads of predecessors' out-of-band cells are expected, not prevented. **Resolved:** safety restated on saturating-sentinel domination (§Safety model); the false invariant is deleted.
- **FATAL 2 — "infinite band ⇒ exact" gate never exercises `beg_sn>0`.** With `w ≥ L`, `beg=0` always. **Resolved:** demoted to a smoke test; PRIMARY gate is now a per-cell exact oracle with the band forced to `beg_sn ≥ 1` (testing §1/§2).
- **FATAL 3 — left-edge horizontal carry seed impossible/wrong.** The needed cell (`beg−1` in this row) is never computed; a finite seed over-estimates. **Resolved:** seed all band-edge carries to `NEG_INF` (edge closure), matching ksw2 (§Fill clip).
- **MAJOR 4 — interior `NEG_INF` wraps int16 via repeated `+g`.** Headroom was derived for boundary-only sentinels. **Resolved:** saturating adds on the banded path (§Safety model) + saturation test (§3).
- **MAJOR 5 — Global endpoint outside band ⇒ `NEG_INF` max + vacuously-passing assert + garbage traceback; multi-sink forced to `anchor=L`.** **Resolved:** band-aware in-band endpoint scan + `max_score==NEG_INF` guard (§Fill clip, §API); multi-sink anchor flagged Open (§Risks) with a fallback.
- **MAJOR 6 — `end_sn = end/LANES` off-by-one when `L % LANES == 0`.** **Resolved:** half-open `[beg_sn, end_sn)` with `end_sn = min(end.div_ceil(LANES), matrix_width_vecs)` (§Band computation.3).
- **MAJOR 7 — `w = base + round(frac·L)` `u32` overflow; the `u32::MAX` sentinel triggers it.** **Resolved:** compute `w` in `usize`, saturating; no MAX-arithmetic sentinel (§API).
- **MINOR 8 — `R` node-count is a biased query-offset proxy.** Acknowledged tradeoff (§Band computation.1, §Risks).
- **MINOR 9 — backtrack "never selects out-of-band" asserted, not guaranteed.** **Resolved:** guaranteed by saturating sentinel + pinned by in-band-traceback proptest (§Backtrack, §Testing.6).

### Pass 2 (re-review of the revision)

A second Fable pass re-verified every pass-1 resolution against the code and attacked the revision itself. It confirmed the safety-model restatement, the half-open range (MAJOR 6), the saturating `usize` width (MAJOR 7), the horizontal-carry `NEG_INF` closure, the `beg_sn≥1` gate's constructibility, and that saturation never masks escalation. It found two load-bearing errors and three refinements (all verified against `fill.rs`/`lanes.rs` and folded in above):

- **FATAL (pass 2) — "close every band-edge carry to `NEG_INF`" is wrong for the *diagonal* carry.** The fill has two distinct lane-0 carries; closing the diagonal one discards `H_pred[beg−1] + s` (the match transition, dominant for near-identical reads) and would fail the §1 gate by construction. **Resolved:** only the *horizontal* carry closes to `NEG_INF`; the diagonal carry is seeded from the predecessor buffer's `beg_sn−1` top lane (naturally `NEG_INF` iff that pred cell is itself out of band). F/O are vertical (no carry); only convex has `y` (§Fill clip).
- **MAJOR (pass 2) — "uses saturating adds" is not an available primitive.** The trait exposes only wrapping `add`/`sub`; saturating variants are new methods on all 5 backends, and **x86 has no native saturating int32 add/sub** (the escalation tier), which needs emulation and cannot be dodged by re-deriving headroom. **Resolved:** documented as a budgeted implementation cost; escalation-masking ruled out (§Safety model).
- **MAJOR (pass 2) — Global "best in-band endpoint cell" wording invites a silently-wrong partial-query score.** **Resolved:** rule is strictly "read `H[sink, L]` or sentinel," never an in-band column scan (§Fill clip).
- **MINOR (pass 2) — multi-sink guard catches total misses, not suboptimality; Overlap left ungated.** **Resolved:** documented the residual suboptimality (within the heuristic contract) and extended the `NEG_INF` guard to Overlap (§Fill clip).
- **MINOR (pass 2) — `best_col` argmax tie-break unspecified ⇒ cross-ISA non-determinism.** **Resolved:** must reuse the `index_of` flat-scan (lowest column), `LANES`-independent (§best_col propagation).
- **MINOR (pass 2) — empty band (`beg_sn == end_sn`) unguarded; floor-quantized left edge.** **Resolved:** min-non-empty-window guard + tests target the segment boundary (§Band computation.3).
