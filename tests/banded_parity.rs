//! Property tests pinning the **heuristic contract** and the **safety guarantees** of the opt-in,
//! abPOA-style banded alignment engine (`SimdEngine::banded`, Tasks 6-10) against the exact engine
//! (`SimdEngine::new`, bit-exact with spoa).
//!
//! Three invariants are pinned here, each as a `deterministic_config`-seeded proptest so any failure
//! reproduces exactly:
//!
//! 1. **Accuracy** ([`banded_equals_exact_when_indels_within_band`]) — for near-identical read
//!    families whose every deviation stays inside the band, the banded engine returns the *identical*
//!    `(Alignment, score)` the exact engine does, folded step-for-step into the same growing graph.
//! 2. **Saturation safety** ([`banded_never_beats_exact`]) — the banded score can never *exceed* the
//!    exact score. Banding only ever *removes* cells from the DP, so its optimum is a subset optimum;
//!    a banded score above exact would prove the `NEG_INF` sentinel leaked and wrapped positive
//!    (the Task 2/9 saturation guarantee, observed end-to-end).
//! 3. **No-panic + in-band traceback** ([`banded_never_panics_and_stays_in_band`]) — over genuinely
//!    reconvergent/branching graphs, across every alignment type × gap mode × a range of band
//!    configs, `align` never panics and every emitted alignment is structurally valid (indices in
//!    range, query positions non-decreasing).
//!
//! # ISA note
//!
//! `SimdEngine::align` only threads the band into a *vectorized* kernel; on a host with no usable
//! SIMD ISA it delegates to the scalar [`spoars::align::SisdEngine`] and the band is ignored, so
//! banded and exact coincide and these properties hold *vacuously*. On any host with a vectorized
//! ISA (native aarch64 NEON here, SSE4.1/AVX2 on x86_64, including Rosetta 2) the real banded
//! kernels execute and the properties are the actual test. Because every assertion below is true on
//! *both* kinds of host, the tests run unconditionally rather than gating on ISA detection.

mod support;

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

use spoars::align::{Alignment, AlignmentEngine, AlignmentType, BandConfig, Scoring, SimdEngine};
use spoars::graph::{Graph, NodeId};

use support::generators::deterministic_config;

/// The consumer's regime: spoa's CLI-default match/mismatch with a **convex** two-piece gap penalty
/// (`Scoring::spoa_default`). This is the primary scoring for the accuracy and saturation
/// properties; the no-panic sweep additionally exercises linear and affine.
fn convex_scoring() -> Scoring {
    Scoring::spoa_default()
}

/// An **affine** gap penalty (`g < e`, `q = g`, `c = e`) — the `simd_parity.rs` affine parameters.
fn affine_scoring() -> Scoring {
    Scoring::new(5, -4, -8, -6, -8, -6).unwrap()
}

/// A **linear** gap penalty (`g == e`) — the `simd_parity.rs` linear parameters.
fn linear_scoring() -> Scoring {
    Scoring::new(5, -4, -8, -8, -8, -8).unwrap()
}

// ---- deterministic corpus generator ------------------------------------------------------------
//
// A tiny xorshift64* PRNG seeded from the proptest-provided `u64`, so a family of near-identical
// reads can be *derived* imperatively (base molecule -> substituted/indel'd variants) rather than
// drawn as an opaque strategy. Everything is generated programmatically — there are no committed
// fixture files.

/// A minimal, self-contained xorshift64* PRNG. Deterministic for a given seed, so a failing
/// proptest case (which prints its `seed`) reproduces the exact same read family.
struct Rng {
    state: u64,
}

impl Rng {
    /// Seeds the generator, mixing the raw proptest seed and forcing a non-zero (odd) state so the
    /// xorshift never degenerates to the all-zero fixed point.
    fn new(seed: u64) -> Self {
        Rng {
            state: (seed ^ 0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    /// Advances the state and returns the next 64-bit value (xorshift64*).
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform value in `0..n` (caller guarantees `n > 0`).
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// The four plain-ACGT bases the corpus is drawn from (banding accuracy is about indel/substitution
/// geometry, not the IUPAC alphabet, so a 4-letter alphabet keeps the derived families crisp).
const BASES: &[u8; 4] = b"ACGT";

/// A random ACGT base different from `b` (used to make every substitution a genuine change).
fn different_base(rng: &mut Rng, b: u8) -> u8 {
    loop {
        let c = BASES[rng.below(4) as usize];
        if c != b {
            return c;
        }
    }
}

/// A fresh random ACGT molecule of length `len`.
fn random_dna(rng: &mut Rng, len: usize) -> Vec<u8> {
    (0..len).map(|_| BASES[rng.below(4) as usize]).collect()
}

/// `base` with `n_subs` random single-base substitutions applied (positions may repeat; each write
/// is still a genuine change at that position). Never changes the length, so the derived read stays
/// on the pure diagonal relative to `base` — no indel is ever optimal under these scorings, which is
/// exactly what makes the banded == exact accuracy property hold for this family.
fn with_substitutions(rng: &mut Rng, base: &[u8], n_subs: usize) -> Vec<u8> {
    let mut seq = base.to_vec();
    for _ in 0..n_subs {
        if seq.is_empty() {
            break;
        }
        let pos = rng.below(seq.len() as u64) as usize;
        seq[pos] = different_base(rng, seq[pos]);
    }
    seq
}

/// `base` with a few substitutions and *optionally* one indel run. `max_indel` caps the run length:
/// small (well within the band) for the branching-graph corpus, large (deliberately overflowing a
/// tiny band) for the saturation stress corpus. Always returns a non-empty sequence.
fn with_edits(rng: &mut Rng, base: &[u8], max_indel: usize) -> Vec<u8> {
    let n_subs = rng.below(4) as usize;
    let mut seq = with_substitutions(rng, base, n_subs);
    match rng.below(3) {
        // Insertion of a random run.
        1 => {
            let run = 1 + rng.below(max_indel as u64) as usize;
            let pos = rng.below(seq.len() as u64 + 1) as usize;
            let insert = random_dna(rng, run);
            seq.splice(pos..pos, insert);
        }
        // Deletion of a random run, always leaving at least one base.
        2 if seq.len() > 1 => {
            let run = (1 + rng.below(max_indel as u64) as usize).min(seq.len() - 1);
            let pos = rng.below((seq.len() - run) as u64 + 1) as usize;
            seq.drain(pos..pos + run);
        }
        _ => {}
    }
    if seq.is_empty() {
        seq.push(BASES[0]);
    }
    seq
}

/// A random base-molecule length in `40..=80` (the brief's regime).
fn random_base_len(rng: &mut Rng) -> usize {
    40 + rng.below(41) as usize
}

/// Folds `alignment` for `read` into `graph`, panicking on the (should-be-impossible) error — a
/// failing `add_alignment_weight` would be a graph-construction bug, not a property violation.
fn add(graph: &mut Graph, alignment: &Alignment, read: &[u8]) {
    graph
        .add_alignment_weight(alignment, read, 1)
        .expect("add_alignment_weight failed on a structurally valid alignment");
}

/// Asserts `alignment` is structurally valid against a graph of `num_nodes` nodes and a query of
/// `query_len` bases: every node index is `-1` (deletion sentinel) or in `[0, num_nodes)`, every
/// query index is `-1` (insertion sentinel) or in `[0, query_len)`, and the non-sentinel query
/// indices are non-decreasing along the trace (the query is consumed monotonically). This is the
/// in-band-traceback safety check: a rank-map misindex or an out-of-band step would surface here as
/// an out-of-range or out-of-order index.
fn assert_valid_alignment(
    alignment: &Alignment,
    num_nodes: usize,
    query_len: usize,
) -> TestCaseResult {
    let mut last_query = -1i32;
    for &(node_idx, query_idx) in alignment {
        prop_assert!(
            node_idx == -1 || (node_idx >= 0 && (node_idx as usize) < num_nodes),
            "node index {node_idx} out of range (num_nodes={num_nodes})"
        );
        prop_assert!(
            query_idx == -1 || (query_idx >= 0 && (query_idx as usize) < query_len),
            "query index {query_idx} out of range (query_len={query_len})"
        );
        if query_idx != -1 {
            prop_assert!(
                query_idx >= last_query,
                "query index {query_idx} decreased below previous {last_query}"
            );
            last_query = query_idx;
        }
    }
    Ok(())
}

proptest! {
    // Each case builds a whole read family through two engines and compares every step, so a
    // modest case count already exercises many (read, graph) pairs. Deterministic seed for repro.
    #![proptest_config(ProptestConfig { cases: 48, ..deterministic_config() })]

    /// **Accuracy invariant.** For a near-identical family — one random ~40-80 bp base molecule plus
    /// substitution-only variants — the banded engine (default `BandConfig`) must return the
    /// *identical* `(Alignment, score)` the exact engine returns, for every read, folded into the
    /// same growing graph.
    ///
    /// Substitution-only variants are the brief's explicitly-blessed reliable form of "deviations
    /// within the band": a substitution keeps the read on the pure diagonal (no indel is optimal
    /// when a mismatch is cheaper than two gaps), so the optimal path never leaves the band and is
    /// unique — the banded fill reaches the exact optimum *and* backtracks it identically. Building
    /// both graphs in lock-step (identical per-read alignments keep the two graphs identical by
    /// induction) turns this into an end-to-end check that `SimdEngine::banded` reproduces
    /// `SimdEngine::new` across a realistic POA build, not just a single alignment.
    #[test]
    fn banded_equals_exact_when_indels_within_band(seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let base_len = random_base_len(&mut rng);
        let base = random_dna(&mut rng, base_len);
        let n_reads = 4 + rng.below(3) as usize; // 4..=6 reads

        let scoring = convex_scoring();
        let mut exact = SimdEngine::new(AlignmentType::Global, scoring);
        let mut banded = SimdEngine::banded(AlignmentType::Global, scoring, BandConfig::default());

        let mut graph_exact = Graph::new();
        let mut graph_banded = Graph::new();

        for i in 0..n_reads {
            // Read 0 is the base molecule itself; the rest carry a light dusting of substitutions
            // (up to ~10% of the length), all strictly within the default band.
            let read = if i == 0 {
                base.clone()
            } else {
                let n_subs = rng.below((base.len() / 10 + 1) as u64) as usize;
                with_substitutions(&mut rng, &base, n_subs)
            };

            let (align_exact, score_exact) = exact.align(&read, &graph_exact);
            let (align_banded, score_banded) = banded.align(&read, &graph_banded);

            prop_assert_eq!(
                &align_banded, &align_exact,
                "banded alignment != exact for read {} (seed={})", i, seed
            );
            prop_assert_eq!(
                score_banded, score_exact,
                "banded score != exact for read {} (seed={})", i, seed
            );

            add(&mut graph_exact, &align_exact, &read);
            add(&mut graph_banded, &align_banded, &read);
        }
    }

    /// **Saturation-safety invariant.** For an arbitrary family (substitutions *and* indels, some
    /// deliberately wider than the band) under Global convex, the banded score aligned against the
    /// exact-built graph must never *exceed* the exact score for the same `(read, graph)`.
    ///
    /// Banding only ever removes cells from the DP, so a banded optimum is a subset optimum:
    /// `banded_score <= exact_score` must hold with no exceptions. A banded score *above* exact is
    /// the cheapest strong signal that a `NEG_INF` sentinel leaked into a live cell and wrapped
    /// positive — the Task 2/9 saturation guarantee observed end-to-end.
    ///
    /// Indels here run up to 30bp specifically because that width reliably overflows
    /// `BandConfig::default()`'s adaptive band (empirically ~10% of reads land strictly below exact
    /// at this width, vs. essentially 0% at 12bp, where the adaptive band absorbs the indel and the
    /// property only ever observes `banded == exact`). Because proptest cases are independent, this
    /// property alone can't *prove* it ever exercises the `<` side — a case count that happens to
    /// avoid every miss would still pass vacuously. [`banded_strictly_below_exact_occurs_for_wide_indels`]
    /// is the non-vacuity backstop: a plain deterministic test using the same generator that asserts
    /// at least one strict miss actually occurs.
    #[test]
    fn banded_never_beats_exact(seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let base_len = random_base_len(&mut rng);
        let base = random_dna(&mut rng, base_len);
        let n_reads = 4 + rng.below(4) as usize; // 4..=7 reads

        let scoring = convex_scoring();
        let mut exact = SimdEngine::new(AlignmentType::Global, scoring);
        let mut banded = SimdEngine::banded(AlignmentType::Global, scoring, BandConfig::default());

        let mut graph = Graph::new();
        for i in 0..n_reads {
            let read = if i == 0 {
                base.clone()
            } else {
                with_edits(&mut rng, &base, 30) // wide indels: reliably overflow the default band
            };

            let (align_exact, score_exact) = exact.align(&read, &graph);
            let (_align_banded, score_banded) = banded.align(&read, &graph);

            prop_assert!(
                score_banded <= score_exact,
                "banded score {} beat exact score {} for read {} (seed={}) — sentinel leak?",
                score_banded, score_exact, i, seed
            );

            // Fold the EXACT alignment in, so both engines always face the identical graph.
            add(&mut graph, &align_exact, &read);
        }
    }

    /// **No-panic + in-band-traceback invariant.** Over a genuinely reconvergent/branching graph,
    /// across every `AlignmentType` × gap mode × a range of `BandConfig`s (a degenerate `w=1` band,
    /// the default, and a fractional band), `SimdEngine::banded::align` must never panic and must
    /// always return a structurally valid alignment.
    ///
    /// The graph is built by folding a substitution+small-indel family through an exact engine:
    /// substitutions create *bubbles* (two alternative bases sharing a predecessor and a successor),
    /// so the shared successor gains multiple in-edges — the reconvergent topology that exercises the
    /// per-node band and the rank map, which a purely linear graph cannot. This is the property most
    /// likely to catch a rank-map misindex or an out-of-band traceback step: those surface as an
    /// out-of-range index, an out-of-order query position, or a panic — all caught by
    /// [`assert_valid_alignment`] and proptest's panic capture.
    #[test]
    fn banded_never_panics_and_stays_in_band(seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let base_len = random_base_len(&mut rng);
        let base = random_dna(&mut rng, base_len);
        let n_family = 5 + rng.below(4) as usize; // 5..=8 reads -> a richly branching graph

        // Build ONE branching graph with an exact engine (small edits keep it well-formed but
        // reconvergent: substitutions bubble, 1-3 bp indels branch).
        let build_scoring = convex_scoring();
        let mut builder = SimdEngine::new(AlignmentType::Global, build_scoring);
        let mut graph = Graph::new();
        for i in 0..n_family {
            let read = if i == 0 {
                base.clone()
            } else {
                with_edits(&mut rng, &base, 3)
            };
            let (alignment, _score) = builder.align(&read, &graph);
            add(&mut graph, &alignment, &read);
        }
        let num_nodes = graph.num_nodes();
        // Sanity: the family really did branch beyond a single linear backbone. `num_nodes >=
        // base.len()` is NOT sufficient here — a purely linear chain already satisfies it (and
        // `add_alignment` only ever adds nodes), so that check is a tautology that can't fail even
        // when the family never reconverges. What actually exercises the per-node band / rank map
        // is a genuine reconvergent node: one with >= 2 in-edges (two alternative predecessors
        // sharing a successor, e.g. a substitution bubble). Assert that directly.
        let has_branch = (0..num_nodes).any(|i| graph.node(NodeId(i as u32)).inedges.len() >= 2);
        prop_assert!(
            has_branch,
            "family must build a reconvergent graph (some node with >= 2 in-edges); seed={seed}"
        );

        // A handful of fresh query reads to align against the fixed branching graph.
        let queries: Vec<Vec<u8>> = (0..3)
            .map(|_| with_edits(&mut rng, &base, 6))
            .collect();

        let types = [
            AlignmentType::Global,
            AlignmentType::Local,
            AlignmentType::Overlap,
        ];
        let scorings: [(&str, Scoring); 3] = [
            ("linear", linear_scoring()),
            ("affine", affine_scoring()),
            ("convex", convex_scoring()),
        ];
        let configs = [
            BandConfig { base: 1, frac: 0.0 },  // degenerate w=1 band
            BandConfig::default(),              // production default
            BandConfig { base: 0, frac: 0.05 }, // purely fractional band
        ];

        for alignment_type in types {
            for (mode, scoring) in scorings {
                for cfg in configs {
                    let mut banded = SimdEngine::banded(alignment_type, scoring, cfg);
                    for query in &queries {
                        // Must not panic (proptest captures panics as failures).
                        let (alignment, _score) = banded.align(query, &graph);
                        assert_valid_alignment(&alignment, num_nodes, query.len())
                            .map_err(|e| {
                                TestCaseError::fail(format!(
                                    "invalid banded alignment: {e} \
                                     (type={alignment_type:?} mode={mode} cfg={cfg:?} seed={seed})"
                                ))
                            })?;
                    }
                }
            }
        }
    }
}

/// Non-vacuity backstop for [`banded_never_beats_exact`]'s `<=` invariant.
///
/// Proptest cases are independent, so `banded_never_beats_exact` can pass forever while never once
/// observing `banded_score < exact_score` — exactly the gap a reviewer found empirically: at
/// `with_edits(..., 12)` against `BandConfig::default()`, 0/10961 simulated reads produced a strict
/// miss (the adaptive band silently absorbed every indel). Widening the indel to 30bp
/// (`banded_never_beats_exact` now uses this width) restores the miss rate to ~10%, but a property
/// test still can't *prove* it hit that branch — only a deterministic count can.
///
/// This plain test builds the same kind of family (base molecule + substitutions + up-to-30bp
/// indels) across a fixed, deterministic set of seeds, and asserts (a) the `<=` invariant holds for
/// every read (duplicating `banded_never_beats_exact`'s check under the exact same generator, so a
/// regression here is doubly confirmed) and (b) at least one read strictly favors the exact engine —
/// i.e. the band genuinely missed at least once. If a future change to the adaptive band (or the
/// generator) makes every indel fit again, this test fails loudly instead of the property quietly
/// going vacuous.
#[test]
fn banded_strictly_below_exact_occurs_for_wide_indels() {
    const N_FAMILIES: u64 = 200;

    let scoring = convex_scoring();
    let mut strict_misses = 0usize;

    for seed in 0..N_FAMILIES {
        let mut rng = Rng::new(seed);
        let base_len = random_base_len(&mut rng);
        let base = random_dna(&mut rng, base_len);
        let n_reads = 4 + rng.below(4) as usize; // 4..=7 reads, matching banded_never_beats_exact

        let mut exact = SimdEngine::new(AlignmentType::Global, scoring);
        let mut banded = SimdEngine::banded(AlignmentType::Global, scoring, BandConfig::default());

        let mut graph = Graph::new();
        for i in 0..n_reads {
            let read = if i == 0 {
                base.clone()
            } else {
                with_edits(&mut rng, &base, 30)
            };

            let (align_exact, score_exact) = exact.align(&read, &graph);
            let (_align_banded, score_banded) = banded.align(&read, &graph);

            assert!(
                score_banded <= score_exact,
                "banded score {score_banded} beat exact score {score_exact} for read {i} \
                 (seed={seed}) — sentinel leak?"
            );
            if score_banded < score_exact {
                strict_misses += 1;
            }

            add(&mut graph, &align_exact, &read);
        }
    }

    assert!(
        strict_misses > 0,
        "expected at least one banded < exact miss across {N_FAMILIES} deterministic families \
         (with 30bp indels vs BandConfig::default()) — got {strict_misses}; the band may have \
         stopped genuinely missing, making banded_never_beats_exact's <= invariant vacuous again"
    );
}
