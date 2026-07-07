//! Band geometry for the opt-in, heuristic abPOA-style banded alignment mode.
//!
//! Everything here is pure and unit-testable without a SIMD backend. The geometry it produces —
//! `R`, `anchor`, `best_col`, and the half-open column window `[beg, end)` — is `LANES`-independent
//! (identical on every ISA). Note the fill then realizes that window at `LANES`-wide *segment*
//! granularity (`[beg_sn * LANES, end_sn * LANES)`), so the actual computed region — and thus a
//! banded result — can still differ across ISAs; every variant satisfies `banded <= exact`.

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
        BandConfig {
            base: 10,
            frac: 0.01,
        }
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

/// Remaining heaviest-support path length per rank: `R[sink] = 0`, and for every other node `n`,
/// `R[n] = 1 + R[s*]` where `s*` is the successor reached by `n`'s heaviest out-edge (max
/// [`crate::graph::Edge::weight`], ties broken by the **lowest rank** among tied successors, for
/// determinism across runs/ISAs — the same cross-run/ISA-parity concern the SIMD kernels are
/// built around applies here too). Computed in a single reverse topological pass
/// (`graph.rank_order()` reversed), which is well-founded since every out-edge points to a
/// strictly-later rank on the DAG.
///
/// Returned `Vec` is indexed **by rank**, not by [`crate::graph::NodeId`] — use
/// `node_id_to_rank[node_id.0 as usize]` to look up a specific node's `R`.
///
/// # Heuristic bias (documented tradeoff, not a bug)
/// This counts *nodes* on the heaviest-support path as a proxy for query *columns* remaining. If
/// the query has indels relative to that path, the true number of query columns left from a given
/// node can differ from `R[n]`; the anchor derived from `R` via [`anchor`] absorbs that slack via
/// [`BandConfig`]'s half-width rather than tracking it exactly. This mirrors abPOA's own banding
/// heuristic.
pub(crate) fn remaining_path(graph: &crate::graph::Graph, node_id_to_rank: &[u32]) -> Vec<u32> {
    let rank_to_node = graph.rank_order();
    let mut r = vec![0u32; rank_to_node.len()];
    for &node_id in rank_to_node.iter().rev() {
        let node = graph.node(node_id);
        let rank = node_id_to_rank[node_id.0 as usize] as usize;

        // Heaviest out-edge, ties broken by lowest successor rank.
        let mut best: Option<(i64, usize)> = None; // (weight, successor's rank)
        for &edge_id in &node.outedges {
            let edge = graph.edge(edge_id);
            let succ_rank = node_id_to_rank[edge.head.0 as usize] as usize;
            let better = match best {
                None => true,
                Some((best_weight, best_rank)) => {
                    edge.weight > best_weight
                        || (edge.weight == best_weight && succ_rank < best_rank)
                }
            };
            if better {
                best = Some((edge.weight, succ_rank));
            }
        }

        r[rank] = best.map_or(0, |(_, succ_rank)| 1 + r[succ_rank]);
    }
    r
}

/// Query-column anchor for a node with remaining-path length `r_len`: `clamp(L - r_len, 0, L)`
/// where `L = query_len`. `saturating_sub` alone already yields a value in `[0, L]` (it cannot
/// underflow past 0, and subtracting a non-negative amount from `L` cannot exceed `L`), so no
/// separate upper clamp is needed.
pub(crate) fn anchor(r_len: u32, query_len: usize) -> usize {
    query_len.saturating_sub(r_len as usize)
}

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
    let end = hi.saturating_add(w).saturating_add(1).min(query_len);
    (beg, end)
}

/// Segment (vector-lane) range `[beg_sn, end_sn)` covering `[beg, end)` query columns, clamped to
/// the row block and guaranteed **non-empty** (MINOR 6). `beg_sn` floors, `end_sn` ceils — so the
/// effective computed band is `[beg_sn*lanes, end_sn*lanes)`; the left-edge carry closure therefore
/// happens at `beg_sn*lanes`, which unit tests must target (not the unquantized `beg`).
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
        graph: &crate::graph::Graph,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_base_plus_rounded_fraction() {
        let cfg = BandConfig {
            base: 10,
            frac: 0.01,
        };
        assert_eq!(cfg.width(0), 10); // base only
        assert_eq!(cfg.width(235), 12); // 10 + round(2.35) = 10 + 2
        assert_eq!(cfg.width(1000), 20); // 10 + round(10.0)
    }

    #[test]
    fn width_saturates_and_never_panics() {
        // Huge/degenerate configs must clamp, not overflow or panic (MAJOR 7).
        let huge = BandConfig {
            base: u32::MAX,
            frac: f32::MAX,
        };
        let _ = huge.width(usize::MAX); // must not panic
        let neg = BandConfig {
            base: 5,
            frac: -1.0,
        };
        assert_eq!(neg.width(100), 5); // negative fraction floors to 0 contribution
        let nan = BandConfig {
            base: 7,
            frac: f32::NAN,
        };
        assert_eq!(nan.width(100), 7); // NaN -> 0 contribution
    }

    #[test]
    fn default_is_abpoa() {
        assert_eq!(BandConfig::default().base, 10);
        assert!((BandConfig::default().frac - 0.01).abs() < 1e-9);
    }

    // ---- remaining_path / anchor fixtures & tests --------------------------------------------

    use crate::graph::{Graph, NodeId};

    /// Inverts `graph.rank_order()` into a `node_id -> rank` lookup, the same shape
    /// `ScalarInit::node_id_to_rank` hands the real fill (see `sisd.rs`'s
    /// `node_id_to_rank[node_id.0 as usize] = rank as u32` seeding loop).
    fn node_id_to_rank_from(graph: &Graph) -> Vec<u32> {
        let mut node_id_to_rank = vec![0u32; graph.num_nodes()];
        for (rank, &node_id) in graph.rank_order().iter().enumerate() {
            node_id_to_rank[node_id.0 as usize] = rank as u32;
        }
        node_id_to_rank
    }

    /// Linear chain `A -> C -> G`, built the same way as `graph::tests::public_accessors_...`'s
    /// single fresh sequence: one `add_alignment` call with an empty alignment.
    fn linear_chain_3() -> (Graph, Vec<u32>) {
        let mut graph = Graph::new();
        graph.add_alignment(&[], b"ACG", &[1, 1, 1]).unwrap();
        let node_id_to_rank = node_id_to_rank_from(&graph);
        (graph, node_id_to_rank)
    }

    /// Diamond `A -> X -> Z -> C` (longer branch) and `A -> Y -> C` (shorter branch), both
    /// reconverging at `C`. The two edges leaving `A` (`A->X`, `A->Y`) carry equal weight, so
    /// `remaining_path` can only pick a successor deterministically by falling back to the
    /// lowest-ranked one. Node ids are assigned in add order: `A=0, X=1, Z=2, C=3, Y=4` (`Y` is
    /// the only node the second `add_alignment` call creates fresh; `A` and `C` are matched onto
    /// the existing nodes from the first sequence).
    fn diamond_equal_weights() -> (Graph, Vec<u32>) {
        let mut graph = Graph::new();
        graph.add_alignment(&[], b"AXZC", &[1, 1, 1, 1]).unwrap();
        graph
            .add_alignment(&[(0, 0), (-1, 1), (3, 2)], b"AYC", &[1, 1, 1])
            .unwrap();
        let node_id_to_rank = node_id_to_rank_from(&graph);
        (graph, node_id_to_rank)
    }

    #[test]
    fn remaining_path_counts_heaviest_successor_chain() {
        let (graph, node_id_to_rank) = linear_chain_3();
        let r = remaining_path(&graph, &node_id_to_rank);
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
        let (graph, node_id_to_rank) = diamond_equal_weights();
        let r = remaining_path(&graph, &node_id_to_rank);

        let a_rank = node_id_to_rank[NodeId(0).0 as usize] as usize; // A
        let x_rank = node_id_to_rank[NodeId(1).0 as usize] as usize; // X (longer branch)
        let y_rank = node_id_to_rank[NodeId(4).0 as usize] as usize; // Y (shorter branch)
        assert!(
            x_rank < y_rank,
            "fixture invariant: X must rank lower than Y for this to test the tie-break"
        );

        // The branches differ in length (X's is one node longer than Y's), so picking the
        // lower-ranked successor (X) versus the higher-ranked one (Y) is numerically
        // distinguishable here — this is what makes the assertion below a real tie-break check,
        // not a coincidence.
        assert_eq!(
            r[a_rank],
            1 + r[x_rank],
            "A must route through X (lower rank), not Y"
        );
        assert_ne!(
            r[a_rank],
            1 + r[y_rank],
            "A must not route through Y (higher rank)"
        );

        // Exact expected R, hand-derived from the fixture: C (sink) = 0; Y = 1; Z = 1;
        // X = 1 + R[Z] = 2; A = 1 + R[X] (tie-break picks X) = 3.
        let c_rank = node_id_to_rank[NodeId(3).0 as usize] as usize;
        let z_rank = node_id_to_rank[NodeId(2).0 as usize] as usize;
        assert_eq!(r[c_rank], 0);
        assert_eq!(r[y_rank], 1);
        assert_eq!(r[z_rank], 1);
        assert_eq!(r[x_rank], 2);
        assert_eq!(r[a_rank], 3);
    }

    // ---- node_window / segment_range / BandState ---------------------------------------------

    #[test]
    fn node_window_is_union_of_anchor_and_predecessors_widened() {
        // beg = max(0, min(Mstart, anchor) - w); end = min(L, max(Mend, anchor) + w + 1)
        let (beg, end) = node_window(
            /*anchor*/ 100, /*mstart*/ 90, /*mend*/ 110, /*w*/ 12,
            /*L*/ 235,
        );
        assert_eq!(beg, 90 - 12);
        assert_eq!(end, 110 + 12 + 1);
        // clamps at 0 and L
        let (b0, e0) = node_window(5, 5, 5, 12, 235);
        assert_eq!(b0, 0);
        let _ = e0;
        let (_, e_l) = node_window(230, 230, 230, 12, 235);
        assert_eq!(e_l, 235);
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

    #[test]
    fn segment_range_widens_when_end_sn_le_beg_sn() {
        // beg==end==L at the last segment: raw beg_sn=29 (clamped), raw end_sn=29 -> guard widens
        // end_sn to 30 so the range is non-empty (drops-whole-row-to-NEG_INF regression guard).
        assert_eq!(segment_range(232, 232, 8, 30), (29, 30));
    }

    #[test]
    fn band_state_new_builds_r_and_best_col_from_graph() {
        let (graph, node_id_to_rank) = linear_chain_3();
        let cfg = BandConfig {
            base: 10,
            frac: 0.01,
        };
        let state = BandState::new(&graph, &node_id_to_rank, 235, cfg);
        assert_eq!(state.r.len(), graph.num_nodes());
        assert_eq!(state.best_col.len(), state.r.len());
        assert!(state.best_col.iter().all(|&c| c == 0));
        assert_eq!(state.w, cfg.width(235));
        // r matches remaining_path directly.
        assert_eq!(state.r, remaining_path(&graph, &node_id_to_rank));
    }
}
