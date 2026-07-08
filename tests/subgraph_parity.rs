//! Differential parity of `Graph::subgraph` against the C++ spoa oracle.
mod support;

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SisdEngine};
use spoars::graph::{Graph, NodeId};
use support::oracle::{run_oracle, OracleCase, OracleResult};

/// Builds a spoars graph from `seqs` by aligning each with a fresh [`SisdEngine`] of the given
/// `alignment_type` (spoa-default scoring) and folding the result into the graph in order —
/// mirrors what `oracle/spoa_oracle.cpp` does before it calls `Graph::Subgraph`, so the graph
/// this produces is directly comparable to the oracle's.
fn build_with_type(seqs: &[&str], alignment_type: AlignmentType) -> Graph {
    let mut g = Graph::new();
    let mut engine = SisdEngine::new(alignment_type, Scoring::spoa_default());
    for s in seqs {
        let bytes = s.as_bytes();
        let (aln, _) = engine.align(bytes, &g);
        g.add_alignment_weight(&aln, bytes, 1).unwrap();
    }
    g
}

/// Global (NW)-aligned build, the mode exercised by most of this file's tests.
fn build(seqs: &[&str]) -> Graph {
    build_with_type(seqs, AlignmentType::Global)
}

/// Runs `g.subgraph(begin, end)` and asserts every field matches the oracle's own
/// `spoa::Graph::Subgraph` output for the equivalent request.
///
/// `case` must already carry the same sequences (and `ty`) used to build `g` — callers get
/// this by constructing `case` with `OracleCase::nw`/`::sw`/`::ov` and building `g` via
/// `build`/`build_with_type` with the matching [`AlignmentType`]. This helper overwrites
/// `case.subgraph` with `(begin, end)` and takes ownership since `run_oracle` needs it moved
/// into a one-element batch.
fn assert_subgraph_matches(g: &Graph, mut case: OracleCase, begin: u32, end: u32) {
    let (sub, map) = g.subgraph(NodeId(begin), NodeId(end));

    case.subgraph = Some((begin, end));
    let res: &OracleResult = &run_oracle(&[case])[0];

    // Map: subgraph id -> parent id.
    let my_map: Vec<u32> = map.iter().map(|n| n.0).collect();
    assert_eq!(
        my_map, res.subgraph_map,
        "map mismatch begin={begin} end={end}"
    );
    // Codes per subgraph node id.
    let my_codes: Vec<u32> = sub.nodes().iter().map(|nd| nd.code).collect();
    assert_eq!(
        my_codes, res.subgraph_codes,
        "codes mismatch begin={begin} end={end}"
    );
    // Edge set + weights in arena order.
    let my_edges: Vec<(u32, u32, i64)> = sub
        .edges()
        .iter()
        .map(|e| (e.tail.0, e.head.0, e.weight))
        .collect();
    assert_eq!(
        my_edges, res.subgraph_edges,
        "edges mismatch begin={begin} end={end}"
    );
    // Topological order.
    let my_rank: Vec<u32> = sub.rank_order().iter().map(|n| n.0).collect();
    assert_eq!(
        my_rank, res.subgraph_rank,
        "rank mismatch begin={begin} end={end}"
    );
    // Aligned-node groups.
    let mut my_aligned: Vec<(u32, u32)> = sub
        .nodes()
        .iter()
        .enumerate()
        .flat_map(|(i, nd)| nd.aligned_nodes.iter().map(move |a| (i as u32, a.0)))
        .collect();
    let mut oracle_aligned = res.subgraph_aligned.clone();
    my_aligned.sort_unstable();
    oracle_aligned.sort_unstable();
    assert_eq!(
        my_aligned, oracle_aligned,
        "aligned mismatch begin={begin} end={end}"
    );
    // Invariant: every subgraph edge is labeled 0 (sequences not copied).
    for e in sub.edges() {
        assert_eq!(e.labels, vec![0]);
    }
}

#[test]
fn subgraph_matches_oracle_on_small_families() {
    let families: &[&[&str]] = &[
        &["ACGT", "AGT", "ACGGT"],
        &["ACGTACGT", "ACGAACGT", "ACGTAAGT", "ACGTACG"],
        &["GATTACA", "GATTTACA", "GATACA", "GACTACA"],
    ];
    for seqs in families {
        let g = build(seqs);
        let n = g.num_nodes() as u32;
        // A few representative windows: whole graph, middle, tail.
        for &(begin, end) in &[(0u32, n - 1), (n / 4, (3 * n) / 4), (n / 2, n - 1)] {
            assert_subgraph_matches(&g, OracleCase::nw(seqs), begin, end);
        }
    }
}

#[test]
fn subgraph_matches_oracle_across_alignment_modes() {
    let seqs: &[&str] = &["GATTACA", "GATTTACA", "GATACA", "GACTACA"];
    let cases: Vec<(AlignmentType, OracleCase)> = vec![
        (AlignmentType::Global, OracleCase::nw(seqs)),
        (AlignmentType::Local, OracleCase::sw(seqs)),
        (AlignmentType::Overlap, OracleCase::ov(seqs)),
    ];
    for (alignment_type, case) in cases {
        let g = build_with_type(seqs, alignment_type);
        let n = g.num_nodes() as u32;
        let (begin, end) = (n / 4, (3 * n) / 4);
        assert_subgraph_matches(&g, case, begin, end);
    }
}
