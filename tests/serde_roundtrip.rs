//! Round-trip (de)serialization of a built `Graph` via the optional `serde` feature.
#![cfg(feature = "serde")]

use spoars::align::{AlignmentEngine, AlignmentType, Scoring, SisdEngine};
use spoars::graph::Graph;

fn build(seqs: &[&str]) -> Graph {
    let mut g = Graph::new();
    let mut engine = SisdEngine::new(AlignmentType::Global, Scoring::spoa_default());
    for s in seqs {
        let bytes = s.as_bytes();
        let (aln, _) = engine.align(bytes, &g);
        g.add_alignment_weight(&aln, bytes, 1).unwrap();
    }
    g
}

#[test]
fn json_round_trip_preserves_consensus_msa_and_gfa() {
    let mut original = build(&["ACGTACGT", "ACGAACGT", "ACGTAAGT", "ACGTACG"]);

    let json = serde_json::to_string(&original).expect("serialize");
    let mut restored: Graph = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(original.generate_consensus(), restored.generate_consensus());
    assert_eq!(original.generate_msa(true), restored.generate_msa(true));
    let headers: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
    let rev = [false; 4];
    assert_eq!(
        original.to_gfa(&headers, &rev, true),
        restored.to_gfa(&headers, &rev, true),
    );
    // Structural sanity: node/edge counts and the code alphabet survive.
    assert_eq!(original.num_nodes(), restored.num_nodes());
    assert_eq!(original.num_edges(), restored.num_edges());
    assert_eq!(original.num_codes(), restored.num_codes());
}

#[test]
fn deserialized_graph_encodes_bases_via_restored_coder() {
    // The [i32; 256] coder must survive so a restored graph can still encode bases.
    let g = build(&["ACGT", "ACGT"]);
    let json = serde_json::to_string(&g).unwrap();
    let restored: Graph = serde_json::from_str(&json).unwrap();
    for base in [b'A', b'C', b'G', b'T'] {
        assert_eq!(g.encode(base), restored.encode(base), "coder byte {base}");
    }
}
