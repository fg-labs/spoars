import pickle

import pytest
import spoars


def test_poa_convenience_consensus_and_msa() -> None:
    g = spoars.poa(["ACGTACGT", "ACGTTCGT", "ACGTACGT"])
    assert g.consensus() == "ACGTACGT"
    msa = g.msa()
    assert len(msa) == 3
    assert all(len(row) == len(msa[0]) for row in msa)
    assert g.num_sequences() == 3
    assert len(g) == 3


def test_identical_sequences_collapse_onto_one_path() -> None:
    g = spoars.poa(["ACGT", "ACGT", "ACGT"])
    assert g.consensus() == "ACGT"
    # All three identical reads share one 4-node path.
    assert g.num_nodes() == 4


def test_builder_add_returns_sequence_index() -> None:
    g = spoars.Poa(alignment_type="global")
    assert g.add("ACGT") == 0
    assert g.add("ACGT") == 1
    assert g.add("ACGT") == 2
    assert g.num_sequences() == 3


def test_msa_include_consensus_adds_a_row() -> None:
    g = spoars.poa(["ACGTACGT", "ACGTTCGT"])
    assert len(g.msa(include_consensus=False)) == 2
    assert len(g.msa(include_consensus=True)) == 3


@pytest.mark.parametrize("alignment_type", ["global", "local", "overlap", "NW", "SW", "OV"])
def test_alignment_types_accepted(alignment_type: str) -> None:
    g = spoars.poa(["ACGT", "ACGT"], alignment_type=alignment_type)
    assert g.num_sequences() == 2


def test_unknown_alignment_type_raises() -> None:
    with pytest.raises(ValueError):
        spoars.Poa(alignment_type="diagonal")


def test_scoring_default_is_convex() -> None:
    assert spoars.Scoring.default().gap_mode() == "convex"


def test_scoring_linear_and_custom() -> None:
    linear = spoars.Scoring(1, -1, -1, -1, -1, -1)
    assert linear.gap_mode() == "linear"
    g = spoars.poa(["ACGT", "ACTT"], scoring=linear)
    assert g.num_sequences() == 2


def test_scoring_rejects_positive_gap_penalty() -> None:
    with pytest.raises(ValueError):
        spoars.Scoring(1, -1, 5, -1, -1, -1)


def test_weights_must_match_sequence_count() -> None:
    with pytest.raises(ValueError):
        spoars.poa(["ACGT", "ACGT"], weights=[1])


def test_gfa_and_dot_render() -> None:
    g = spoars.poa(["ACGT", "ACGT"])
    gfa = g.gfa()
    assert gfa.startswith("H\t")
    assert "P\t" in gfa  # path lines for the two sequences
    dot = g.dot()
    assert "digraph" in dot


def test_gfa_custom_headers_length_checked() -> None:
    g = spoars.poa(["ACGT", "ACGT"])
    assert "read0" in g.gfa(headers=["read0", "read1"])
    with pytest.raises(ValueError):
        g.gfa(headers=["only-one"])


def test_min_coverage_consensus() -> None:
    # A minority variant read; a high min-coverage prunes low-coverage nodes.
    g = spoars.poa(["ACGTACGT", "ACGTACGT", "ACGTACGT", "TTTTTTTT"])
    assert g.consensus(min_coverage=2) == "ACGTACGT"


def test_version_is_exposed() -> None:
    assert isinstance(spoars.__version__, str)
    assert spoars.__version__


def test_subgraph_full_span_of_a_linear_graph_keeps_every_node() -> None:
    # Identical sequences collapse onto a single unbranched path (no aligned-node
    # siblings), so the last node's ancestors are every node in the graph and a
    # full-span window is a faithful copy.
    g = spoars.poa(["ACGTACGT", "ACGTACGT", "ACGTACGT"])
    n = g.num_nodes()
    sub, mapping = g.subgraph(0, n - 1)
    assert sub.num_nodes() == n
    assert len(mapping) == n
    assert mapping == list(range(n))  # ascending parent ids for a full span
    # Sub-Poa is usable for inspection (GFA export of a window).
    assert sub.gfa().startswith("H\t")


def test_subgraph_narrow_window_is_a_subset() -> None:
    # `subgraph` walks backward (ancestors) from `end`, so with a branching graph the
    # sub-graph is a subset even at a full-id window; narrow it further here to also
    # exercise the `begin` lower bound.
    g = spoars.poa(["ACGTACGT", "ACGAACGT", "ACGTAAGT"])
    n = g.num_nodes()
    sub, mapping = g.subgraph(n // 2, n - 1)
    assert sub.num_nodes() < n
    assert len(mapping) == sub.num_nodes()
    assert all(n // 2 <= parent <= n - 1 for parent in mapping)


def test_subgraph_on_a_branching_graph_keeps_only_ancestors_of_end() -> None:
    # A window spanning every node id still yields a sub-graph, not the whole graph,
    # because `subgraph` only keeps ancestors of `end` (plus their aligned siblings)
    # that fall within the id window -- nodes downstream of `end` are excluded even
    # when their id is < end.
    g = spoars.poa(["ACGTACGT", "ACGAACGT", "ACGTAAGT"])
    n = g.num_nodes()
    sub, mapping = g.subgraph(0, n - 1)
    assert sub.num_nodes() < n
    assert len(mapping) == sub.num_nodes()
    assert all(0 <= parent <= n - 1 for parent in mapping)
    assert mapping == sorted(mapping)  # ascending parent ids (arena iteration order)


def test_subgraph_out_of_range_raises_value_error() -> None:
    # An out-of-range node id must raise a catchable ValueError, not surface the
    # underlying Rust panic as an (uncatchable) PanicException.
    g = spoars.poa(["ACGT", "ACGT"])
    n = g.num_nodes()
    with pytest.raises(ValueError):
        g.subgraph(0, n)  # end == num_nodes is out of range
    with pytest.raises(ValueError):
        g.subgraph(n, 0)  # begin out of range


def test_consensus_with_coverage_returns_per_base_totals() -> None:
    g = spoars.poa(["ACGT", "ACGT", "AGGT"])
    plain = g.consensus()
    cons, cov = g.consensus(with_coverage=True)
    assert cons == plain
    assert len(cov) == len(cons)
    assert all(c >= 1 for c in cov)


def test_consensus_composition_matrix_shape_and_column_sums() -> None:
    seqs = ["ACGT", "ACGT", "AGGT"]
    g = spoars.poa(seqs)
    cons, matrix = g.consensus_composition()
    assert len(matrix) >= 1
    stride = len(cons)
    assert all(len(row) == stride for row in matrix)
    # Each column sums to at most the number of sequences.
    for col in range(stride):
        assert sum(row[col] for row in matrix) <= len(seqs)


def test_consensus_composition_on_empty_poa_is_empty() -> None:
    # An empty graph has a zero-length consensus (stride == 0); the reshape must not
    # panic and must return an empty consensus with an empty matrix.
    g = spoars.Poa()
    cons, matrix = g.consensus_composition()
    assert cons == ""
    assert matrix == []


def test_json_round_trip_preserves_consensus_msa_gfa() -> None:
    g = spoars.poa(["ACGTACGT", "ACGAACGT", "ACGTAAGT"])
    restored = spoars.Poa.from_json(g.to_json())
    assert restored.consensus() == g.consensus()
    assert restored.msa() == g.msa()
    assert restored.gfa() == g.gfa()
    assert restored.num_nodes() == g.num_nodes()


def test_pickle_round_trip_preserves_graph_and_stays_functional() -> None:
    g = spoars.Poa(alignment_type="global", scoring=spoars.Scoring.default())
    for read in ["ACGTACGT", "ACGAACGT", "ACGTAAGT"]:
        g.add(read)
    restored = pickle.loads(pickle.dumps(g))
    assert restored.consensus() == g.consensus()
    assert restored.msa() == g.msa()
    assert restored.gfa() == g.gfa()
    # A restored Poa is fully functional: its engine (alignment type + scoring) survived,
    # so further reads align the same way.
    restored.add("ACGTACGT")
    g.add("ACGTACGT")
    assert restored.consensus() == g.consensus()


def test_from_json_rejects_malformed_input() -> None:
    # Malformed JSON surfaces as a catchable ValueError, not a panic.
    with pytest.raises(ValueError):
        spoars.Poa.from_json("not json at all")


def test_round_trip_preserves_non_default_engine_for_further_alignments() -> None:
    """
    Round-tripping must carry over the non-default alignment type and scoring.

    The reads below have divergent flanks around a shared core, and `new_read` (added
    *after* restoring) has its own pair of flanks that don't match the core's flanks in
    any of the three reads. Under local alignment those flanks are free overhangs
    (soft-clipped: they add disconnected nodes to the graph but don't force new aligned
    columns), while under the default global alignment every base -- including the
    flanks -- must be aligned end-to-end (forced into the existing columns as
    mismatches). A flank-free `new_read` would align identically under both engines
    (nothing to clip), exercising only the scoring difference; giving it non-matching
    flanks makes the alignment-type dimension load-bearing too. That difference is only
    visible once we align a *further* read after restoring, so a restored engine that
    quietly reconstructed with default parameters would still reproduce the pre-restore
    consensus/MSA but diverge here.
    """
    reads = ["TTTTACGTACGTACGTGGGG", "CCCCACGTACGTACGTAAAA", "GGGGACGTACGTACGTCCCC"]
    new_read = "AAAAACGTACGTACGTTTTT"
    scoring = spoars.Scoring(3, -2, -5, -3, -8, -2)

    original = spoars.Poa(alignment_type="local", scoring=scoring)
    for read in reads:
        original.add(read)

    restored_via_pickle = pickle.loads(pickle.dumps(original))
    restored_via_json = spoars.Poa.from_json(original.to_json())

    for restored in (restored_via_pickle, restored_via_json):
        restored.add(new_read)
    original.add(new_read)

    assert restored_via_pickle.consensus() == original.consensus()
    assert restored_via_pickle.msa() == original.msa()
    assert restored_via_json.consensus() == original.consensus()
    assert restored_via_json.msa() == original.msa()

    # Discrimination check: a default-engine Poa (global alignment, default scoring)
    # given the identical reads must diverge from the non-default result above --
    # otherwise this test wouldn't actually distinguish a correct restore from a
    # buggy one that ignores the round-tripped alignment type + scoring.
    default_engine = spoars.Poa()
    for read in reads:
        default_engine.add(read)
    default_engine.add(new_read)

    assert (
        default_engine.consensus() != original.consensus() or default_engine.msa() != original.msa()
    )


def test_node_and_scalar_accessors_are_consistent() -> None:
    reads = ["ACGTACGT", "ACGAACGT", "ACGTAAGT"]
    g = spoars.poa(reads)
    n = g.num_nodes()
    # rank_order is a permutation of all node ids.
    assert sorted(g.rank_order()) == list(range(n))
    # encode/decode round-trip for every seen base.
    for base in "ACGT":
        code = g.encode(base)
        assert code is not None
        assert g.decode(code) == base
    assert g.encode("Z") is None  # unseen base
    assert g.num_codes() == 4
    # sequence_path reconstructs each read's bases in order.
    for i, read in enumerate(reads):
        path = g.sequence_path(i)
        assert "".join(g.node_base(node) or "" for node in path) == read
    # sequence_starts[i] is the first node of sequence_path(i).
    starts = g.sequence_starts()
    assert [g.sequence_path(i)[0] for i in range(len(reads))] == starts
    # node_coverage is >= 1 for every node; node_labels length == coverage.
    for node in range(n):
        cov = g.node_coverage(node)
        assert cov >= 1
        assert len(g.node_labels(node)) == cov
    # consensus_nodes decode to the consensus string.
    g.consensus()  # populate the consensus path
    assert "".join(g.node_base(x) or "" for x in g.consensus_nodes()) == g.consensus()


def test_node_successor_walks_a_sequence() -> None:
    g = spoars.poa(["ACGT", "ACGT"])
    # Walking node_successor from the sequence start reproduces sequence_path(0).
    walked: list[int] = []
    node: int | None = g.sequence_starts()[0]
    while node is not None:
        walked.append(node)
        node = g.node_successor(node, 0)
    assert walked == g.sequence_path(0)


def test_inspection_accessors_reject_out_of_range() -> None:
    g = spoars.poa(["ACGT", "ACGT"])
    n = g.num_nodes()
    for call in (
        lambda: g.node_code(n),
        lambda: g.node_coverage(n),
        lambda: g.node_base(n),
        lambda: g.node_successor(n, 0),
        lambda: g.node_labels(n),
        lambda: g.sequence_path(g.num_sequences()),
    ):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):
        g.encode("AC")  # not a single character
