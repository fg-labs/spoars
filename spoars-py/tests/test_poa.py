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
