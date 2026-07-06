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
