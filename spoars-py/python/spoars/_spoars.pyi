"""Type stubs for the spoars `_spoars` extension module."""

from __future__ import annotations

from typing import Literal
from typing import overload

__version__: str

class Scoring:
    """Validated match/mismatch/gap scoring (spoa sign convention)."""

    def __init__(
        self,
        match: int,
        mismatch: int,
        gap_open: int,
        gap_extend: int,
        gap_open2: int,
        gap_extend2: int,
    ) -> None:
        """Create scoring; raises ``ValueError`` if a gap penalty is positive."""

    @staticmethod
    def default() -> Scoring:
        """The spoa/CLI default convex scoring (5, -4, -8, -6, -10, -4)."""

    def gap_mode(self) -> str:
        """The inferred gap model: ``"linear"``, ``"affine"``, or ``"convex"``."""

    def __repr__(self) -> str: ...

class Poa:
    """A partial order alignment graph builder."""

    def __init__(
        self,
        alignment_type: str = "global",
        scoring: Scoring | None = None,
    ) -> None:
        """Create a builder. ``alignment_type`` is ``"global"``/``"local"``/``"overlap"``."""

    def add(self, sequence: str, weight: int = 1) -> int:
        """Align and merge ``sequence``; return its 0-based sequence index."""

    # The consensus sequence (optionally pruning low-coverage nodes); with
    # `with_coverage=True`, returns `(consensus, per_base_coverage)` instead.
    @overload
    def consensus(
        self, min_coverage: int | None = None, with_coverage: Literal[False] = False
    ) -> str: ...
    @overload
    def consensus(
        self, min_coverage: int | None = None, with_coverage: Literal[True] = ...
    ) -> tuple[str, list[int]]: ...
    @overload
    def consensus(
        self, min_coverage: int | None = None, with_coverage: bool = False
    ) -> str | tuple[str, list[int]]: ...
    def consensus_composition(self) -> tuple[str, list[list[int]]]:
        """The consensus plus a per-column base-composition matrix (rows = codes + a gap row)."""

    def msa(self, include_consensus: bool = False) -> list[str]:
        """The multiple sequence alignment, one row per added sequence."""

    def gfa(
        self,
        headers: list[str] | None = None,
        is_reversed: list[bool] | None = None,
        include_consensus: bool = False,
    ) -> str:
        """The graph in GFA v1 format."""

    def dot(self) -> str:
        """The graph in Graphviz DOT format."""

    def subgraph(self, begin: int, end: int) -> tuple[Poa, list[int]]:
        """
        Extract the sub-graph over parent node ids ``begin..=end``.

        Returns the sub-graph as a new :class:`Poa` plus a list mapping each
        sub-graph node index to its parent node id. Node selection walks
        backwards from ``end`` (ancestors plus aligned siblings, id ``>= begin``),
        so a full-span window does not necessarily keep every node on a
        branching graph.
        """

    def num_nodes(self) -> int: ...
    def num_edges(self) -> int: ...
    def num_sequences(self) -> int: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def to_json(self) -> str:
        """Serialize the builder (graph + alignment type + scoring) to a JSON string."""

    @staticmethod
    def from_json(data: str) -> Poa:
        """Rebuild a :class:`Poa` from a :meth:`to_json` string."""

    def __getstate__(self) -> str: ...
    def __setstate__(self, state: str) -> None: ...
    def __getnewargs__(self) -> tuple[str]: ...

def poa(
    sequences: list[str],
    alignment_type: str = "global",
    scoring: Scoring | None = None,
    weights: list[int] | None = None,
) -> Poa:
    """Build a POA graph from ``sequences`` in one call and return the ``Poa``."""
