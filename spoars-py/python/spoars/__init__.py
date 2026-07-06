"""
Python bindings for spoars — partial order alignment (POA) consensus and MSA.

spoars is a faithful, SIMD-accelerated native-Rust reimplementation of spoa. This
package wraps it with a small, Pythonic API.

Classes:
    Poa      — a POA graph builder: add sequences, then read consensus/MSA/GFA/DOT.
    Scoring  — validated match/mismatch/gap penalties (see ``Scoring.default``).

Functions:
    poa      — build a graph from a list of sequences in one call.

Example:
    >>> import spoars
    >>> g = spoars.poa(["ACGTACGT", "ACGTTCGT", "ACGTACGT"])
    >>> g.consensus()
    'ACGTACGT'
    >>> len(g.msa())
    3
"""

from ._spoars import Poa
from ._spoars import Scoring
from ._spoars import __version__
from ._spoars import poa

__all__ = ["Poa", "Scoring", "poa", "__version__"]
