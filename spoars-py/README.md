# spoars (Python)

Python bindings for [spoars](https://github.com/fg-labs/spoars) — a faithful,
SIMD-accelerated native-Rust reimplementation of the spoa partial order alignment
(POA) library, for consensus generation and multiple sequence alignment.

## Install

```bash
pip install spoars
```

## Usage

```python
import spoars

# One-call convenience:
g = spoars.poa(["ACGTACGT", "ACGTTCGT", "ACGTACGT"])
print(g.consensus())  # "ACGTACGT"
print(g.msa())  # ['ACGTACGT', 'ACGTTCGT', 'ACGTACGT']

# Or build incrementally, with an alignment type and scoring:
g = spoars.Poa(alignment_type="global", scoring=spoars.Scoring.default())
for read in ["ACGTACGT", "ACGTTCGT", "ACGTACGT"]:
    g.add(read)
print(g.consensus(min_coverage=2))
print(g.gfa())  # GFA v1
print(g.dot())  # Graphviz DOT

# Consensus with per-base total coverage, or the per-column base composition:
consensus, coverage = g.consensus(with_coverage=True)  # (str, list[int])
consensus, matrix = g.consensus_composition()  # rows = codes + a trailing gap row
# Cache or transmit a graph — pickle, or JSON via to_json / from_json:
import pickle

restored = pickle.loads(pickle.dumps(g))
restored = spoars.Poa.from_json(g.to_json())
```

`alignment_type` is one of `"global"`, `"local"`, or `"overlap"`. `Scoring` takes
`(match, mismatch, gap_open, gap_extend, gap_open2, gap_extend2)`; the gap model
(linear/affine/convex) is inferred, and `Scoring.default()` is spoa's convex
default `(5, -4, -8, -6, -10, -4)`.

## Development

This package is built with [maturin](https://www.maturin.rs/) and developed with
[pixi](https://pixi.sh/):

```bash
pixi run develop      # build & install the extension into the dev env
pixi run test         # rebuild + pytest
pixi run format-check
pixi run lint
pixi run typecheck
```

## License

MIT — see the [repository](https://github.com/fg-labs/spoars).
