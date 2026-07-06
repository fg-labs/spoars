# Third-party software

## spoa

spoars is a faithful native-Rust reimplementation of **spoa** and reproduces its
output bit-for-bit. The published `spoars` crate contains **no** spoa source and
does not link it.

For development and testing only, this repository pins spoa as a git submodule at
`third_party/spoa`. It is compiled solely by the C++ *differential oracle* under
`oracle/`, which spoars' parity tests run against to prove bit-exactness. The
submodule is excluded from the published crate (see `exclude` in `Cargo.toml`) and
is not a runtime or build dependency of the library.

- Project: <https://github.com/rvaser/spoa> (Robert Vaser)
- License: MIT
- Used as: pinned submodule at `third_party/spoa`, compiled only by the
  test-time C++ oracle; the pinned commit is recorded by the submodule.

The spoa algorithm and its consensus/MSA/alignment semantics are described in:

> Vaser, R., Sović, I., Nagarajan, N., Šikić, M. (2017). Fast and accurate de novo
> genome assembly from long uncorrected reads. *Genome Research*, 27, 737-746.

This repository itself is licensed under the MIT License (see `LICENSE`).
