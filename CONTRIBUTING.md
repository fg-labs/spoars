# Contributing to spoars

## Development Setup

### Prerequisites

- Rust (stable toolchain; pinned via `rust-toolchain.toml`)
- [cargo-nextest](https://nexte.st/) (for running tests: `cargo ci-test`)
- A C++ toolchain and CMake — **only** needed to build the differential oracle
  used by the `oracle`/parity tests, which links the pinned `spoa` submodule.

Clone with submodules so the C++ oracle can build:

```bash
git clone --recurse-submodules https://github.com/fg-labs/spoars.git
# or, if already cloned:
git submodule update --init --recursive
```

### Install Git Hooks

We use pre-commit hooks to ensure code quality. Install them after cloning:

```bash
./scripts/install-hooks.sh
```

This installs hooks that run before each commit:
- `cargo ci-fmt` — Check code formatting
- `cargo ci-lint` — Run clippy lints

### Running Checks Manually

```bash
# Format check (fails if formatting differs)
cargo ci-fmt

# Lint check (fails on any warnings)
cargo ci-lint

# Run all tests (nextest)
cargo ci-test

# Run doctests (nextest does not execute these)
cargo test --doc
```

### Pre-Commit Hook Options

**Run tests in the pre-commit hook:**
```bash
SPOARS_PRECOMMIT_TEST=1 git commit -m "message"
```

**Bypass hooks (use sparingly):**
```bash
git commit --no-verify -m "message"
```

## Code Style

- Run `cargo fmt` before committing (`max_width = 100`)
- Fix all clippy warnings (`cargo ci-lint`)
- Add backticks around identifiers in doc comments

## Faithfulness contract

spoars reproduces spoa's output **bit-for-bit**. Any change touching the DP fill,
tie-breaks, consensus, MSA, or GFA/DOT export must keep every parity test green
(the SIMD kernels are validated against `SisdEngine`, which is validated against
the C++ `spoa` oracle). A change that alters alignment output is a bug unless it
is explicitly justified with updated parity tests explaining why it is correct.

## Testing

All new features should include tests. Generate test data programmatically — do
not commit test-data files. Run the full suite with:

```bash
cargo ci-test && cargo test --doc
```

## Pull Requests

1. Ensure all CI checks pass (`cargo ci-fmt`, `cargo ci-lint`, `cargo ci-test`, `cargo test --doc`)
2. Keep PRs focused and reasonably sized (250-1000 LOC ideal)
3. Include tests for new functionality
4. Use [conventional commits](https://www.conventionalcommits.org/)
