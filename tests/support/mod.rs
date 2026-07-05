//! Shared support code for spoars's differential-testing integration tests.
//!
//! This is a module directory (`tests/support/`), not a top-level file under
//! `tests/`, so it is NOT compiled as its own test binary. Each real test
//! binary (`tests/oracle_harness.rs` and later `graph_parity.rs`,
//! `engine_parity.rs`, `cli_parity.rs`) pulls this in via `mod support;` and
//! re-exports what it needs, giving all parity tests one shared
//! implementation of the oracle subprocess wrapper and input generators.
//!
//! Each individual `tests/*.rs` binary only exercises a slice of this
//! module's public API (e.g. `oracle_harness.rs` never calls `OracleCase::sw`
//! or `generators::small_dna`), so `dead_code` is allowed crate-wide here
//! rather than per item: the full surface is used collectively once the
//! later parity-test binaries land, and per-binary dead-code warnings about
//! not-yet-consumed shared helpers are expected, not a signal of a real bug.
#![allow(dead_code)]

pub mod generators;
pub mod oracle;
pub mod runner;
