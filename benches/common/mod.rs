//! Shared corpus generation for the POA microbenchmarks.
//!
//! Dependency-free and fully reproducible (fixed-seed xorshift64), so a `cargo bench`
//! run is deterministic across machines and repeats — no committed fixtures. The
//! generator models a UMI "family": one random truth molecule plus `n` mutant reads
//! carrying a small rate of substitutions *and* short indels. The indels matter — they
//! are what exercise spoars' convex-gap fill and the backtrack path the fgumi consumer
//! actually hits (SBX homopolymer errors); a substitution-only corpus would leave the
//! gap machinery cold and under-profile it.

// This module is shared into both `benches/poa.rs` and (via `#[path]`)
// `examples/profile_family.rs`. Each consumer uses a different subset, so some items are
// unused from any single crate's view — silence the resulting dead-code warnings here.
#![allow(dead_code)]

const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// Minimal xorshift64 PRNG — reproducible, no external deps (mirrors the generator in
/// `examples/profile_align.rs` / `tests/simd_bench.rs` so corpora are comparable).
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the PRNG. Any non-zero seed gives a full-period stream; `0` is remapped.
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // Top 53 bits → mantissa; classic construction.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn random_base(&mut self) -> u8 {
        BASES[(self.next_u64() % 4) as usize]
    }
}

/// A random ACGT molecule of length `len`.
pub fn random_molecule(len: usize, rng: &mut Rng) -> Vec<u8> {
    (0..len).map(|_| rng.random_base()).collect()
}

/// One mutant read of `truth`: each position may be substituted (`sub_rate`), deleted
/// (`indel_rate / 2`), or preceded by a random inserted base (`indel_rate / 2`). Rates
/// are per-base probabilities; defaults in [`family`] are deliberately modest so reads
/// stay recognizably related, as real UMI-family members are.
pub fn mutant(truth: &[u8], sub_rate: f64, indel_rate: f64, rng: &mut Rng) -> Vec<u8> {
    let ins = indel_rate / 2.0;
    let del = indel_rate / 2.0;
    let mut out = Vec::with_capacity(truth.len() + 8);
    for &b in truth {
        if rng.next_f64() < ins {
            out.push(rng.random_base());
        }
        let roll = rng.next_f64();
        if roll < del {
            continue; // deletion: drop this base
        } else if roll < del + sub_rate {
            out.push(rng.random_base()); // substitution (may coincide with truth base)
        } else {
            out.push(b);
        }
    }
    out
}

/// A UMI family: `(truth, reads)` where `reads` holds `n` mutants of `truth` at the
/// default modest error rates (1% substitution, 0.5% indel). Fixed `seed` → identical
/// family every run.
pub fn family(truth_len: usize, n: usize, seed: u64) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut rng = Rng::new(seed);
    let truth = random_molecule(truth_len, &mut rng);
    let reads = (0..n)
        .map(|_| mutant(&truth, 0.01, 0.005, &mut rng))
        .collect();
    (truth, reads)
}

/// The size × length grid the benches sweep: the real fgumi regime (3-10 reads, ~235 bp)
/// plus one larger point to expose where the small-N overhead stops dominating.
pub const REGIME: &[(usize, usize)] = &[
    (3, 235),
    (4, 235),
    (6, 235),
    (10, 235),
    (50, 1000), // scaling reference (assembly-ish); not the fgumi workload
];
