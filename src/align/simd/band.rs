//! Band geometry for the opt-in, heuristic abPOA-style banded alignment mode.
//!
//! Everything here is pure and `LANES`-independent so it can be unit-tested without a
//! SIMD backend and produces identical bands on every ISA. See
//! `docs/design/2026-07-06-banded-poa-alignment-design.md`.

/// Adaptive-band configuration (abPOA-style). APPROXIMATE: banded alignment may miss the
/// optimal path when it needs an indel larger than the band. `SimdEngine::new` stays exact
/// (bit-exact with spoa); use this only when the speed/accuracy trade-off is acceptable
/// (near-identical reads).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandConfig {
    /// Constant half-width added to every band, in query columns.
    pub base: u32,
    /// Fraction of the query length added to the half-width (`round(frac * L)`).
    pub frac: f32,
}

impl Default for BandConfig {
    fn default() -> Self {
        BandConfig {
            base: 10,
            frac: 0.01,
        }
    }
}

impl BandConfig {
    /// Per-align half-width `w = base + round(frac * L)`, computed in `usize` and **saturating**
    /// so no config can overflow or panic. Negative/NaN `frac` contributes 0. A width `>= L`
    /// means "no effective band" (used only by the smoke test); production values are small.
    pub fn width(&self, query_len: usize) -> usize {
        let frac_cols = (f64::from(self.frac) * query_len as f64).round();
        // A negative or NaN product yields 0 columns; a huge product saturates at usize::MAX.
        let frac_cols = if frac_cols.is_finite() && frac_cols > 0.0 {
            frac_cols as usize // saturating float->int cast (Rust: clamps, NaN->0)
        } else {
            0
        };
        (self.base as usize).saturating_add(frac_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_base_plus_rounded_fraction() {
        let cfg = BandConfig {
            base: 10,
            frac: 0.01,
        };
        assert_eq!(cfg.width(0), 10); // base only
        assert_eq!(cfg.width(235), 12); // 10 + round(2.35) = 10 + 2
        assert_eq!(cfg.width(1000), 20); // 10 + round(10.0)
    }

    #[test]
    fn width_saturates_and_never_panics() {
        // Huge/degenerate configs must clamp, not overflow or panic (MAJOR 7).
        let huge = BandConfig {
            base: u32::MAX,
            frac: f32::MAX,
        };
        let _ = huge.width(usize::MAX); // must not panic
        let neg = BandConfig {
            base: 5,
            frac: -1.0,
        };
        assert_eq!(neg.width(100), 5); // negative fraction floors to 0 contribution
        let nan = BandConfig {
            base: 7,
            frac: f32::NAN,
        };
        assert_eq!(nan.width(100), 7); // NaN -> 0 contribution
    }

    #[test]
    fn default_is_abpoa() {
        assert_eq!(BandConfig::default().base, 10);
        assert!((BandConfig::default().frac - 0.01).abs() < 1e-9);
    }
}
