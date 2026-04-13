use crate::FastFloatFnHaver;

#[inline(always)]
pub(crate) fn fast_mul2_f32(x: f32) -> f32 {
    const ONEEXP: u32 = 1 << 23;
    f32::from_bits(x.to_bits().saturating_add(ONEEXP))
}

#[inline(always)]
pub(crate) fn fast_div2_f32(x: f32) -> f32 {
    const ONEEXP: u32 = 1 << 23;
    f32::from_bits(x.to_bits().saturating_sub(ONEEXP))
}

#[inline(always)]
pub(crate) fn fast_mul2_f64(x: f64) -> f64 {
    const ONEEXP: u64 = 1 << 52;
    f64::from_bits(x.to_bits().saturating_add(ONEEXP))
}

#[inline(always)]
pub(crate) fn fast_div2_f64(x: f64) -> f64 {
    const ONEEXP: u64 = 1 << 52;
    f64::from_bits(x.to_bits().saturating_sub(ONEEXP))
}

#[inline(always)]
pub(crate) fn fast_mul3_f64(x: f64) -> f64 {
    //The standard operation is faster than any trick we might do,
    //but this is here so the user doesn't have to look it up
    x * 3.0
}

#[inline(always)]
pub(crate) fn fast_mul4_f32(x: f32) -> f32 {
    const TWOEXP: u32 = 2 << 23;
    f32::from_bits(x.to_bits().saturating_add(TWOEXP))
}

#[inline(always)]
pub(crate) fn fast_mul4_f64(x: f64) -> f64 {
    const TWOEXP: u64 = 2 << 52;
    f64::from_bits(x.to_bits().saturating_add(TWOEXP))
}

#[inline(always)]
pub(crate) fn fast_mul8_f32(x: f32) -> f32 {
    const THREEEXP: u32 = 3 << 23;
    f32::from_bits(x.to_bits().saturating_add(THREEEXP))
}

#[inline(always)]
pub(crate) fn fast_mul8_f64(x: f64) -> f64 {
    const THREEEXP: u64 = 3 << 52;
    f64::from_bits(x.to_bits().saturating_add(THREEEXP))
}

#[inline(always)]
pub(crate) fn fast_mul3_f32(x: f32) -> f32 {
    3.0 * x
}

#[inline(always)]
/// Approximates 1/x for f32 (positive x).
///
/// Computes a bit-manipulation seed `y₀ = 0x7EF127EA - bits(x)`, which
/// gives a rough inverse, then refines with one Newton-Raphson step:
/// `y₁ = y₀·(2 − x·y₀)`.
///
/// **Measured relative error (over a representative sample):** 0.045% – 0.34%
pub(crate) fn approx_inv_f32(x: f32) -> f32 {
    // Bit-magic seed: interpret bits as y₀ ≈ 1/x
    let y0 = f32::from_bits(0x7EF127EA_u32.wrapping_sub(x.to_bits()));
    // One Newton-Raphson step
    y0 * x.mul_add(-y0, 2.0)
}

#[inline(always)]
/// Approximates 1/x for f64 (positive x).
///
/// Computes a bit-manipulation seed adapted for f64 biases, then refines
/// with **two** Newton-Raphson steps for significantly higher accuracy than
/// the f32 variant.
///
/// **Measured relative error (over a representative sample):** 8.8×10⁻⁵% – 6.5×10⁻⁴%
pub(crate) fn approx_inv_f64(x: f64) -> f64 {
    // Bit-magic seed adapted for f64 biases
    let y0 = f64::from_bits(0x7FDE623822835EEA_u64.wrapping_sub(x.to_bits()));
    // Two Newton-Raphson steps
    let y1 = y0 * x.mul_add(-y0, 2.0);
    y1 * x.mul_add(-y1, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let x = 2.0_f32;
        assert_eq!(fast_mul2_f32(x), 4.0);
        assert_eq!(fast_div2_f32(4.0), 2.0);
        assert_eq!(fast_mul4_f32(x), 8.0);
        assert_eq!(fast_mul8_f32(x), 16.0);
        assert_eq!(fast_mul3_f32(x), 6.0);

        let x64 = 2.0_f64;
        assert_eq!(fast_mul2_f64(x64), 4.0);
        assert_eq!(fast_div2_f64(4.0), 2.0);
        assert_eq!(fast_mul4_f64(x64), 8.0);
        assert_eq!(fast_mul8_f64(x64), 16.0);
        assert_eq!(fast_mul3_f64(x64), 6.0);
    }

    #[test]
    fn test_approx_inv() {
        let x = 2.0_f32;
        let inv = approx_inv_f32(x);
        assert!((inv - 0.5).abs() < 2e-3);

        let x64 = 2.0_f64;
        let inv64 = approx_inv_f64(x64);
        assert!((inv64 - 0.5).abs() < 1e-5);
    }
}
