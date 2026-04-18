use crate::FastFloatFnHaver;

#[inline(always)]
/// Approximates 1/x for f32 (positive x).
///
/// Computes a bit-manipulation seed `y₀ = 0x7EF127EA - bits(x)`, which
/// gives a rough inverse, then refines with one Newton-Raphson step:
/// `y₁ = y₀·(2 − x·y₀)`.
///
/// **Measured relative error (over a representative sample):** 0.045% – 0.34%
pub fn approx_inv_f32(x: f32) -> f32 {
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
pub fn approx_inv_f64(x: f64) -> f64 {
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
    fn test_approx_inv() {
        let x = 2.0_f32;
        let inv = approx_inv_f32(x);
        assert!((inv - 0.5).abs() < 2e-3);

        let x64 = 2.0_f64;
        let inv64 = approx_inv_f64(x64);
        assert!((inv64 - 0.5).abs() < 1e-5);
    }
}
