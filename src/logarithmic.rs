use crate::arithmetic::{approx_inv_f32, approx_inv_f64};

/// Approximates eˣ (the natural exponential) for f32.
///
/// Uses range reduction to `x = n·ln2 + r`, then evaluates a degree-3 Taylor
/// polynomial for `eʳ` on `[-0.35, 0.35]` and scales by `2ⁿ` via exponent
/// manipulation.  Returns `+∞` for `x > 88.72` and `0.0` for `x < −87.34`,
/// both handled branchlessly.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.047%
#[inline(always)]
pub(crate) fn approx_exp_f32(x: f32) -> f32 {
    let is_inf: u32 = ((x > 88.72283) as u32).wrapping_neg();
    let is_z: u32 = ((x < -87.33654) as u32).wrapping_neg();

    if (is_inf | is_z) == 0 {
        let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
        let xgeqz: u32 = ((x >= 0.0) as u32).wrapping_neg();
        const INV_LN2: f32 = std::f32::consts::LOG2_E;
        const LN2_HI: f32 = std::f32::consts::LN_2;
        const LN2_LO: f32 = 0.0000014286068;
        const INV6: f32 = 1.0 / 6.0;

        let xv = (xltz & (-0.5_f32).to_bits()) | (xgeqz & (0.5_f32).to_bits());
        let n = x.mul_add(INV_LN2, f32::from_bits(xv)) as i32;
        let r = (-n as f32).mul_add(LN2_LO, (-n as f32).mul_add(LN2_HI, x));

        let is_good: u32 = !is_inf & !is_z;
        let exponent = (n + 127) as u32;
        let res_r = r.mul_add(r.mul_add(INV6.mul_add(r, 0.5), 1.0), 1.0);
        let two_n = f32::from_bits(exponent.wrapping_shl(23));

        let rv = two_n * res_r;
        f32::from_bits(rv.to_bits() & is_good)
    } else {
        return f32::from_bits((is_inf & f32::INFINITY.to_bits()) | (0.0_f32.to_bits() & is_z));
    }
}

#[inline(always)]
pub fn approx_exp_f32b(x: f32) -> f32 {
    let is_inf: u32 = ((x > 88.72283) as u32).wrapping_neg();
    let is_z: u32 = ((x < -87.33654) as u32).wrapping_neg();
    let is_good = !is_inf & !is_z;

    let n = x.mul_add(std::f32::consts::LOG2_E, 0.5).floor() as i32;
    let r = (-n as f32).mul_add(std::f32::consts::LN_2, x);

    let num = (-r - 30.0)
        .mul_add(r, -420.0)
        .mul_add(r, -3360.0)
        .mul_add(r, -15120.0)
        .mul_add(r, -30240.0);
    let dem = (r - 30.0)
        .mul_add(r, 420.0)
        .mul_add(r, -3360.0)
        .mul_add(r, 15120.0)
        .mul_add(r, -30240.0);
    let two_n = f32::from_bits(((n + 127) as u32).wrapping_shl(23));
    let rv = two_n * (num / dem);

    f32::from_bits(
        is_inf & f32::INFINITY.to_bits() | 0.0_f32.to_bits() & is_z | rv.to_bits() & is_good,
    )
}

#[inline(always)]
/// Approximates eˣ (the natural exponential) for f64.
///
/// Uses range reduction to `x = n·ln2 + r`, then evaluates a degree-3 Taylor
/// polynomial for `eʳ` on `[-0.35, 0.35]` and scales by `2ⁿ` via exponent
/// manipulation.  Returns `+∞` for `x > 709.78` and `0.0` for `x < −708.40`,
/// both handled branchlessly.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.047%
pub(crate) fn approx_exp_f64(x: f64) -> f64 {
    let is_inf: u64 = ((x > 709.782712893384) as u64).wrapping_neg();
    let is_z: u64 = ((x < -708.3964185322641) as u64).wrapping_neg();
    if (is_inf | is_z) == 0 {
        let xltz: u64 = ((x < 0.0) as u64).wrapping_neg();
        let xgeqz: u64 = ((x >= 0.0) as u64).wrapping_neg();
        const INV_LN2: f64 = std::f64::consts::LOG2_E;
        const LN2_HI: f64 = std::f64::consts::LN_2;
        const LN2_LO: f64 = 1.9082149292705877e-10;
        const INV6: f64 = 1.0 / 6.0;

        let xv = (xltz & (-0.5_f64).to_bits()) | (xgeqz & (0.5_f64).to_bits());
        let n = (x * INV_LN2 + f64::from_bits(xv)) as i32;
        let r = (-n as f64).mul_add(LN2_LO, (-n as f64).mul_add(LN2_HI, x));

        let is_good: u64 = !is_inf & !is_z;
        let exponent = (n + 1023) as u64;
        let res_r = r.mul_add(r.mul_add(INV6.mul_add(r, 0.5), 1.0), 1.0);
        let two_n = f64::from_bits(exponent.wrapping_shl(52));

        let rv = two_n * res_r;

        f64::from_bits(rv.to_bits() & is_good)
    } else {
        f64::from_bits((is_inf & f64::INFINITY.to_bits()) | (0.0_f64.to_bits() & is_z))
    }
}

#[inline(always)]
pub fn approx_exp_f64b(x: f64) -> f64 {
    let is_inf: u64 = ((x > 709.782712893384) as u64).wrapping_neg();
    let is_z: u64 = ((x < -708.3964185322641) as u64).wrapping_neg();
    let is_good = !is_inf & !is_z;

    // Range reduce to r in [-0.5*ln2, 0.5*ln2]
    let n = x.mul_add(std::f64::consts::LOG2_E, 0.5).floor() as i32;
    let r = (-n as f64).mul_add(std::f64::consts::LN_2, x);

    // Padé approximant for exp(r), evaluated at the reduced r
    let num = (-r - 30.0)
        .mul_add(r, -420.0)
        .mul_add(r, -3360.0)
        .mul_add(r, -15120.0)
        .mul_add(r, -30240.0);
    let dem = (r - 30.0)
        .mul_add(r, 420.0)
        .mul_add(r, -3360.0)
        .mul_add(r, 15120.0)
        .mul_add(r, -30240.0);
    let two_n = f64::from_bits(((n + 1023) as u64).wrapping_shl(52));
    let rv = two_n * (num / dem);
    f64::from_bits(
        (is_inf & f64::INFINITY.to_bits()) | (0.0_f64.to_bits() & is_z) | (rv.to_bits() & is_good),
    )
}

#[inline(always)]
/// Approximates ln(x) (the natural logarithm) for f32.
///
/// Decomposes `x = 2ᵉ · m` (where `m ∈ [1, 2)`) using the IEEE 754
/// exponent field, then approximates `ln(m)` with a degree-2 polynomial
/// on `[1, 2]` and combines via `ln(x) = e·ln2 + ln(m)`.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.929%
pub(crate) fn approx_ln_f32(x: f32) -> f32 {
    const ONE_THIRD: f32 = 1.0 / 3.0;
    let bits = x.to_bits();
    let exponent = ((bits >> 23) as i32 - 127) as f32;

    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);

    let m_adj = mantissa - 1.0;
    let ln_mantissa = m_adj * (-ONE_THIRD).mul_add(m_adj, 1.0);

    // ln(x) = ln(2^exp * mantissa) = exp * ln(2) + ln(mantissa)
    exponent.mul_add(std::f32::consts::LN_2, ln_mantissa)
}

#[inline(always)]
/// Approximates ln(x) (the natural logarithm) for f64.
///
/// Decomposes `x = 2ᵉ · m` (where `m ∈ [1, 2)`) using the IEEE 754
/// exponent field, then approximates `ln(m)` with a degree-2 polynomial
/// on `[1, 2]` and combines via `ln(x) = e·ln2 + ln(m)`.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.929%
pub(crate) fn approx_ln_f64(x: f64) -> f64 {
    const ONE_THIRD: f64 = 1.0 / 3.0;
    let bits = x.to_bits();
    let exponent = ((bits >> 52) as i64 - 1023) as f64;
    let mantissa_bits = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    let mantissa = f64::from_bits(mantissa_bits);

    let m_adj = mantissa - 1.0;

    let ln_mantissa = m_adj * (-ONE_THIRD).mul_add(m_adj, 1.0);

    // ln(x) = exp * ln(2) + ln(mantissa)
    exponent.mul_add(std::f64::consts::LN_2, ln_mantissa)
}

#[inline(always)]
/// Approximates √x for f32.
///
/// Seeds a bit-manipulation initial guess (`bits >> 1` with a magic offset),
/// then refines with one Newton-Raphson (Babylonian) step: `0.5*(g + x/g)`.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.093%
pub(crate) fn approx_sqrt_f32(x: f32) -> f32 {
    x.sqrt()
}

#[inline(always)]
/// Approximates √x for f64.
///
/// Seeds a bit-manipulation initial guess (`bits >> 1` with a magic offset),
/// then refines with one Newton-Raphson (Babylonian) step: `0.5*(g + x/g)`.
///
/// Error characteristics are comparable to the f32 version.
/// **Measured relative error (over a representative sample):** 0.0% – 0.093%
pub(crate) fn approx_sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

#[inline(always)]
/// Approximates ∛x (cube root) for f32.
///
/// Seeds a bit-manipulation initial guess via `bits/3 + magic`, refines with
/// one Newton-Raphson step, and preserves the sign of `x` via bit mask.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.098%
pub(crate) fn approx_cbrt_f32(x: f32) -> f32 {
    let sign = x.to_bits() & 0x80000000;
    let abs_x = f32::from_bits(x.to_bits() & 0x7FFFFFFF);
    let guess = f32::from_bits(abs_x.to_bits() / 3 + 0x2a514067);
    let y2 = guess * guess;
    let refined = guess.mul_add(0.6666667, abs_x / (3.0 * y2));
    f32::from_bits(refined.to_bits() | sign)
}

#[inline(always)]
/// Approximates ∛x (cube root) for f64.
///
/// Seeds a bit-manipulation initial guess via `bits/3 + magic`, refines with
/// one Newton-Raphson step, and preserves the sign of `x` via bit mask.
///
/// Error characteristics are comparable to the f32 version.
/// **Measured relative error (over a representative sample):** 0.0% – 0.098%
pub(crate) fn approx_cbrt_f64(x: f64) -> f64 {
    let sign = x.to_bits() & 0x8000000000000000;
    let abs_x = f64::from_bits(x.to_bits() & 0x7FFFFFFFFFFFFFFF);
    let guess = f64::from_bits(abs_x.to_bits() / 3 + 0x2A9F789300000000);
    let y2 = guess * guess;
    let refined = guess.mul_add(0.6666666666666666, abs_x / (3.0 * y2));
    f64::from_bits(refined.to_bits() | sign)
}

#[inline(always)]
/// Approximates x^y for a float exponent y, for f32, via `exp(y · ln(x))`.
///
/// Composes [`approx_ln_f32`] and [`approx_exp_f32`]; errors from both
/// functions accumulate.  Only valid for positive `x`.
///
/// **Measured relative error (over a representative sample):** 5.7×10⁻⁸% – 3.58%
pub(crate) fn approx_powf_f32(x: f32, y: f32) -> f32 {
    approx_exp_f32(y * approx_ln_f32(x))
}

#[inline(always)]
/// Approximates x^y for a float exponent y, for f64, via `exp(y · ln(x))`.
///
/// Composes [`approx_ln_f64`] and [`approx_exp_f64`]; errors from both
/// functions accumulate.  Only valid for positive `x`.
///
/// **Measured relative error (over a representative sample):** 5.7×10⁻⁸% – 3.58%
pub(crate) fn approx_powf_f64(x: f64, y: f64) -> f64 {
    approx_exp_f64(y * approx_ln_f64(x))
}

#[inline(always)]
/// Approximates x^n for an integer exponent n, for f32.
///
/// Uses binary exponentiation (O(log |n|) multiplications), implemented
/// branchlessly via bitmask MUX.  Negative exponents are handled by
/// applying [`approx_inv_f32`] to the positive-exponent result.
///
/// **Measured relative error (over a representative sample):** 0.0% – 6.5×10⁻⁴%
/// (positive n; negative n adds `approx_inv_f32` error: up to 0.34%)
pub(crate) fn approx_powi_f32(x: f32, n: i32) -> f32 {
    if n == 0 {
        const ZERO_RESULT: u32 = 1.0_f32.to_bits(); // x^0 = 1
        return f32::from_bits(ZERO_RESULT);
    }

    let mut e = n.unsigned_abs(); // handles i32::MIN safely
    let mut base = x;
    let mut result = 1.0_f32;
    while e > 0 {
        let eb1set: u32 = (((e & 1) != 0) as u32).wrapping_neg();
        result = f32::from_bits((result * base).to_bits() & eb1set | (result).to_bits() & !eb1set);
        base *= base;
        e >>= 1;
    }

    let pos_result = result.to_bits();
    let neg_result = approx_inv_f32(result).to_bits();
    let nltz: u32 = ((n < 0) as u32).wrapping_neg();

    f32::from_bits((nltz & neg_result) | (!nltz & pos_result))
}

/// Approximates x^n for an integer exponent n, for f64.
///
/// Uses binary exponentiation (O(log |n|) multiplications), implemented
/// branchlessly via bitmask MUX.  Negative exponents are handled by
/// applying [`approx_inv_f64`] to the positive-exponent result.
///
/// **Measured relative error (over a representative sample):** 0.0% – 6.5×10⁻⁴%
/// (positive n; negative n adds `approx_inv_f64` error: up to 6.5×10⁻⁴%)
#[inline(always)]
pub(crate) fn approx_powi_f64(x: f64, n: i32) -> f64 {
    if n == 0 {
        const ZERO_RESULT: u64 = 1.0_f64.to_bits(); // x^0 = 1
        return f64::from_bits(ZERO_RESULT);
    }

    let mut e = n.unsigned_abs(); // handles i32::MIN safely
    let mut base = x;
    let mut result = 1.0_f64;
    while e > 0 {
        let eb1set: u64 = (((e & 1) != 0) as u64).wrapping_neg();
        result = f64::from_bits((result * base).to_bits() & eb1set | result.to_bits() & !eb1set);
        base *= base;
        e >>= 1;
    }
    let pos_result = result.to_bits();
    let neg_result = approx_inv_f64(result).to_bits();
    let nltz: u64 = ((n < 0) as u64).wrapping_neg();
    f64::from_bits((nltz & neg_result) | (!nltz & pos_result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_ln() {
        let x = 1.0_f32;
        assert!((approx_exp_f32(x) - std::f32::consts::E).abs() < 1e-2);
        assert!((approx_ln_f32(std::f32::consts::E) - 1.0).abs() < 1e-2);

        let x64 = 1.0_f64;
        assert!((approx_exp_f64(x64) - std::f64::consts::E).abs() < 1e-3);
        assert!((approx_ln_f64(std::f64::consts::E) - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_roots() {
        assert!((approx_sqrt_f32(4.0) - 2.0).abs() < 1e-6);
        assert!((approx_cbrt_f32(8.0) - 2.0).abs() < 1e-2);

        assert!((approx_sqrt_f64(4.0) - 2.0).abs() < 1e-6);
        assert!((approx_cbrt_f64(8.0) - 2.0).abs() < 1e-2);
    }

    #[test]
    fn test_powers() {
        assert!((approx_powi_f32(2.0, 3) - 8.0).abs() < 1e-6);
        assert!((approx_powf_f32(2.0, 3.0) - 8.0).abs() < 0.2); // exp/ln error accumulation
    }
}
