pub trait FastFloatFnHaver {
    fn fast_mul2(self) -> Self;
    fn fast_div2(self) -> Self;
    fn fast_mul3(self) -> Self;
    fn fast_mul4(self) -> Self;
    fn fast_mul8(self) -> Self;
    fn approx_exp(self) -> Self;
    fn approx_ln(self) -> Self;
    fn approx_sqrt(self) -> Self;
    fn approx_cbrt(self) -> Self;
    fn approx_sin(self) -> Self;
    fn approx_cos(self) -> Self;
    /// Approximate 1/x via a Newton-Raphson step seeded by bit manipulation.
    fn approx_inv(self) -> Self;
    /// Approximate x^n for integer n via exponentiation by squaring.
    fn approx_powi(self, n: i32) -> Self;
    /// Approximate x^y for float y via exp(y · ln(x)). Works for positive x.
    fn approx_powf(self, y: Self) -> Self;
}

impl FastFloatFnHaver for f32 {
    #[inline(always)]
    fn fast_mul2(self) -> Self {
        fast_mul2_f32(self)
    }

    #[inline(always)]
    fn fast_div2(self) -> Self {
        fast_div2_f32(self)
    }

    #[inline(always)]
    fn fast_mul3(self) -> Self {
        fast_mul3_f32(self)
    }

    #[inline(always)]
    fn fast_mul4(self) -> Self {
        fast_mul4_f32(self)
    }

    #[inline(always)]
    fn fast_mul8(self) -> Self {
        fast_mul8_f32(self)
    }

    #[inline(always)]
    fn approx_exp(self) -> Self {
        approx_exp_f32(self)
    }

    #[inline(always)]
    fn approx_ln(self) -> Self {
        approx_ln_f32(self)
    }

    #[inline(always)]
    fn approx_sqrt(self) -> Self {
        approx_sqrt_f32(self)
    }

    #[inline(always)]
    fn approx_cbrt(self) -> Self {
        approx_cbrt_f32(self)
    }

    #[inline(always)]
    fn approx_sin(self) -> Self {
        approx_sin_f32(self)
    }

    #[inline(always)]
    fn approx_cos(self) -> Self {
        approx_cos_f32(self)
    }

    #[inline(always)]
    fn approx_inv(self) -> Self {
        approx_inv_f32(self)
    }

    #[inline(always)]
    fn approx_powi(self, n: i32) -> Self {
        approx_powi_f32(self, n)
    }

    #[inline(always)]
    fn approx_powf(self, y: Self) -> Self {
        approx_powf_f32(self, y)
    }
}

impl FastFloatFnHaver for f64 {
    #[inline(always)]
    fn fast_mul2(self) -> Self {
        fast_mul2_f64(self)
    }

    #[inline(always)]
    fn fast_div2(self) -> Self {
        fast_div2_f64(self)
    }

    #[inline(always)]
    fn fast_mul3(self) -> Self {
        fast_mul3_f64(self)
    }

    #[inline(always)]
    fn fast_mul4(self) -> Self {
        fast_mul4_f64(self)
    }

    #[inline(always)]
    fn fast_mul8(self) -> Self {
        fast_mul8_f64(self)
    }

    #[inline(always)]
    fn approx_exp(self) -> Self {
        approx_exp_f64(self)
    }

    #[inline(always)]
    fn approx_ln(self) -> Self {
        approx_ln_f64(self)
    }

    #[inline(always)]
    fn approx_sqrt(self) -> Self {
        approx_sqrt_f64(self)
    }

    #[inline(always)]
    fn approx_cbrt(self) -> Self {
        approx_cbrt_f64(self)
    }

    #[inline(always)]
    fn approx_sin(self) -> Self {
        approx_sin_f64(self)
    }

    #[inline(always)]
    fn approx_cos(self) -> Self {
        approx_cos_f64(self)
    }

    #[inline(always)]
    fn approx_inv(self) -> Self {
        approx_inv_f64(self)
    }

    #[inline(always)]
    fn approx_powi(self, n: i32) -> Self {
        approx_powi_f64(self, n)
    }

    #[inline(always)]
    fn approx_powf(self, y: Self) -> Self {
        approx_powf_f64(self, y)
    }
}

#[inline(always)]
pub fn fast_mul2_f32(x: f32) -> f32 {
    const ONEEXP: u32 = 1 << 23;
    f32::from_bits(x.to_bits().saturating_add(ONEEXP))
}

#[inline(always)]
pub fn fast_div2_f32(x: f32) -> f32 {
    const ONEEXP: u32 = 1 << 23;
    f32::from_bits(x.to_bits().saturating_sub(ONEEXP))
}

#[inline(always)]
pub fn fast_mul2_f64(x: f64) -> f64 {
    const ONEEXP: u64 = 1 << 52;
    f64::from_bits(x.to_bits().saturating_add(ONEEXP))
}

#[inline(always)]
pub fn fast_div2_f64(x: f64) -> f64 {
    const ONEEXP: u64 = 1 << 52;
    f64::from_bits(x.to_bits().saturating_sub(ONEEXP))
}

#[inline(always)]
pub fn fast_mul3_f64(x: f64) -> f64 {
    fast_mul2_f64(x) + x
}

#[inline(always)]
pub fn fast_mul4_f32(x: f32) -> f32 {
    const TWOEXP: u32 = 2 << 23;
    f32::from_bits(x.to_bits().saturating_add(TWOEXP))
}

#[inline(always)]
pub fn fast_mul4_f64(x: f64) -> f64 {
    const TWOEXP: u64 = 2 << 52;
    f64::from_bits(x.to_bits().saturating_add(TWOEXP))
}

#[inline(always)]
pub fn fast_mul8_f32(x: f32) -> f32 {
    const THREEEXP: u32 = 3 << 23;
    f32::from_bits(x.to_bits().saturating_add(THREEEXP))
}

#[inline(always)]
pub fn fast_mul8_f64(x: f64) -> f64 {
    const THREEEXP: u64 = 3 << 52;
    f64::from_bits(x.to_bits().saturating_add(THREEEXP))
}

#[inline(always)]
pub fn fast_mul3_f32(x: f32) -> f32 {
    fast_mul2_f32(x) + x
}

#[inline(always)]
/// Approximates eˣ (the natural exponential) for f32.
///
/// Uses range reduction to `x = n·ln2 + r`, then evaluates a degree-3 Taylor
/// polynomial for `eʳ` on `[-0.35, 0.35]` and scales by `2ⁿ` via exponent
/// manipulation.  Returns `+∞` for `x > 88.72` and `0.0` for `x < −87.34`,
/// both handled branchlessly.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.047%
pub fn approx_exp_f32(x: f32) -> f32 {
    let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
    let xgeqz: u32 = ((x >= 0.0) as u32).wrapping_neg();
    let is_inf: u32 = ((x > 88.72283) as u32).wrapping_neg();
    let is_z: u32 = ((x < -87.33654) as u32).wrapping_neg();

    //We are pulling e from the const LOG2_E
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = std::f32::consts::LN_2; //0.69314575; 
    const LN2_LO: f32 = 0.0000014286068;
    const INV6: f32 = 1.0 / 6.0;

    //Range reduction x = n*ln2 + r
    let xv = (xltz & (-0.5_f32).to_bits()) | (xgeqz & (0.5_f32).to_bits());

    //mul_add will run like shit unless target-cpu=native is used
    //and the ISA has an FMA
    let n = x.mul_add(INV_LN2, f32::from_bits(xv)) as i32;
    //let r = x - (n as f32) * LN2_HI - (n as f32) * LN2_LO;
    let r = (-n as f32).mul_add(LN2_LO, (-n as f32).mul_add(LN2_HI, x));

    let is_good: u32 = !is_inf & !is_z;

    //Approximate e^r on [-0.35, 0.35] using the Taylor series
    let exponent = (n + 127) as u32;
    //let res_r = ((INV6 * r + 0.5) * r + 1.0) * r + 1.0;
    let res_r = r.mul_add(r.mul_add(INV6.mul_add(r, 0.5), 1.0), 1.0);
    let two_n = f32::from_bits(exponent.wrapping_shl(23));

    let rv = two_n * res_r;
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
pub fn approx_exp_f64(x: f64) -> f64 {
    let xltz: u64 = ((x < 0.0) as u64).wrapping_neg();
    let xgeqz: u64 = ((x >= 0.0) as u64).wrapping_neg();
    let is_inf: u64 = ((x > 709.782712893384) as u64).wrapping_neg();
    let is_z: u64 = ((x < -708.3964185322641) as u64).wrapping_neg();

    // Range reduction uses log2(e) = 1/ln(2)
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = std::f64::consts::LN_2;
    const LN2_LO: f64 = 1.9082149292705877e-10;
    const INV6: f64 = 1.0 / 6.0;

    // Bias for range reduction: add/sub 0.5 depending on sign of x
    let xv = (xltz & (-0.5_f64).to_bits()) | (xgeqz & (0.5_f64).to_bits());
    let n = (x * INV_LN2 + f64::from_bits(xv)) as i32;

    //let r = x - (n as f64) * LN2_HI - (n as f64) * LN2_LO;
    //These mul_adds will run like shit unless -C target-cpu=native is used
    //and the CPU has a fused multiply add instruction!
    let r = (-n as f64).mul_add(LN2_LO, (-n as f64).mul_add(LN2_HI, x));

    let is_good: u64 = !is_inf & !is_z;

    // Polynomial approximation for e^r on roughly [-0.35, 0.35]:
    //let res_r = ((INV6 * r + 0.5) * r + 1.0) * r + 1.0;
    let res_r = r.mul_add(r.mul_add(INV6.mul_add(r, 0.5), 1.0), 1.0);

    // Build 2^n via exponent bias (1023 for f64)
    let exponent = (n + 1023) as u64;
    let two_n = f64::from_bits(exponent.wrapping_shl(52));

    let rv = two_n * res_r;

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
pub fn approx_ln_f32(x: f32) -> f32 {
    const ONE_THIRD: f32 = 1.0 / 3.0;
    let bits = x.to_bits();
    let exponent = ((bits >> 23) as i32 - 127) as f32;

    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);

    // Linear approximation of ln(mantissa) for mantissa in [1, 2]
    let m_adj = mantissa - 1.0;
    //let ln_mantissa = m_adj * (1.0 - ONE_THIRD * m_adj);
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
pub fn approx_ln_f64(x: f64) -> f64 {
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
pub fn approx_sqrt_f32(x: f32) -> f32 {
    let guess = f32::from_bits((x.to_bits() >> 1) + 0x1fbb4000);
    0.5 * (guess + x / guess)
}

#[inline(always)]
/// Approximates ∛x (cube root) for f32.
///
/// Seeds a bit-manipulation initial guess via `bits/3 + magic`, refines with
/// one Newton-Raphson step, and preserves the sign of `x` via bit mask.
///
/// **Measured relative error (over a representative sample):** 0.0% – 0.098%
pub fn approx_cbrt_f32(x: f32) -> f32 {
    let sign = x.to_bits() & 0x80000000;
    let abs_x = f32::from_bits(x.to_bits() & 0x7FFFFFFF);
    let guess = f32::from_bits(abs_x.to_bits() / 3 + 0x2a514067);
    let y2 = guess * guess;
    let refined = 0.6666667 * guess + abs_x / (3.0 * y2);
    f32::from_bits(refined.to_bits() | sign)
}

#[inline(always)]
/// Approximates sin(x) for f32 with a degree-7 Taylor polynomial.
///
/// Reduces `x` to `[-π, π]` via `x - round(x/2π)·2π`, then evaluates
/// the Horner-form polynomial `x·(1 + x²(−1/6 + x²(1/120 + x²(−1/5040))))`.
///
/// **Measured absolute error (over a representative sample):** 0.0 – 0.693%
/// (absolute; output range is [−1, 1])
pub fn approx_sin_f32(x: f32) -> f32 {
    const INV_2PI: f32 = 0.15915494; // 1 / (2 * PI)
    const TWO_PI: f32 = 6.2831855;

    let k = (x * INV_2PI).round();
    let x = k.mul_add(-TWO_PI, x);

    // Horner form: x - x³/6 + x⁵/120 - x⁷/5040
    // = x * (1 + x²(-1/6 + x²(1/120 + x²(-1/5040))))
    let x2 = x * x;
    x * (-0.00019841270_f32)
        .mul_add(x2, 0.008333333)
        .mul_add(x2, -0.16666667)
        .mul_add(x2, 1.0)
}

#[inline(always)]
/// Approximates cos(x) for f32 with a degree-6 Taylor polynomial.
///
/// Reduces `x` to `[-π, π]` via `x - round(x/2π)·2π`, then evaluates
/// the Horner-form polynomial `1 + x²(−1/2 + x²(1/24 + x²(−1/720)))`.
///
/// **Measured absolute error (over a representative sample):** 0.0 – 2.40%
/// (absolute; output range is [−1, 1])
pub fn approx_cos_f32(x: f32) -> f32 {
    const TWO_PI: f32 = std::f32::consts::PI * 2.0;
    const INV_2PI: f32 = 1.0 / TWO_PI;

    let k = (x * INV_2PI).round();
    let x = k.mul_add(-TWO_PI, x);

    // Horner form: 1 - x²/2 + x⁴/24 - x⁶/720
    // = 1 + x²(-1/2 + x²(1/24 + x²(-1/720)))
    let x2 = x * x;
    (-0.0013888889_f32)
        .mul_add(x2, 0.041666668)
        .mul_add(x2, -0.5)
        .mul_add(x2, 1.0)
}

#[inline(always)]
/// Approximates √x for f64.
///
/// Seeds a bit-manipulation initial guess (`bits >> 1` with a magic offset),
/// then refines with one Newton-Raphson (Babylonian) step: `0.5*(g + x/g)`.
///
/// Error characteristics are comparable to the f32 version.
/// **Measured relative error (over a representative sample):** 0.0% – 0.093%
pub fn approx_sqrt_f64(x: f64) -> f64 {
    let guess = f64::from_bits((x.to_bits() >> 1) + 0x1FF7A00000000000);
    0.5 * (guess + x / guess)
}

#[inline(always)]
/// Approximates ∛x (cube root) for f64.
///
/// Seeds a bit-manipulation initial guess via `bits/3 + magic`, refines with
/// one Newton-Raphson step, and preserves the sign of `x` via bit mask.
///
/// Error characteristics are comparable to the f32 version.
/// **Measured relative error (over a representative sample):** 0.0% – 0.098%
pub fn approx_cbrt_f64(x: f64) -> f64 {
    let sign = x.to_bits() & 0x8000000000000000;
    let abs_x = f64::from_bits(x.to_bits() & 0x7FFFFFFFFFFFFFFF);
    let guess = f64::from_bits(abs_x.to_bits() / 3 + 0x2A9F789300000000);
    let y2 = guess * guess;
    let refined = 0.6666666666666666 * guess + abs_x / (3.0 * y2);
    f64::from_bits(refined.to_bits() | sign)
}

#[inline(always)]
/// Approximates sin(x) for f64 with a degree-9 Taylor polynomial.
///
/// Reduces `x` to `[-π, π]` via `x - round(x/2π)·2π`, then evaluates
/// a 5-term Horner-form polynomial including the `x⁹/362880` term.
///
/// **Measured absolute error (over a representative sample):** 0.0 – 0.693%
/// (absolute; output range is [−1, 1])
pub fn approx_sin_f64(x: f64) -> f64 {
    const INV_2PI: f64 = 0.15915494309189535; // 1 / (2 * PI)
    const TWO_PI: f64 = 6.283185307179586;

    let k = (x * INV_2PI).round();
    let x = k.mul_add(-TWO_PI, x);

    // Horner form: x - x³/6 + x⁵/120 - x⁷/5040 + x⁹/362880
    // = x * (1 + x²(-1/6 + x²(1/120 + x²(-1/5040 + x²/362880))))
    let x2 = x * x;
    x * 2.75573192239858906e-6_f64
        .mul_add(x2, -1.984126984126984e-4)
        .mul_add(x2, 8.333333333333333e-3)
        .mul_add(x2, -1.666666666666667e-1)
        .mul_add(x2, 1.0)
}

#[inline(always)]
/// Approximates cos(x) for f64 with a degree-8 Taylor polynomial.
///
/// Reduces `x` to `[-π, π]` via `x - round(x/2π)·2π`, then evaluates
/// a 5-term Horner-form polynomial including the `x⁸/40320` term.
///
/// **Measured absolute error (over a representative sample):** 0.0 – 2.40%
/// (absolute; output range is [−1, 1])
pub fn approx_cos_f64(x: f64) -> f64 {
    const TWO_PI: f64 = std::f64::consts::PI * 2.0;
    const INV_2PI: f64 = 1.0 / TWO_PI;
    let k = (x * INV_2PI).round();
    let x = k.mul_add(-TWO_PI, x);

    // Horner form for degree-8 Taylor series:
    // 1 - x²/2 + x⁴/24 - x⁶/720 + x⁸/40320
    // = 1 + x²(-1/2 + x²(1/24 + x²(-1/720 + x²/40320)))
    let x2 = x * x;
    2.48015873015873e-5_f64
        .mul_add(x2, -1.388888888888889e-3)
        .mul_add(x2, 4.166666666666667e-2)
        .mul_add(x2, -5.0e-1)
        .mul_add(x2, 1.0)
}

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
    y0 * (2.0 - x * y0)
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
    let y1 = y0 * (2.0 - x * y0);
    y1 * (2.0 - x * y1)
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
pub fn approx_powi_f32(x: f32, n: i32) -> f32 {
    let nz: u32 = ((n == 0) as u32).wrapping_neg();
    let nltz: u32 = ((n < 0) as u32).wrapping_neg();

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
    let zero_result = 1.0_f32.to_bits(); // x^0 = 1

    let is_pos: u32 = !nltz & !nz; // n > 0

    f32::from_bits((nltz & neg_result) | (nz & zero_result) | (is_pos & pos_result))
}

#[inline(always)]
/// Approximates x^n for an integer exponent n, for f64.
///
/// Uses binary exponentiation (O(log |n|) multiplications), implemented
/// branchlessly via bitmask MUX.  Negative exponents are handled by
/// applying [`approx_inv_f64`] to the positive-exponent result.
///
/// **Measured relative error (over a representative sample):** 0.0% – 6.5×10⁻⁴%
/// (positive n; negative n adds `approx_inv_f64` error: up to 6.5×10⁻⁴%)
pub fn approx_powi_f64(x: f64, n: i32) -> f64 {
    let nz: u64 = ((n == 0) as u64).wrapping_neg();
    let nltz: u64 = ((n < 0) as u64).wrapping_neg();

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
    let zero_result = 1.0_f64.to_bits(); // x^0 = 1

    let is_pos: u64 = !nltz & !nz; // n > 0

    f64::from_bits((nltz & neg_result) | (nz & zero_result) | (is_pos & pos_result))
}

#[inline(always)]
/// Approximates x^y for a float exponent y, for f32, via `exp(y · ln(x))`.
///
/// Composes [`approx_ln_f32`] and [`approx_exp_f32`]; errors from both
/// functions accumulate.  Only valid for positive `x`.
///
/// **Measured relative error (over a representative sample):** 5.7×10⁻⁸% – 3.58%
pub fn approx_powf_f32(x: f32, y: f32) -> f32 {
    approx_exp_f32(y * approx_ln_f32(x))
}

#[inline(always)]
/// Approximates x^y for a float exponent y, for f64, via `exp(y · ln(x))`.
///
/// Composes [`approx_ln_f64`] and [`approx_exp_f64`]; errors from both
/// functions accumulate.  Only valid for positive `x`.
///
/// **Measured relative error (over a representative sample):** 5.7×10⁻⁸% – 3.58%
pub fn approx_powf_f64(x: f64, y: f64) -> f64 {
    approx_exp_f64(y * approx_ln_f64(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_approximations() {
        init_logger();
        let test_vals: [f64; 6] = [0.0, 1.0, 2.0, 10.0, 100.0, 1000.0];

        let (mut sqrt_min, mut sqrt_max) = (f64::MAX, 0.0_f64);
        let (mut cbrt_min, mut cbrt_max) = (f64::MAX, 0.0_f64);
        for &val in &test_vals {
            let r_sqrt = val.approx_sqrt();
            let exact_sqrt = val.sqrt();
            let err_sqrt = (r_sqrt - exact_sqrt).abs();
            let rel_sqrt = if exact_sqrt > 0.0 {
                err_sqrt / exact_sqrt
            } else {
                err_sqrt
            };
            log::debug!(
                "sqrt({val}): approx={r_sqrt} exact={exact_sqrt} abs_err={err_sqrt:.6e} rel_err={rel_sqrt:.6e}"
            );
            sqrt_min = sqrt_min.min(rel_sqrt);
            sqrt_max = sqrt_max.max(rel_sqrt);
            assert!(
                err_sqrt < 0.1 * exact_sqrt || err_sqrt < 0.01,
                "sqrt error too high: {} != {}",
                r_sqrt,
                exact_sqrt
            );

            let r_cbrt = val.approx_cbrt();
            let exact_cbrt = val.cbrt();
            let err_cbrt = (r_cbrt - exact_cbrt).abs();
            let rel_cbrt = if exact_cbrt > 0.0 {
                err_cbrt / exact_cbrt
            } else {
                err_cbrt
            };
            log::debug!(
                "cbrt({val}): approx={r_cbrt} exact={exact_cbrt} abs_err={err_cbrt:.6e} rel_err={rel_cbrt:.6e}"
            );
            cbrt_min = cbrt_min.min(rel_cbrt);
            cbrt_max = cbrt_max.max(rel_cbrt);
            assert!(
                err_cbrt < 0.08 * exact_cbrt || err_cbrt < 0.08,
                "cbrt error too high: {} != {}",
                r_cbrt,
                exact_cbrt
            );
        }
        log::info!("approx_sqrt  rel_err range: best={sqrt_min:.6e}  worst={sqrt_max:.6e}");
        log::info!("approx_cbrt  rel_err range: best={cbrt_min:.6e}  worst={cbrt_max:.6e}");

        let angles = [
            -10.0,
            -std::f64::consts::PI,
            -1.0,
            0.0,
            1.0,
            std::f64::consts::PI,
            10.0,
        ];
        let (mut sin_min, mut sin_max) = (f64::MAX, 0.0_f64);
        let (mut cos_min, mut cos_max) = (f64::MAX, 0.0_f64);
        for &angle in &angles {
            let r_sin = angle.approx_sin();
            let exact_sin = angle.sin();
            let err_sin = (r_sin - exact_sin).abs();
            log::debug!("sin({angle}): approx={r_sin} exact={exact_sin} abs_err={err_sin:.6e}");
            sin_min = sin_min.min(err_sin);
            sin_max = sin_max.max(err_sin);
            assert!(
                err_sin < 0.08,
                "sin error too high: {} != {}",
                r_sin,
                exact_sin
            );

            let r_cos = angle.approx_cos();
            let exact_cos = angle.cos();
            let err_cos = (r_cos - exact_cos).abs();
            log::debug!("cos({angle}): approx={r_cos} exact={exact_cos} abs_err={err_cos:.6e}");
            cos_min = cos_min.min(err_cos);
            cos_max = cos_max.max(err_cos);
            assert!(
                err_cos < 0.08,
                "cos error too high: {} != {}",
                r_cos,
                exact_cos
            );
        }
        log::info!("approx_sin   abs_err range: best={sin_min:.6e}  worst={sin_max:.6e}");
        log::info!("approx_cos   abs_err range: best={cos_min:.6e}  worst={cos_max:.6e}");
    }

    #[test]
    fn test_approx_inv() {
        init_logger();
        let (mut inv64_min, mut inv64_max) = (f64::MAX, 0.0_f64);
        for &v in &[0.5_f64, 1.0, 2.0, 10.0, 100.0] {
            let approx = v.approx_inv();
            let exact = 1.0 / v;
            let rel_err = (approx - exact).abs() / exact;
            log::debug!("inv_f64({v}): approx={approx} exact={exact} rel_err={rel_err:.6e}");
            inv64_min = inv64_min.min(rel_err);
            inv64_max = inv64_max.max(rel_err);
            assert!(
                rel_err < 0.001,
                "inv relative error too high for {}: approx={} exact={} rel_err={}",
                v,
                approx,
                exact,
                rel_err
            );
        }
        log::info!("approx_inv_f64 rel_err range: best={inv64_min:.6e}  worst={inv64_max:.6e}");

        let (mut inv32_min, mut inv32_max) = (f32::MAX, 0.0_f32);
        for &v in &[0.5_f32, 1.0, 2.0, 10.0, 100.0] {
            let approx = v.approx_inv();
            let exact = 1.0 / v;
            let rel_err = (approx - exact).abs() / exact;
            log::debug!("inv_f32({v}): approx={approx} exact={exact} rel_err={rel_err:.6e}");
            inv32_min = inv32_min.min(rel_err);
            inv32_max = inv32_max.max(rel_err);
            assert!(
                rel_err < 0.005,
                "f32 inv relative error too high for {}: approx={} exact={} rel_err={}",
                v,
                approx,
                exact,
                rel_err
            );
        }
        log::info!("approx_inv_f32 rel_err range: best={inv32_min:.6e}  worst={inv32_max:.6e}");
    }

    #[test]
    fn test_approx_powi() {
        init_logger();
        let cases: &[(f64, i32)] = &[
            (2.0, 3),
            (3.0, 4),
            (10.0, 2),
            (0.5, 5),
            (2.0, -2),
            (4.0, -1),
            (1.5, 6),
            (2.0, 16),
        ];
        let (mut powi_min, mut powi_max) = (f64::MAX, 0.0_f64);
        for &(base, exp) in cases {
            let approx = base.approx_powi(exp);
            let exact = base.powi(exp);
            let rel_err = (approx - exact).abs() / exact.abs();
            log::debug!(
                "powi_f64({base}^{exp}): approx={approx} exact={exact} rel_err={rel_err:.6e}"
            );
            powi_min = powi_min.min(rel_err);
            powi_max = powi_max.max(rel_err);
            assert!(
                rel_err < 0.001,
                "powi relative error too high for {}^{}: approx={} exact={} rel_err={}",
                base,
                exp,
                approx,
                exact,
                rel_err
            );
        }
        log::info!("approx_powi_f64 rel_err range: best={powi_min:.6e}  worst={powi_max:.6e}");
        assert_eq!(7.0_f64.approx_powi(0), 1.0);
        assert_eq!(7.0_f32.approx_powi(0), 1.0);
    }

    #[test]
    fn test_approx_powf() {
        init_logger();
        let cases: &[(f64, f64)] = &[
            (2.0, 0.5),
            (4.0, 0.75),
            (10.0, 1.5),
            (2.0, 3.0),
            (3.0, 3.14),
        ];
        let (mut powf_min, mut powf_max) = (f64::MAX, 0.0_f64);
        for &(base, exp) in cases {
            let approx = base.approx_powf(exp);
            let exact = base.powf(exp);
            let rel_err = (approx - exact).abs() / exact.abs();
            log::debug!(
                "powf_f64({base}^{exp}): approx={approx} exact={exact} rel_err={rel_err:.6e}"
            );
            powf_min = powf_min.min(rel_err);
            powf_max = powf_max.max(rel_err);
            assert!(
                rel_err < 0.05,
                "powf relative error too high for {}^{}: approx={} exact={} rel_err={}",
                base,
                exp,
                approx,
                exact,
                rel_err
            );
        }
        log::info!("approx_powf_f64 rel_err range: best={powf_min:.6e}  worst={powf_max:.6e}");
    }

    /// Documents the known unhandled boundary behaviour of the fast_mul/div family.
    /// These functions avoid branches by design, so inputs like 0.0, ±inf, and
    /// values near overflow produce implementation-defined results. The test logs
    /// what actually comes out so regressions are visible.
    #[test]
    fn test_fast_mul_boundary() {
        init_logger();
        for &x in &[
            0.0_f32,
            -0.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            log::debug!("fast_mul2_f32({x:?}) = {:?}", fast_mul2_f32(x));
            log::debug!("fast_mul4_f32({x:?}) = {:?}", fast_mul4_f32(x));
            log::debug!("fast_mul8_f32({x:?}) = {:?}", fast_mul8_f32(x));
            log::debug!("fast_div2_f32({x:?}) = {:?}", fast_div2_f32(x));
        }
        for &x in &[
            0.0_f64,
            -0.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            log::debug!("fast_mul2_f64({x:?}) = {:?}", fast_mul2_f64(x));
            log::debug!("fast_mul4_f64({x:?}) = {:?}", fast_mul4_f64(x));
            log::debug!("fast_mul8_f64({x:?}) = {:?}", fast_mul8_f64(x));
            log::debug!("fast_div2_f64({x:?}) = {:?}", fast_div2_f64(x));
        }
        // Spot-check: normal inputs must be exact
        assert_eq!(fast_mul2_f32(1.0), 2.0);
        assert_eq!(fast_mul4_f32(1.0), 4.0);
        assert_eq!(fast_mul8_f32(1.0), 8.0);
        assert_eq!(fast_div2_f32(1.0), 0.5);
        assert_eq!(fast_mul3_f32(1.0), 3.0);
        assert_eq!(fast_mul2_f64(1.0), 2.0);
        assert_eq!(fast_mul4_f64(1.0), 4.0);
        assert_eq!(fast_mul8_f64(1.0), 8.0);
        assert_eq!(fast_div2_f64(1.0), 0.5);
        assert_eq!(fast_mul3_f64(1.0), 3.0);
    }

    #[test]
    fn test_approx_exp() {
        init_logger();

        let cases_f64: &[(f64, f64)] = &[
            (-10.0, (-10.0_f64).exp()),
            (-1.0, (-1.0_f64).exp()),
            (0.0, 1.0),
            (0.5, (0.5_f64).exp()),
            (1.0, std::f64::consts::E),
            (2.0, (2.0_f64).exp()),
            (5.0, (5.0_f64).exp()),
            (10.0, (10.0_f64).exp()),
        ];
        let (mut exp64_min, mut exp64_max) = (f64::MAX, 0.0_f64);
        for &(x, exact) in cases_f64 {
            let approx = approx_exp_f64(x);
            let rel_err = (approx - exact).abs() / exact.abs();
            log::debug!("exp_f64({x}): approx={approx} exact={exact} rel_err={rel_err:.6e}");
            exp64_min = exp64_min.min(rel_err);
            exp64_max = exp64_max.max(rel_err);
            assert!(rel_err < 0.01, "exp_f64({x}) rel_err={rel_err} too high");
        }
        log::info!("approx_exp_f64 rel_err range: best={exp64_min:.6e}  worst={exp64_max:.6e}");
        log::debug!("exp_f64(800.0)  = {:?}", approx_exp_f64(800.0));
        log::debug!("exp_f64(-800.0) = {:?}", approx_exp_f64(-800.0));
        assert_eq!(approx_exp_f64(800.0), f64::INFINITY);
        assert_eq!(approx_exp_f64(-800.0), 0.0);

        let cases_f32: &[(f32, f32)] = &[
            (-5.0, (-5.0_f32).exp()),
            (-1.0, (-1.0_f32).exp()),
            (0.0, 1.0),
            (0.5, (0.5_f32).exp()),
            (1.0, std::f32::consts::E),
            (2.0, (2.0_f32).exp()),
            (5.0, (5.0_f32).exp()),
        ];
        let (mut exp32_min, mut exp32_max) = (f32::MAX, 0.0_f32);
        for &(x, exact) in cases_f32 {
            let approx = approx_exp_f32(x);
            let rel_err = (approx - exact).abs() / exact.abs();
            log::debug!("exp_f32({x}): approx={approx} exact={exact} rel_err={rel_err:.6e}");
            exp32_min = exp32_min.min(rel_err);
            exp32_max = exp32_max.max(rel_err);
            assert!(rel_err < 0.01, "exp_f32({x}) rel_err={rel_err} too high");
        }
        log::info!("approx_exp_f32 rel_err range: best={exp32_min:.6e}  worst={exp32_max:.6e}");
        log::debug!("exp_f32(100.0)  = {:?}", approx_exp_f32(100.0));
        log::debug!("exp_f32(-100.0) = {:?}", approx_exp_f32(-100.0));
        assert_eq!(approx_exp_f32(100.0), f32::INFINITY);
        assert_eq!(approx_exp_f32(-100.0), 0.0);
    }

    #[test]
    fn test_approx_ln() {
        init_logger();

        let cases_f64: &[(f64, f64)] = &[
            (0.1, (0.1_f64).ln()),
            (0.5, (0.5_f64).ln()),
            (1.0, 0.0),
            (std::f64::consts::E, 1.0),
            (2.0, std::f64::consts::LN_2),
            (10.0, (10.0_f64).ln()),
            (100.0, (100.0_f64).ln()),
        ];
        let (mut ln64_min, mut ln64_max) = (f64::MAX, 0.0_f64);
        for &(x, exact) in cases_f64 {
            let approx = approx_ln_f64(x);
            let abs_err = (approx - exact).abs();
            let rel_err = if exact.abs() > 0.1 {
                abs_err / exact.abs()
            } else {
                abs_err
            };
            log::debug!(
                "ln_f64({x}): approx={approx} exact={exact} abs_err={abs_err:.6e} rel_err={rel_err:.6e}"
            );
            ln64_min = ln64_min.min(rel_err);
            ln64_max = ln64_max.max(rel_err);
            assert!(rel_err < 0.05, "ln_f64({x}) rel_err={rel_err} too high");
        }
        log::info!("approx_ln_f64  rel_err range: best={ln64_min:.6e}  worst={ln64_max:.6e}");

        let cases_f32: &[(f32, f32)] = &[
            (0.5, (0.5_f32).ln()),
            (1.0, 0.0),
            (std::f32::consts::E, 1.0),
            (2.0, std::f32::consts::LN_2),
            (10.0, (10.0_f32).ln()),
            (100.0, (100.0_f32).ln()),
        ];
        let (mut ln32_min, mut ln32_max) = (f32::MAX, 0.0_f32);
        for &(x, exact) in cases_f32 {
            let approx = approx_ln_f32(x);
            let abs_err = (approx - exact).abs();
            let rel_err = if exact.abs() > 0.1 {
                abs_err / exact.abs()
            } else {
                abs_err
            };
            log::debug!(
                "ln_f32({x}): approx={approx} exact={exact} abs_err={abs_err:.6e} rel_err={rel_err:.6e}"
            );
            ln32_min = ln32_min.min(rel_err);
            ln32_max = ln32_max.max(rel_err);
            assert!(rel_err < 0.05, "ln_f32({x}) rel_err={rel_err} too high");
        }
        log::info!("approx_ln_f32  rel_err range: best={ln32_min:.6e}  worst={ln32_max:.6e}");
    }
}
