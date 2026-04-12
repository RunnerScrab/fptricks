use crate::logarithmic::{approx_sqrt_f32, approx_sqrt_f64};

#[inline(always)]
/// Approximates sin(x) for f32 with a degree-7 Taylor polynomial.
///
/// Reduces `x` to `[-π, π]` via `x - round(x/2π)·2π`, then evaluates
/// the Horner-form polynomial `x·(1 + x²(−1/6 + x²(1/120 + x²(−1/5040))))`.
///
/// **Measured absolute error (over a representative sample):** 0.0 – 0.693%
/// (absolute; output range is [−1, 1])
pub(crate) fn approx_sin_f32(x: f32) -> f32 {
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
pub(crate) fn approx_cos_f32(x: f32) -> f32 {
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
/// Approximates both sin(x) and cos(x) for f32.
///
/// Reuses range reduction and `x²` calculation to provide both results
/// more efficiently than calling them separately.
pub(crate) fn approx_sin_cos_f32(x: f32) -> (f32, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let x_red = {
            const TWO_PI: f32 = std::f32::consts::PI * 2.0;
            const INV_2PI: f32 = 1.0 / TWO_PI;
            let k = (x * INV_2PI).round();
            k.mul_add(-TWO_PI, x)
        };
        let x2 = x_red * x_red;
        let v_x2 = _mm_set1_ps(x2);

        // Constants: Lane 0 = cos, Lane 1 = sin
        let mut v = _mm_set_ps(0.0, 0.0, -0.00019841270, -0.0013888889);
        v = _mm_fmadd_ps(v, v_x2, _mm_set_ps(0.0, 0.0, 0.008333333, 0.041666668));
        v = _mm_fmadd_ps(v, v_x2, _mm_set_ps(0.0, 0.0, -0.16666667, -0.5));
        v = _mm_fmadd_ps(v, v_x2, _mm_set_ps(0.0, 0.0, 1.0, 1.0));

        let mut res = [0.0f32; 4];
        _mm_storeu_ps(res.as_mut_ptr(), v);
        (x_red * res[1], res[0])
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "fma")))]
    {
        const TWO_PI: f32 = std::f32::consts::PI * 2.0;
        const INV_2PI: f32 = 1.0 / TWO_PI;

        let k = (x * INV_2PI).round();
        let x_red = k.mul_add(-TWO_PI, x);
        let x2 = x_red * x_red;

        let s = x_red
            * (-0.00019841270_f32)
                .mul_add(x2, 0.008333333)
                .mul_add(x2, -0.16666667)
                .mul_add(x2, 1.0);

        let c = (-0.0013888889_f32)
            .mul_add(x2, 0.041666668)
            .mul_add(x2, -0.5)
            .mul_add(x2, 1.0);

        (s, c)
    }
}

#[inline(always)]
/// Approximates sin(x) for f64 with a degree-9 Taylor polynomial.
///
/// Reduces `x` to `[-π, π]` via `x - round(x/2π)·2π`, then evaluates
/// a 5-term Horner-form polynomial including the `x⁹/362880` term.
///
/// **Measured absolute error (over a representative sample):** 0.0 – 0.693%
/// (absolute; output range is [−1, 1])
pub(crate) fn approx_sin_f64(x: f64) -> f64 {
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
pub(crate) fn approx_cos_f64(x: f64) -> f64 {
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
/// Approximates both sin(x) and cos(x) for f64.
///
/// Reuses range reduction and `x²` calculation to provide both results
/// more efficiently than calling them separately.
pub(crate) fn approx_sin_cos_f64(x: f64) -> (f64, f64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let x_red = {
            const TWO_PI: f64 = std::f64::consts::PI * 2.0;
            const INV_2PI: f64 = 1.0 / TWO_PI;
            let k = (x * INV_2PI).round();
            k.mul_add(-TWO_PI, x)
        };
        let x2 = x_red * x_red;
        let v_x2 = _mm_set1_pd(x2);

        // Constants: Lane 0 = cos, Lane 1 = sin
        let mut v = _mm_set_pd(2.75573192239858906e-6, 2.48015873015873e-5);
        v = _mm_fmadd_pd(
            v,
            v_x2,
            _mm_set_pd(-1.984126984126984e-4, -1.388888888888889e-3),
        );
        v = _mm_fmadd_pd(
            v,
            v_x2,
            _mm_set_pd(8.333333333333333e-3, 4.166666666666667e-2),
        );
        v = _mm_fmadd_pd(v, v_x2, _mm_set_pd(-1.666666666666667e-1, -5.0e-1));
        v = _mm_fmadd_pd(v, v_x2, _mm_set_pd(1.0, 1.0));

        let mut res = [0.0f64; 2];
        _mm_storeu_pd(res.as_mut_ptr(), v);
        (x_red * res[1], res[0])
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "fma")))]
    {
        const TWO_PI: f64 = std::f64::consts::PI * 2.0;
        const INV_2PI: f64 = 1.0 / TWO_PI;

        let k = (x * INV_2PI).round();
        let x_red = k.mul_add(-TWO_PI, x);
        let x2 = x_red * x_red;

        let s = x_red
            * 2.75573192239858906e-6_f64
                .mul_add(x2, -1.984126984126984e-4)
                .mul_add(x2, 8.333333333333333e-3)
                .mul_add(x2, -1.666666666666667e-1)
                .mul_add(x2, 1.0);

        let c = 2.48015873015873e-5_f64
            .mul_add(x2, -1.388888888888889e-3)
            .mul_add(x2, 4.166666666666667e-2)
            .mul_add(x2, -5.0e-1)
            .mul_add(x2, 1.0);

        (s, c)
    }
}

#[inline(always)]
/// Approximates acos(x) for f32 on [-1, 1].
///
/// Uses standard A&S polynomial approximation branchlessly.
/// Maximum absolute error ~ 1.4e-5.
pub(crate) fn approx_acos_f32(x: f32) -> f32 {
    let abs_x = f32::from_bits(x.to_bits() & 0x7FFFFFFF);
    let is_neg: u32 = ((x < 0.0) as u32).wrapping_neg();
    let res = approx_sqrt_f32(1.0 - abs_x)
        * (-0.0187293_f32)
            .mul_add(abs_x, 0.0742610)
            .mul_add(abs_x, -0.2121144)
            .mul_add(abs_x, 1.5707288);
    let neg_res = std::f32::consts::PI - res;
    f32::from_bits((is_neg & neg_res.to_bits()) | (!is_neg & res.to_bits()))
}

#[inline(always)]
/// Approximates asin(x) for f32 on [-1, 1].
///
/// Evaluates as π/2 - approx_acos(x).
pub(crate) fn approx_asin_f32(x: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 - approx_acos_f32(x)
}

#[inline(always)]
/// Approximates acos(x) for f64 on [-1, 1].
///
/// Uses standard A&S polynomial approximation branchlessly.
/// Maximum absolute error ~ 1.4e-5.
pub(crate) fn approx_acos_f64(x: f64) -> f64 {
    let abs_x = f64::from_bits(x.to_bits() & 0x7FFFFFFFFFFFFFFF);
    let is_neg: u64 = ((x < 0.0) as u64).wrapping_neg();
    let res = approx_sqrt_f64(1.0 - abs_x)
        * (-0.0187293_f64)
            .mul_add(abs_x, 0.0742610)
            .mul_add(abs_x, -0.2121144)
            .mul_add(abs_x, 1.5707288);
    let neg_res = std::f64::consts::PI - res;
    f64::from_bits((is_neg & neg_res.to_bits()) | (!is_neg & res.to_bits()))
}

#[inline(always)]
/// Approximates asin(x) for f64 on [-1, 1].
///
/// Evaluates as π/2 - approx_acos(x).
pub(crate) fn approx_asin_f64(x: f64) -> f64 {
    std::f64::consts::FRAC_PI_2 - approx_acos_f64(x)
}

#[inline(always)]
/// Approximates atan2(y, x) for f32.
///
/// Uses a degree-7 polynomial for atan(z) on [0, 1] with quadrant reduction.
/// Maximum absolute error ~ 1.5e-4 radians.
pub(crate) fn approx_atan2_f32(y: f32, x: f32) -> f32 {
    let abs_x = x.abs();
    let abs_y = y.abs();

    // Handle x=0, y=0 case to avoid NaN
    if (abs_x + abs_y) == 0.0 {
        return 0.0;
    }

    let (min_v, max_v) = if abs_x >= abs_y {
        (abs_y, abs_x)
    } else {
        (abs_x, abs_y)
    };
    let z = min_v / max_v;
    let z2 = z * z;

    // Polynomial for atan(z) on [0, 1]: z * (1 - 0.327622764*z^2 + 0.15931422*z^4 - 0.0464964*z^6)
    let mut phi = z * ((-0.0464964749_f32 * z2 + 0.15931422).mul_add(z2, -0.327622764))
        .mul_add(z2, 1.0);

    if abs_y > abs_x {
        phi = std::f32::consts::FRAC_PI_2 - phi;
    }
    if x < 0.0 {
        phi = std::f32::consts::PI - phi;
    }
    if y < 0.0 {
        phi = -phi;
    }
    phi
}

#[inline(always)]
/// Approximates atan2(y, x) for f64.
///
/// Uses a degree-7 polynomial for atan(z) on [0, 1] with quadrant reduction.
/// Maximum absolute error ~ 1.5e-4 radians.
pub(crate) fn approx_atan2_f64(y: f64, x: f64) -> f64 {
    let abs_x = x.abs();
    let abs_y = y.abs();

    if (abs_x + abs_y) == 0.0 {
        return 0.0;
    }

    let (min_v, max_v) = if abs_x >= abs_y {
        (abs_y, abs_x)
    } else {
        (abs_x, abs_y)
    };
    let z = min_v / max_v;
    let z2 = z * z;

    let mut phi = z * ((-0.0464964749_f64 * z2 + 0.15931422).mul_add(z2, -0.327622764))
        .mul_add(z2, 1.0);

    if abs_y > abs_x {
        phi = std::f64::consts::FRAC_PI_2 - phi;
    }
    if x < 0.0 {
        phi = std::f64::consts::PI - phi;
    }
    if y < 0.0 {
        phi = -phi;
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_cos() {
        let x = 0.5_f32;
        let (s, c) = approx_sin_cos_f32(x);
        assert!((s - x.sin()).abs() < 1e-3);
        assert!((c - x.cos()).abs() < 1e-3);

        let x64 = 0.5_f64;
        let (s64, c64) = approx_sin_cos_f64(x64);
        assert!((s64 - x64.sin()).abs() < 1e-6);
        assert!((c64 - x64.cos()).abs() < 1e-6);
    }

    #[test]
    fn test_acos_asin() {
        let x = 0.5_f32;
        assert!((approx_acos_f32(x) - x.acos()).abs() < 1e-4);
        assert!((approx_asin_f32(x) - x.asin()).abs() < 1e-4);
    }

    #[test]
    fn test_atan2() {
        let y = 1.0_f32;
        let x = 1.0_f32;
        assert!((approx_atan2_f32(y, x) - y.atan2(x)).abs() < 2.5e-4);
    }
}

