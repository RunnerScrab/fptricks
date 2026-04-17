use crate::FastFloatFnHaver;

#[inline(always)]
pub fn batch_approx_sin_cos_f32(x: [f32; 8]) -> ([f32; 8], [f32; 8]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);

        const TWO_PI: f32 = core::f32::consts::PI * 2.0;
        const INV_2PI: f32 = 1.0 / TWO_PI;

        let k = _mm256_round_ps(
            _mm256_mul_ps(v_x, _mm256_set1_ps(INV_2PI)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let x_red = _mm256_fmadd_ps(k, _mm256_set1_ps(-TWO_PI), v_x);
        let x2 = _mm256_mul_ps(x_red, x_red);

        let s1 = _mm256_fmadd_ps(
            _mm256_set1_ps(-0.00019841270),
            x2,
            _mm256_set1_ps(0.008333333),
        );
        let s2 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(-0.16666667));
        let s3 = _mm256_fmadd_ps(s2, x2, _mm256_set1_ps(1.0));
        let s = _mm256_mul_ps(x_red, s3);

        let c1 = _mm256_fmadd_ps(
            _mm256_set1_ps(-0.0013888889),
            x2,
            _mm256_set1_ps(0.041666668),
        );
        let c2 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(-0.5));
        let c = _mm256_fmadd_ps(c2, x2, _mm256_set1_ps(1.0));

        core::mem::transmute((s, c))
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out_s = [0.0; 8];
        let mut out_c = [0.0; 8];
        for i in 0..8 {
            let (s, c) = x[i].approx_sin_cos();
            out_s[i] = s;
            out_c[i] = c;
        }
        (out_s, out_c)
    }
}

#[inline(always)]
pub fn batch_approx_sin_cos_f64(x: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);

        const TWO_PI: f64 = core::f64::consts::PI * 2.0;
        const INV_2PI: f64 = 1.0 / TWO_PI;

        let k = _mm256_round_pd(
            _mm256_mul_pd(v_x, _mm256_set1_pd(INV_2PI)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let x_red = _mm256_fmadd_pd(k, _mm256_set1_pd(-TWO_PI), v_x);
        let x2 = _mm256_mul_pd(x_red, x_red);

        let s_p1 = _mm256_fmadd_pd(
            _mm256_set1_pd(2.75573192239858906e-6),
            x2,
            _mm256_set1_pd(-1.984126984126984e-4),
        );
        let s_p2 = _mm256_fmadd_pd(s_p1, x2, _mm256_set1_pd(8.333333333333333e-3));
        let s_p3 = _mm256_fmadd_pd(s_p2, x2, _mm256_set1_pd(-1.666666666666667e-1));
        let s_p4 = _mm256_fmadd_pd(s_p3, x2, _mm256_set1_pd(1.0));
        let s = _mm256_mul_pd(x_red, s_p4);

        let c_p1 = _mm256_fmadd_pd(
            _mm256_set1_pd(2.48015873015873e-5),
            x2,
            _mm256_set1_pd(-1.388888888888889e-3),
        );
        let c_p2 = _mm256_fmadd_pd(c_p1, x2, _mm256_set1_pd(4.166666666666667e-2));
        let c_p3 = _mm256_fmadd_pd(c_p2, x2, _mm256_set1_pd(-5.0e-1));
        let c = _mm256_fmadd_pd(c_p3, x2, _mm256_set1_pd(1.0));

        core::mem::transmute((s, c))
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out_s = [0.0; 4];
        let mut out_c = [0.0; 4];
        for i in 0..4 {
            let (s, c) = x[i].approx_sin_cos();
            out_s[i] = s;
            out_c[i] = c;
        }
        (out_s, out_c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_approx_sin_cos_f32() {
        let input = [0.0, 1.0, -1.0, 3.14159, -3.14159, 10.0, -10.0, 5.0];
        let (batch_sin, batch_cos) = batch_approx_sin_cos_f32(input);
        for i in 0..8 {
            let (scalar_sin, scalar_cos) = crate::approx_sin_cos_f32(input[i]);
            assert_eq!(batch_sin[i].to_bits(), scalar_sin.to_bits());
            assert_eq!(batch_cos[i].to_bits(), scalar_cos.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_sin_cos_f64() {
        let input = [0.0, 1.0, 3.14159, -10.0];
        let (batch_sin, batch_cos) = batch_approx_sin_cos_f64(input);
        for i in 0..4 {
            let (scalar_sin, scalar_cos) = crate::approx_sin_cos_f64(input[i]);
            assert_eq!(batch_sin[i].to_bits(), scalar_sin.to_bits());
            assert_eq!(batch_cos[i].to_bits(), scalar_cos.to_bits());
        }
    }
}
