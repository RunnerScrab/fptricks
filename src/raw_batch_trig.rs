use core::arch::x86_64::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_sin_cos_f32(v_x: __m256) -> (__m256, __m256) {
    unsafe {
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

        (s, c)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_sin_cos_f32(v_x: __m128) -> (__m128, __m128) {
    unsafe {
        const TWO_PI: f32 = core::f32::consts::PI * 2.0;
        const INV_2PI: f32 = 1.0 / TWO_PI;

        let k = _mm_round_ps(
            _mm_mul_ps(v_x, _mm_set1_ps(INV_2PI)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let x_red = _mm_fmadd_ps(k, _mm_set1_ps(-TWO_PI), v_x);
        let x2 = _mm_mul_ps(x_red, x_red);

        let s1 = _mm_fmadd_ps(
            _mm_set1_ps(-0.00019841270),
            x2,
            _mm_set1_ps(0.008333333),
        );
        let s2 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(-0.16666667));
        let s3 = _mm_fmadd_ps(s2, x2, _mm_set1_ps(1.0));
        let s = _mm_mul_ps(x_red, s3);

        let c1 = _mm_fmadd_ps(
            _mm_set1_ps(-0.0013888889),
            x2,
            _mm_set1_ps(0.041666668),
        );
        let c2 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(-0.5));
        let c = _mm_fmadd_ps(c2, x2, _mm_set1_ps(1.0));

        (s, c)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_sin_cos_f64(v_x: __m256d) -> (__m256d, __m256d) {
    unsafe {
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

        (s, c)
    }
}
