use crate::FastFloatFnHaver;

#[inline(always)]
pub fn batch_approx_ln_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v: __m256 = core::mem::transmute(x);

        let bits = _mm256_castps_si256(v);
        let shifted = _mm256_srli_epi32(bits, 23);
        let exp_int = _mm256_sub_epi32(shifted, _mm256_set1_epi32(127));
        let exp_f32 = _mm256_cvtepi32_ps(exp_int);

        let mask = _mm256_set1_epi32(0x007FFFFF);
        let c_3f80 = _mm256_set1_epi32(0x3F800000);
        let m_bits = _mm256_or_si256(_mm256_and_si256(bits, mask), c_3f80);
        let mantissa = _mm256_castsi256_ps(m_bits);

        let m_adj = _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0));

        let neg_one_third = _mm256_set1_ps(-1.0 / 3.0);
        let c_1 = _mm256_set1_ps(1.0);
        let inner = _mm256_fmadd_ps(neg_one_third, m_adj, c_1);
        let ln_mantissa = _mm256_mul_ps(m_adj, inner);

        let ln_2 = _mm256_set1_ps(core::f32::consts::LN_2);
        let res = _mm256_fmadd_ps(exp_f32, ln_2, ln_mantissa);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_ln();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_sqrt_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);
        let res = _mm256_sqrt_ps(v_x);
        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_sqrt();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_exp_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);

        let zero = _mm256_setzero_ps();
        let mask_ltz = _mm256_cmp_ps(v_x, zero, _CMP_LT_OQ);
        let is_inf = _mm256_cmp_ps(v_x, _mm256_set1_ps(88.72283), _CMP_GT_OQ);
        let is_z = _mm256_cmp_ps(v_x, _mm256_set1_ps(-87.33654), _CMP_LT_OQ);

        let xv = _mm256_blendv_ps(_mm256_set1_ps(0.5), _mm256_set1_ps(-0.5), mask_ltz);

        let n = _mm256_cvttps_epi32(_mm256_fmadd_ps(
            v_x,
            _mm256_set1_ps(core::f32::consts::LOG2_E),
            xv,
        ));
        let nf = _mm256_cvtepi32_ps(n);
        let neg_nf = _mm256_sub_ps(zero, nf);

        const LN2_HI: f32 = core::f32::consts::LN_2;
        const LN2_LO: f32 = 0.0000014286068;

        let inner = _mm256_fmadd_ps(neg_nf, _mm256_set1_ps(LN2_HI), v_x);
        let r = _mm256_fmadd_ps(neg_nf, _mm256_set1_ps(LN2_LO), inner);

        let exponent = _mm256_add_epi32(n, _mm256_set1_epi32(127));
        let two_n = _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23));

        let inv6 = _mm256_set1_ps(1.0 / 6.0);
        let c_05 = _mm256_set1_ps(0.5);
        let c_1 = _mm256_set1_ps(1.0);

        let p1 = _mm256_fmadd_ps(inv6, r, c_05);
        let p2 = _mm256_fmadd_ps(r, p1, c_1);
        let res_r = _mm256_fmadd_ps(r, p2, c_1);

        let rv = _mm256_mul_ps(two_n, res_r);

        let v_inf = _mm256_set1_ps(f32::INFINITY);

        let rv_masked = _mm256_blendv_ps(rv, zero, is_z);
        let res = _mm256_blendv_ps(rv_masked, v_inf, is_inf);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_exp();
        }
        out
    }
}

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
pub fn batch_approx_powf_f32(x: f32, y: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let lnx = x.approx_ln();
        let v_lnx = _mm256_set1_ps(lnx);
        let v_y: __m256 = core::mem::transmute(y);

        let y_lnx = _mm256_mul_ps(v_y, v_lnx);
        let y_lnx_arr: [f32; 8] = core::mem::transmute(y_lnx);

        batch_approx_exp_f32(y_lnx_arr)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        let lnx = x.approx_ln();
        for i in 0..8 {
            out[i] = (y[i] * lnx).approx_exp();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_ln_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v: __m256d = core::mem::transmute(x);

        let bits = _mm256_castpd_si256(v);
        let exp_shifted = _mm256_srli_epi64(bits, 52);
        let exp_int = _mm256_sub_epi64(exp_shifted, _mm256_set1_epi64x(1023));

        let lower128 = _mm_castsi128_ps(_mm256_castsi256_si128(exp_int));
        let upper128 = _mm_castsi128_ps(_mm256_extracti128_si256(exp_int, 1));
        let packed_ps = _mm_shuffle_ps(lower128, upper128, 136); // _MM_SHUFFLE(2, 0, 2, 0)
        let packed_epi32 = _mm_castps_si128(packed_ps);

        let exp_f64 = _mm256_cvtepi32_pd(packed_epi32);

        let mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
        let c_3ff0 = _mm256_set1_epi64x(0x3FF0000000000000i64);
        let m_bits = _mm256_or_si256(_mm256_and_si256(bits, mask), c_3ff0);
        let mantissa = _mm256_castsi256_pd(m_bits);

        let m_adj = _mm256_sub_pd(mantissa, _mm256_set1_pd(1.0));
        let neg_one_third = _mm256_set1_pd(-1.0 / 3.0);
        let c_1 = _mm256_set1_pd(1.0);
        let inner = _mm256_fmadd_pd(neg_one_third, m_adj, c_1);
        let ln_mantissa = _mm256_mul_pd(m_adj, inner);

        let ln_2 = _mm256_set1_pd(core::f64::consts::LN_2);
        let res = _mm256_fmadd_pd(exp_f64, ln_2, ln_mantissa);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_ln();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_sqrt_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let res = _mm256_sqrt_pd(v_x);
        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_sqrt();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_exp_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let zero = _mm256_setzero_pd();

        let mask_ltz = _mm256_cmp_pd(v_x, zero, _CMP_LT_OQ);
        let is_inf = _mm256_cmp_pd(v_x, _mm256_set1_pd(709.782712893384), _CMP_GT_OQ);
        let is_z = _mm256_cmp_pd(v_x, _mm256_set1_pd(-708.3964185322641), _CMP_LT_OQ);

        let xv = _mm256_blendv_pd(_mm256_set1_pd(0.5), _mm256_set1_pd(-0.5), mask_ltz);
        let n_pd = _mm256_fmadd_pd(v_x, _mm256_set1_pd(core::f64::consts::LOG2_E), xv);
        let n_epi32 = _mm256_cvttpd_epi32(n_pd);

        let nf = _mm256_cvtepi32_pd(n_epi32);
        let neg_nf = _mm256_sub_pd(zero, nf);

        const LN2_HI: f64 = core::f64::consts::LN_2;
        const LN2_LO: f64 = 1.9082149292705877e-10;

        let inner = _mm256_fmadd_pd(neg_nf, _mm256_set1_pd(LN2_HI), v_x);
        let r = _mm256_fmadd_pd(neg_nf, _mm256_set1_pd(LN2_LO), inner);

        let exp_int32 = _mm_add_epi32(n_epi32, _mm_set1_epi32(1023));
        let exp_int64 = _mm256_cvtepi32_epi64(exp_int32);
        let two_n = _mm256_castsi256_pd(_mm256_slli_epi64(exp_int64, 52));

        let inv6 = _mm256_set1_pd(1.0 / 6.0);
        let c_05 = _mm256_set1_pd(0.5);
        let c_1 = _mm256_set1_pd(1.0);

        let p1 = _mm256_fmadd_pd(inv6, r, c_05);
        let p2 = _mm256_fmadd_pd(r, p1, c_1);
        let res_r = _mm256_fmadd_pd(r, p2, c_1);

        let rv = _mm256_mul_pd(two_n, res_r);

        let v_inf = _mm256_set1_pd(core::f64::INFINITY);
        let rv_masked = _mm256_blendv_pd(rv, zero, is_z);
        let res = _mm256_blendv_pd(rv_masked, v_inf, is_inf);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_exp();
        }
        out
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

#[inline(always)]
pub fn batch_approx_powf_f64(x: f64, y: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let lnx = x.approx_ln();
        let v_lnx = _mm256_set1_pd(lnx);
        let v_y: __m256d = core::mem::transmute(y);

        let y_lnx = _mm256_mul_pd(v_y, v_lnx);
        let y_lnx_arr: [f64; 4] = core::mem::transmute(y_lnx);

        batch_approx_exp_f64(y_lnx_arr)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        let lnx = x.approx_ln();
        for i in 0..4 {
            out[i] = (y[i] * lnx).approx_exp();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_cbrt_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);
        let bits = _mm256_castps_si256(v_x);

        let sign_mask = _mm256_set1_epi32(0x80000000u32 as i32);
        let abs_mask = _mm256_set1_epi32(0x7FFFFFFFi32);

        let v_sign = _mm256_and_si256(bits, sign_mask);
        let v_abs_bits = _mm256_and_si256(bits, abs_mask);
        let v_abs_x = _mm256_castsi256_ps(v_abs_bits);

        // guess = bits / 3 + 0x2a514067 (vectorized version of scalar magic)
        // We use a store/load for the division by 3 as there is no single epi32 divide
        let mut abs_bits_arr = [0u32; 8];
        _mm256_storeu_si256(abs_bits_arr.as_mut_ptr() as *mut __m256i, v_abs_bits);
        for i in 0..8 {
            abs_bits_arr[i] = abs_bits_arr[i] / 3 + 0x2a514067;
        }
        let v_guess =
            _mm256_castsi256_ps(_mm256_loadu_si256(abs_bits_arr.as_ptr() as *const __m256i));

        // Newton-Raphson: refined = 0.6666667 * guess + abs_x / (3.0 * guess * guess)
        // We replace division with reciprocal to increase throughput
        let g2 = _mm256_mul_ps(v_guess, v_guess);
        let three_g2 = _mm256_mul_ps(_mm256_set1_ps(3.0), g2);

        // Fast reciprocal of 3 * guess^2
        let inv_3g2 = _mm256_rcp_ps(three_g2);
        // One NR step for reciprocal: y = y * (2 - x * y)
        let inv_3g2_refined = _mm256_mul_ps(
            inv_3g2,
            _mm256_fnmadd_ps(three_g2, inv_3g2, _mm256_set1_ps(2.0)),
        );

        let term2 = _mm256_mul_ps(v_abs_x, inv_3g2_refined);
        let refined = _mm256_fmadd_ps(_mm256_set1_ps(0.6666667), v_guess, term2);

        // Restore sign
        let res_bits = _mm256_or_si256(
            _mm256_and_si256(_mm256_castps_si256(refined), abs_mask),
            v_sign,
        );
        let res = _mm256_castsi256_ps(res_bits);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_cbrt();
        }
        out
    }
}

#[inline(always)]
pub fn chunk_approx_powf_cols_f32(x: [f32; 8], y: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_y: __m256 = core::mem::transmute(y);

        let lnx = batch_approx_ln_f32(x);
        let v_lnx: __m256 = core::mem::transmute(lnx);

        let y_lnx = _mm256_mul_ps(v_y, v_lnx);
        let y_lnx_arr: [f32; 8] = core::mem::transmute(y_lnx);

        batch_approx_exp_f32(y_lnx_arr)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_powf(y[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_inv_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);
        let bits = _mm256_castps_si256(v_x);
        let magic = _mm256_set1_epi32(0x7EF127EA_u32 as i32);
        let y0_bits = _mm256_sub_epi32(magic, bits);
        let v_y0 = _mm256_castsi256_ps(y0_bits);

        let res = _mm256_mul_ps(v_y0, _mm256_fnmadd_ps(v_x, v_y0, _mm256_set1_ps(2.0)));

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_inv();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_inv_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let bits = _mm256_castpd_si256(v_x);
        let magic = _mm256_set1_epi64x(0x7FDE623822835EEA_u64 as i64);
        let y0_bits = _mm256_sub_epi64(magic, bits);
        let v_y0 = _mm256_castsi256_pd(y0_bits);

        let two = _mm256_set1_pd(2.0);
        let v_y1 = _mm256_mul_pd(v_y0, _mm256_fnmadd_pd(v_x, v_y0, two));
        let res = _mm256_mul_pd(v_y1, _mm256_fnmadd_pd(v_x, v_y1, two));

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_inv();
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_powi_cols_f32(x: [f32; 8], n: [i32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let mut v_base = _mm256_loadu_ps(x.as_ptr());
        let v_n = _mm256_loadu_si256(n.as_ptr() as *const __m256i);
        let v_is_neg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), v_n);
        let v_is_zero = _mm256_cmpeq_epi32(_mm256_setzero_si256(), v_n);

        let mut v_e = _mm256_abs_epi32(v_n);
        let mut v_result = _mm256_set1_ps(1.0);

        for _ in 0..31 {
            if _mm256_testz_si256(v_e, v_e) == 1 {
                break;
            }
            let bit_set = _mm256_cmpeq_epi32(
                _mm256_and_si256(v_e, _mm256_set1_epi32(1)),
                _mm256_set1_epi32(1),
            );
            let v_mul = _mm256_mul_ps(v_result, v_base);
            v_result = _mm256_blendv_ps(v_result, v_mul, _mm256_castsi256_ps(bit_set));

            v_base = _mm256_mul_ps(v_base, v_base);
            v_e = _mm256_srli_epi32(v_e, 1);
        }

        // Return directly via transmute to avoid an extra stack copy if the compiler allows
        let mut out = [0.0; 8];
        let v_is_neg = _mm256_castsi256_ps(v_is_neg);
        let v_is_zero = _mm256_castsi256_ps(v_is_zero);

        // We still need the inverse for negative exponents
        _mm256_storeu_ps(out.as_mut_ptr(), v_result);
        let res_inv = batch_approx_inv_f32(out);
        let v_inv = _mm256_loadu_ps(res_inv.as_ptr());

        v_result = _mm256_blendv_ps(v_result, v_inv, v_is_neg);
        v_result = _mm256_blendv_ps(v_result, _mm256_set1_ps(1.0), v_is_zero);

        core::mem::transmute(v_result)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].approx_powi(n[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_fmadd_cols_f32(x: [f32; 8], m: [f32; 8], a: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_ps(x.as_ptr());
        let v_m = _mm256_loadu_ps(m.as_ptr());
        let v_a = _mm256_loadu_ps(a.as_ptr());
        let res = _mm256_fmadd_ps(v_x, v_m, v_a);
        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].mul_add(m[i], a[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_fmadd_cols_f64(x: [f64; 4], m: [f64; 4], a: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_pd(x.as_ptr());
        let v_m = _mm256_loadu_pd(m.as_ptr());
        let v_a = _mm256_loadu_pd(a.as_ptr());
        let res = _mm256_fmadd_pd(v_x, v_m, v_a);
        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].mul_add(m[i], a[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_cbrt_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let bits = _mm256_castpd_si256(v_x);

        let sign_mask = _mm256_set1_epi64x(0x8000000000000000u64 as i64);
        let abs_mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFi64);

        let v_sign = _mm256_and_si256(bits, sign_mask);
        let v_abs_bits = _mm256_and_si256(bits, abs_mask);

        let mut abs_bits_arr = [0u64; 4];
        _mm256_storeu_si256(abs_bits_arr.as_mut_ptr() as *mut __m256i, v_abs_bits);
        for i in 0..4 {
            abs_bits_arr[i] = abs_bits_arr[i] / 3 + 0x2A9F789300000000;
        }
        let v_guess =
            _mm256_castsi256_pd(_mm256_loadu_si256(abs_bits_arr.as_ptr() as *const __m256i));

        let v_abs_x = _mm256_castsi256_pd(v_abs_bits);
        let g2 = _mm256_mul_pd(v_guess, v_guess);
        let div = _mm256_div_pd(v_abs_x, _mm256_mul_pd(_mm256_set1_pd(3.0), g2));
        let refined = _mm256_fmadd_pd(_mm256_set1_pd(0.6666666666666666), v_guess, div);

        let res_bits = _mm256_or_si256(
            _mm256_and_si256(_mm256_castpd_si256(refined), abs_mask),
            v_sign,
        );
        let res = _mm256_castsi256_pd(res_bits);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_cbrt();
        }
        out
    }
}

#[inline(always)]
pub fn chunk_approx_powf_cols_f64(x: [f64; 4], y: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_y: __m256d = core::mem::transmute(y);

        let lnx = batch_approx_ln_f64(x);
        let v_lnx: __m256d = core::mem::transmute(lnx);

        let y_lnx = _mm256_mul_pd(v_y, v_lnx);
        let y_lnx_arr: [f64; 4] = core::mem::transmute(y_lnx);

        batch_approx_exp_f64(y_lnx_arr)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_powf(y[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_approx_powi_cols_f64(x: [f64; 4], n: [i32; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let mut v_base = _mm256_loadu_pd(x.as_ptr());
        let v_n = _mm_loadu_si128(n.as_ptr() as *const __m128i);
        let v_n_epi64 = _mm256_cvtepi32_epi64(v_n);

        let v_is_neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v_n_epi64);
        let v_is_zero = _mm256_cmpeq_epi64(_mm256_setzero_si256(), v_n_epi64);

        // abs(n) for epi64 is a bit more manual in AVX2
        // abs_x = (x ^ sign) - sign?
        let v_sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v_n_epi64);
        let v_e = _mm256_sub_epi64(_mm256_xor_si256(v_n_epi64, v_sign), v_sign);

        let mut e_current = v_e;
        let mut v_result = _mm256_set1_pd(1.0);

        for _ in 0..63 {
            if _mm256_testz_si256(e_current, e_current) == 1 {
                break;
            }
            let bit_set = _mm256_cmpeq_epi64(
                _mm256_and_si256(e_current, _mm256_set1_epi64x(1)),
                _mm256_set1_epi64x(1),
            );
            let v_mul = _mm256_mul_pd(v_result, v_base);
            v_result = _mm256_blendv_pd(v_result, v_mul, _mm256_castsi256_pd(bit_set));

            v_base = _mm256_mul_pd(v_base, v_base);
            e_current = _mm256_srli_epi64(e_current, 1);
        }

        let mut out = [0.0; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), v_result);
        let res_inv = batch_approx_inv_f64(out);
        let v_inv = _mm256_loadu_pd(res_inv.as_ptr());

        v_result = _mm256_blendv_pd(v_result, v_inv, _mm256_castsi256_pd(v_is_neg));
        v_result = _mm256_blendv_pd(
            v_result,
            _mm256_set1_pd(1.0),
            _mm256_castsi256_pd(v_is_zero),
        );

        core::mem::transmute(v_result)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].approx_powi(n[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_fmadd_f32(x: [f32; 8], m: f32, a: f32) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);
        let v_m = _mm256_set1_ps(m);
        let v_a = _mm256_set1_ps(a);
        let res = _mm256_fmadd_ps(v_x, v_m, v_a);
        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = x[i].mul_add(m, a);
        }
        out
    }
}

#[inline(always)]
pub fn batch_asymmetric_fma_f32(x: [f32; 8], mode: f32, sigma_lo: f32, sigma_hi: f32) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);
        let v_mode = _mm256_set1_ps(mode);
        let v_lo = _mm256_set1_ps(sigma_lo);
        let v_hi = _mm256_set1_ps(sigma_hi);

        let mask = _mm256_cmp_ps(v_x, _mm256_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_ps(v_hi, v_lo, mask);
        let res = _mm256_fmadd_ps(v_x, sigma, v_mode);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            let sigma = if x[i] < 0.0 { sigma_lo } else { sigma_hi };
            out[i] = x[i].mul_add(sigma, mode);
        }
        out
    }
}

#[inline(always)]
pub fn batch_asymmetric_fma_cols_f32(
    x: [f32; 8],
    mode: [f32; 8],
    sigma_lo: [f32; 8],
    sigma_hi: [f32; 8],
) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256 = core::mem::transmute(x);
        let v_mode: __m256 = core::mem::transmute(mode);
        let v_lo: __m256 = core::mem::transmute(sigma_lo);
        let v_hi: __m256 = core::mem::transmute(sigma_hi);

        let mask = _mm256_cmp_ps(v_x, _mm256_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_ps(v_hi, v_lo, mask);
        let res = _mm256_fmadd_ps(v_x, sigma, v_mode);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 8];
        for i in 0..8 {
            let sigma = if x[i] < 0.0 { sigma_lo[i] } else { sigma_hi[i] };
            out[i] = x[i].mul_add(sigma, mode[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_fmadd_f64(x: [f64; 4], m: f64, a: f64) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let v_m = _mm256_set1_pd(m);
        let v_a = _mm256_set1_pd(a);
        let res = _mm256_fmadd_pd(v_x, v_m, v_a);
        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].mul_add(m, a);
        }
        out
    }
}

#[inline(always)]
pub fn batch_asymmetric_fma_f64(x: [f64; 4], mode: f64, sigma_lo: f64, sigma_hi: f64) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let v_mode = _mm256_set1_pd(mode);
        let v_lo = _mm256_set1_pd(sigma_lo);
        let v_hi = _mm256_set1_pd(sigma_hi);

        let mask = _mm256_cmp_pd(v_x, _mm256_setzero_pd(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_pd(v_hi, v_lo, mask);
        let res = _mm256_fmadd_pd(v_x, sigma, v_mode);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let sigma = if x[i] < 0.0 { sigma_lo } else { sigma_hi };
            out[i] = x[i].mul_add(sigma, mode);
        }
        out
    }
}

#[inline(always)]
pub fn batch_asymmetric_fma_cols_f64(
    x: [f64; 4],
    mode: [f64; 4],
    sigma_lo: [f64; 4],
    sigma_hi: [f64; 4],
) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m256d = core::mem::transmute(x);
        let v_mode: __m256d = core::mem::transmute(mode);
        let v_lo: __m256d = core::mem::transmute(sigma_lo);
        let v_hi: __m256d = core::mem::transmute(sigma_hi);

        let mask = _mm256_cmp_pd(v_x, _mm256_setzero_pd(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_pd(v_hi, v_lo, mask);
        let res = _mm256_fmadd_pd(v_x, sigma, v_mode);

        core::mem::transmute(res)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let sigma = if x[i] < 0.0 { sigma_lo[i] } else { sigma_hi[i] };
            out[i] = x[i].mul_add(sigma, mode[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_sum_f32(data: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let chunks = data.chunks_exact(32);
        let rem = chunks.remainder();

        for chunk in chunks {
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(chunk.as_ptr()));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(chunk.as_ptr().add(8)));
            sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(chunk.as_ptr().add(16)));
            sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(chunk.as_ptr().add(24)));
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        let chunks8 = rem.chunks_exact(8);
        let rem_tail = chunks8.remainder();
        for chunk in chunks8 {
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(chunk.as_ptr()));
        }

        // Horizontal sum
        let x128 = _mm_add_ps(_mm256_extractf128_ps(sum0, 1), _mm256_castps256_ps128(sum0));
        let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        let mut total = _mm_cvtss_f32(x32);

        for &val in rem_tail {
            total += val;
        }
        total
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter().sum()
    }
}

#[inline(always)]
pub fn batch_sum_f64(data: &[f64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();

        let chunks = data.chunks_exact(16);
        let rem = chunks.remainder();

        for chunk in chunks {
            sum0 = _mm256_add_pd(sum0, _mm256_loadu_pd(chunk.as_ptr()));
            sum1 = _mm256_add_pd(sum1, _mm256_loadu_pd(chunk.as_ptr().add(4)));
            sum2 = _mm256_add_pd(sum2, _mm256_loadu_pd(chunk.as_ptr().add(8)));
            sum3 = _mm256_add_pd(sum3, _mm256_loadu_pd(chunk.as_ptr().add(12)));
        }

        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);

        let chunks4 = rem.chunks_exact(4);
        let rem_tail = chunks4.remainder();
        for chunk in chunks4 {
            sum0 = _mm256_add_pd(sum0, _mm256_loadu_pd(chunk.as_ptr()));
        }

        // Horizontal sum
        let x128 = _mm_add_pd(_mm256_extractf128_pd(sum0, 1), _mm256_castpd256_pd128(sum0));
        let x64 = _mm_add_pd(x128, _mm_permute_pd(x128, 1));
        let mut total = _mm_cvtsd_f64(x64);

        for &val in rem_tail {
            total += val;
        }
        total
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter().sum()
    }
}

#[inline(always)]
pub fn batch_sum_i32(data: &[i32]) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();

        let chunks = data.chunks_exact(32);
        let rem = chunks.remainder();

        for chunk in chunks {
            s0 = _mm256_add_epi32(s0, _mm256_loadu_si256(chunk.as_ptr() as *const __m256i));
            s1 = _mm256_add_epi32(
                s1,
                _mm256_loadu_si256(chunk.as_ptr().add(8) as *const __m256i),
            );
            s2 = _mm256_add_epi32(
                s2,
                _mm256_loadu_si256(chunk.as_ptr().add(16) as *const __m256i),
            );
            s3 = _mm256_add_epi32(
                s3,
                _mm256_loadu_si256(chunk.as_ptr().add(24) as *const __m256i),
            );
        }

        s0 = _mm256_add_epi32(s0, s1);
        s2 = _mm256_add_epi32(s2, s3);
        s0 = _mm256_add_epi32(s0, s2);

        let chunks8 = rem.chunks_exact(8);
        let rem_tail = chunks8.remainder();
        for chunk in chunks8 {
            s0 = _mm256_add_epi32(s0, _mm256_loadu_si256(chunk.as_ptr() as *const __m256i));
        }

        let x128 = _mm_add_epi32(_mm256_extractf128_si256(s0, 1), _mm256_castsi256_si128(s0));
        let x64 = _mm_add_epi32(x128, _mm_shuffle_epi32(x128, 0x0E));
        let x32 = _mm_add_epi32(x64, _mm_shuffle_epi32(x64, 0x01));
        let mut total = _mm_cvtsi128_si32(x32);

        for &val in rem_tail {
            total = total.wrapping_add(val);
        }
        total
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
    }
}

#[inline(always)]
pub fn batch_sum_u32(data: &[u32]) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();

        let chunks = data.chunks_exact(32);
        let rem = chunks.remainder();

        for chunk in chunks {
            s0 = _mm256_add_epi32(s0, _mm256_loadu_si256(chunk.as_ptr() as *const __m256i));
            s1 = _mm256_add_epi32(
                s1,
                _mm256_loadu_si256(chunk.as_ptr().add(8) as *const __m256i),
            );
            s2 = _mm256_add_epi32(
                s2,
                _mm256_loadu_si256(chunk.as_ptr().add(16) as *const __m256i),
            );
            s3 = _mm256_add_epi32(
                s3,
                _mm256_loadu_si256(chunk.as_ptr().add(24) as *const __m256i),
            );
        }

        s0 = _mm256_add_epi32(s0, s1);
        s2 = _mm256_add_epi32(s2, s3);
        s0 = _mm256_add_epi32(s0, s2);

        let chunks8 = rem.chunks_exact(8);
        let rem_tail = chunks8.remainder();
        for chunk in chunks8 {
            s0 = _mm256_add_epi32(s0, _mm256_loadu_si256(chunk.as_ptr() as *const __m256i));
        }

        let x128 = _mm_add_epi32(_mm256_extractf128_si256(s0, 1), _mm256_castsi256_si128(s0));
        let x64 = _mm_add_epi32(x128, _mm_shuffle_epi32(x128, 0x0E));
        let x32 = _mm_add_epi32(x64, _mm_shuffle_epi32(x64, 0x01));
        let mut total = _mm_cvtsi128_si32(x32) as u32;

        for &val in rem_tail {
            total = total.wrapping_add(val);
        }
        total
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter().fold(0u32, |acc, &x| acc.wrapping_add(x))
    }
}

#[inline(always)]
pub fn batch_sum_i64(data: &[i64]) -> i64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();

        let chunks = data.chunks_exact(16);
        let rem = chunks.remainder();

        for chunk in chunks {
            s0 = _mm256_add_epi64(s0, _mm256_loadu_si256(chunk.as_ptr() as *const __m256i));
            s1 = _mm256_add_epi64(
                s1,
                _mm256_loadu_si256(chunk.as_ptr().add(4) as *const __m256i),
            );
            s2 = _mm256_add_epi64(
                s2,
                _mm256_loadu_si256(chunk.as_ptr().add(8) as *const __m256i),
            );
            s3 = _mm256_add_epi64(
                s3,
                _mm256_loadu_si256(chunk.as_ptr().add(12) as *const __m256i),
            );
        }

        s0 = _mm256_add_epi64(s0, s1);
        s2 = _mm256_add_epi64(s2, s3);
        s0 = _mm256_add_epi64(s0, s2);

        let mut res = [0i64; 4];
        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, s0);
        let mut total = res[0] + res[1] + res[2] + res[3];
        for &val in rem {
            total += val;
        }
        total
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter().sum()
    }
}

#[inline(always)]
pub fn batch_sum_u64(data: &[u64]) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();

        let chunks = data.chunks_exact(16);
        let rem = chunks.remainder();

        for chunk in chunks {
            s0 = _mm256_add_epi64(s0, _mm256_loadu_si256(chunk.as_ptr() as *const __m256i));
            s1 = _mm256_add_epi64(
                s1,
                _mm256_loadu_si256(chunk.as_ptr().add(4) as *const __m256i),
            );
            s2 = _mm256_add_epi64(
                s2,
                _mm256_loadu_si256(chunk.as_ptr().add(8) as *const __m256i),
            );
            s3 = _mm256_add_epi64(
                s3,
                _mm256_loadu_si256(chunk.as_ptr().add(12) as *const __m256i),
            );
        }

        s0 = _mm256_add_epi64(s0, s1);
        s2 = _mm256_add_epi64(s2, s3);
        s0 = _mm256_add_epi64(s0, s2);

        let mut res = [0u64; 4];
        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, s0);
        let mut total = res[0] + res[1] + res[2] + res[3];
        for &val in rem {
            total += val;
        }
        total
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        data.iter().sum()
    }
}

#[inline(always)]
pub fn batch_mul_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N], out: &mut [f32; N]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 31 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            _mm256_storeu_ps(out_ptr.add(i), _mm256_mul_ps(vx0, vy0));

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            _mm256_storeu_ps(out_ptr.add(i + 8), _mm256_mul_ps(vx1, vy1));

            let vx2 = _mm256_loadu_ps(x_ptr.add(i + 16));
            let vy2 = _mm256_loadu_ps(y_ptr.add(i + 16));
            _mm256_storeu_ps(out_ptr.add(i + 16), _mm256_mul_ps(vx2, vy2));

            let vx3 = _mm256_loadu_ps(x_ptr.add(i + 24));
            let vy3 = _mm256_loadu_ps(y_ptr.add(i + 24));
            _mm256_storeu_ps(out_ptr.add(i + 24), _mm256_mul_ps(vx3, vy3));

            i += 32;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            _mm256_storeu_ps(out_ptr.add(i), _mm256_mul_ps(vx, vy));
            i += 8;
        }

        while i < len {
            out_ptr.add(i).write(*x_ptr.add(i) * *y_ptr.add(i));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] * y[i];
        }
    }
}

#[inline(always)]
pub fn batch_add_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N], out: &mut [f32; N]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 31 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            let vz0 = _mm256_add_ps(vx0, vy0);
            _mm256_storeu_ps(out_ptr.add(i), vz0);

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            let vz1 = _mm256_add_ps(vx1, vy1);
            _mm256_storeu_ps(out_ptr.add(i + 8), vz1);

            let vx2 = _mm256_loadu_ps(x_ptr.add(i + 16));
            let vy2 = _mm256_loadu_ps(y_ptr.add(i + 16));
            let vz2 = _mm256_add_ps(vx2, vy2);
            _mm256_storeu_ps(out_ptr.add(i + 16), vz2);

            let vx3 = _mm256_loadu_ps(x_ptr.add(i + 24));
            let vy3 = _mm256_loadu_ps(y_ptr.add(i + 24));
            let vz3 = _mm256_add_ps(vx3, vy3);
            _mm256_storeu_ps(out_ptr.add(i + 24), vz3);

            i += 32;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let vz = _mm256_add_ps(vx, vy);
            _mm256_storeu_ps(out_ptr.add(i), vz);
            i += 8;
        }

        while i < len {
            out_ptr.add(i).write(*x_ptr.add(i) + *y_ptr.add(i));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] + y[i];
        }
    }
    /*
    for i in 0..N {
        out[i] = x[i] + y[i];
    }
    */
}

#[inline(always)]
pub fn batch_fma_cols_f32<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    out: &mut [f32; N],
) {
    for i in 0..N {
        out[i] = x[i].mul_add(y[i], z[i]);
    }
}

#[inline(always)]
pub fn batch_mul_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N], out: &mut [f64; N]) {
    for i in 0..N {
        out[i] = x[i] * y[i];
    }
}
#[inline(always)]
pub fn batch_add_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N], out: &mut [f64; N]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 15 < len {
            let vx0 = _mm256_loadu_pd(x_ptr.add(i));
            let vy0 = _mm256_loadu_pd(y_ptr.add(i));
            _mm256_storeu_pd(out_ptr.add(i), _mm256_add_pd(vx0, vy0));

            let vx1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            _mm256_storeu_pd(out_ptr.add(i + 4), _mm256_add_pd(vx1, vy1));

            let vx2 = _mm256_loadu_pd(x_ptr.add(i + 8));
            let vy2 = _mm256_loadu_pd(y_ptr.add(i + 8));
            _mm256_storeu_pd(out_ptr.add(i + 8), _mm256_add_pd(vx2, vy2));

            let vx3 = _mm256_loadu_pd(x_ptr.add(i + 12));
            let vy3 = _mm256_loadu_pd(y_ptr.add(i + 12));
            _mm256_storeu_pd(out_ptr.add(i + 12), _mm256_add_pd(vx3, vy3));

            i += 16;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            _mm256_storeu_pd(out_ptr.add(i), _mm256_add_pd(vx, vy));
            i += 4;
        }

        while i < len {
            out_ptr.add(i).write(*x_ptr.add(i) + *y_ptr.add(i));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] + y[i];
        }
    }
}
#[inline(always)]
pub fn batch_fma_cols_f64<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    out: &mut [f64; N],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 15 < len {
            let vx0 = _mm256_loadu_pd(x_ptr.add(i));
            let vy0 = _mm256_loadu_pd(y_ptr.add(i));
            let vz0 = _mm256_loadu_pd(z_ptr.add(i));
            _mm256_storeu_pd(out_ptr.add(i), _mm256_fmadd_pd(vx0, vy0, vz0));

            let vx1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            let vz1 = _mm256_loadu_pd(z_ptr.add(i + 4));
            _mm256_storeu_pd(out_ptr.add(i + 4), _mm256_fmadd_pd(vx1, vy1, vz1));

            let vx2 = _mm256_loadu_pd(x_ptr.add(i + 8));
            let vy2 = _mm256_loadu_pd(y_ptr.add(i + 8));
            let vz2 = _mm256_loadu_pd(z_ptr.add(i + 8));
            _mm256_storeu_pd(out_ptr.add(i + 8), _mm256_fmadd_pd(vx2, vy2, vz2));

            let vx3 = _mm256_loadu_pd(x_ptr.add(i + 12));
            let vy3 = _mm256_loadu_pd(y_ptr.add(i + 12));
            let vz3 = _mm256_loadu_pd(z_ptr.add(i + 12));
            _mm256_storeu_pd(out_ptr.add(i + 12), _mm256_fmadd_pd(vx3, vy3, vz3));
            i += 16;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let vz = _mm256_loadu_pd(z_ptr.add(i));
            _mm256_storeu_pd(out_ptr.add(i), _mm256_fmadd_pd(vx, vy, vz));
            i += 4;
        }

        while i < len {
            out_ptr
                .add(i)
                .write((*x_ptr.add(i)).mul_add(*y_ptr.add(i), *z_ptr.add(i)));
            i += 1;
        }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        for i in 0..N {
            out[i] = x[i].mul_add(y[i], z[i]);
        }
    }
}

// Slice versions

#[inline(always)]
pub fn batch_mul_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    /*
    // ...
     */
    for i in 0..x.len() {
        out[i] = x[i] * y[i];
    }
}

#[inline(always)]
pub fn batch_add_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    /*
    // ...
     */
    for i in 0..x.len() {
        out[i] = x[i] + y[i];
    }
}

#[inline(always)]
pub fn batch_fma_vec_f32(x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == z.len() && z.len() == out.len());
    /*
    // ...
     */
    for i in 0..x.len() {
        out[i] = x[i].mul_add(y[i], z[i]);
    }
}

#[inline(always)]
pub fn batch_mul_vec_f64(x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    /*
    // ...
     */
    for i in 0..x.len() {
        out[i] = x[i] * y[i];
    }
}

#[inline(always)]
pub fn batch_add_vec_f64(x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    /*
    // ...
     */
    for i in 0..x.len() {
        out[i] = x[i] + y[i];
    }
}

#[inline(always)]
pub fn batch_fma_vec_f64(x: &[f64], y: &[f64], z: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == z.len() && z.len() == out.len());
    /*
    // ...
     */
    for i in 0..x.len() {
        out[i] = x[i].mul_add(y[i], z[i]);
    }
}

#[inline(always)]
pub fn batch_approx_powf_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N], out: &mut [f32; N]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr();
        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let mut o = [0.0; 8];
            let mut ax = [0.0; 8];
            let mut ay = [0.0; 8];
            _mm256_storeu_ps(ax.as_mut_ptr(), vx);
            _mm256_storeu_ps(ay.as_mut_ptr(), vy);
            o = chunk_approx_powf_cols_f32(ax, ay);
            _mm256_storeu_ps(out_ptr.add(i), _mm256_loadu_ps(o.as_ptr()));
            i += 8;
        }
        while i < len {
            out[i] = crate::approx_powf_f32(x[i], y[i]);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = crate::approx_powf_f32(x[i], y[i]);
        }
    }
}

#[inline(always)]
pub fn batch_approx_powf_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = x.len();
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr();
        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let mut o = [0.0; 8];
            let mut ax = [0.0; 8];
            let mut ay = [0.0; 8];
            _mm256_storeu_ps(ax.as_mut_ptr(), vx);
            _mm256_storeu_ps(ay.as_mut_ptr(), vy);
            o = chunk_approx_powf_cols_f32(ax, ay);
            _mm256_storeu_ps(out_ptr.add(i), _mm256_loadu_ps(o.as_ptr()));
            i += 8;
        }
        while i < len {
            out[i] = crate::approx_powf_f32(x[i], y[i]);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..x.len() {
            out[i] = crate::approx_powf_f32(x[i], y[i]);
        }
    }
}

#[inline(always)]
pub fn batch_powf_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N], out: &mut [f32; N]) {
    for i in 0..N {
        out[i] = x[i].powf(y[i]);
    }
}

#[inline(always)]
pub fn batch_powf_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i].powf(y[i]);
    }
}

#[inline(always)]
pub fn batch_approx_powf_cols_f64<const N: usize>(x: [f64; N], y: [f64; N]) -> [f64; N] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let mut out: [f64; N] = [0.0; N];
        let out_ptr = out.as_mut_ptr();
        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let mut o = [0.0; 4];
            let mut ax = [0.0; 4];
            let mut ay = [0.0; 4];
            _mm256_storeu_pd(ax.as_mut_ptr(), vx);
            _mm256_storeu_pd(ay.as_mut_ptr(), vy);
            o = chunk_approx_powf_cols_f64(ax, ay);
            _mm256_storeu_pd(out_ptr.add(i), _mm256_loadu_pd(o.as_ptr()));
            i += 4;
        }
        while i < len {
            out[i] = crate::approx_powf_f64(x[i], y[i]);
            i += 1;
        }
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = crate::approx_powf_f64(x[i], y[i]);
        }
    }
}

#[inline(always)]
pub fn batch_approx_powf_vec_f64(x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = x.len();
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr();
        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let mut o = [0.0; 4];
            let mut ax = [0.0; 4];
            let mut ay = [0.0; 4];
            _mm256_storeu_pd(ax.as_mut_ptr(), vx);
            _mm256_storeu_pd(ay.as_mut_ptr(), vy);
            o = chunk_approx_powf_cols_f64(ax, ay);
            _mm256_storeu_pd(out_ptr.add(i), _mm256_loadu_pd(o.as_ptr()));
            i += 4;
        }
        while i < len {
            out[i] = crate::approx_powf_f64(x[i], y[i]);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..x.len() {
            out[i] = crate::approx_powf_f64(x[i], y[i]);
        }
    }
}

#[inline(always)]
pub fn batch_powf_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N], out: &mut [f64; N]) {
    for i in 0..N {
        out[i] = x[i].powf(y[i]);
    }
}

#[inline(always)]
pub fn batch_powf_vec_f64(x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i].powf(y[i]);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::FastFloatFnHaver;

    #[test]
    fn test_batch_approx_ln_f32() {
        let input = [1.0, 1.5, 2.0, 5.0, 10.0, 100.0, 0.5, 0.1];
        let batch_res = batch_approx_ln_f32(input);
        for i in 0..8 {
            let scalar_res = crate::approx_ln_f32(input[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_sqrt_f32() {
        let input = [1.0, 2.0, 4.0, 10.0, 0.5, 100.0, 0.01, 1000.0];
        let batch_res = batch_approx_sqrt_f32(input);
        for i in 0..8 {
            let scalar_res = crate::approx_sqrt_f32(input[i]);
            assert!(
                (batch_res[i] - scalar_res).abs() < 1e-6,
                "Mismatch at {}: batch {}, scalar {}",
                input[i],
                batch_res[i],
                scalar_res
            );
        }
    }

    #[test]
    fn test_batch_approx_exp_f32() {
        let input = [0.0, 1.0, -1.0, 5.0, -5.0, 10.0, 50.0, -50.0];
        let batch_res = batch_approx_exp_f32(input);
        for i in 0..8 {
            let scalar_res = crate::approx_exp_f32(input[i]);
            assert_eq!(
                batch_res[i].to_bits(),
                scalar_res.to_bits(),
                "Mismatch at {}: batch: {} ({:x}), scalar: {} ({:x})",
                input[i],
                batch_res[i],
                batch_res[i].to_bits(),
                scalar_res,
                scalar_res.to_bits()
            );
        }
    }

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
    fn test_batch_approx_powf_f32() {
        let base = 2.0;
        let pwr = [0.0, 0.5, 1.0, 2.0, 3.5, 10.0, -1.0, -2.5];
        let batch_res = batch_approx_powf_f32(base, pwr);
        for i in 0..8 {
            let scalar_res = crate::approx_powf_f32(base, pwr[i]);
            assert_eq!(
                batch_res[i].to_bits(),
                scalar_res.to_bits(),
                "Mismatch at {} ^{}: batch {} ({:x}), scalar {} ({:x})",
                base,
                pwr[i],
                batch_res[i],
                batch_res[i].to_bits(),
                scalar_res,
                scalar_res.to_bits()
            );
        }
    }

    #[test]
    fn test_batch_approx_ln_f64() {
        let input = [1.0, 2.0, 10.0, 0.5];
        let batch_res = batch_approx_ln_f64(input);
        for i in 0..4 {
            let scalar_res = crate::approx_ln_f64(input[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_sqrt_f64() {
        let input = [1.0, 4.0, 10.0, 0.5];
        let batch_res = batch_approx_sqrt_f64(input);
        for i in 0..4 {
            let scalar_res = crate::approx_sqrt_f64(input[i]);
            assert!(
                (batch_res[i] - scalar_res).abs() < 1e-12,
                "Mismatch at {}: batch {}, scalar {}",
                input[i],
                batch_res[i],
                scalar_res
            );
        }
    }

    #[test]
    fn test_batch_approx_exp_f64() {
        let input = [0.0, 1.0, -5.0, 50.0];
        let batch_res = batch_approx_exp_f64(input);
        for i in 0..4 {
            let scalar_res = crate::approx_exp_f64(input[i]);
            assert_eq!(
                batch_res[i].to_bits(),
                scalar_res.to_bits(),
                "Mismatch at {}: batch {} ({:x}), scalar {} ({:x})",
                input[i],
                batch_res[i],
                batch_res[i].to_bits(),
                scalar_res,
                scalar_res.to_bits()
            );
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

    #[test]
    fn test_batch_approx_powf_f64() {
        let base = 2.0;
        let pwr = [0.0, 1.0, 3.5, -2.5];
        let batch_res = batch_approx_powf_f64(base, pwr);
        for i in 0..4 {
            let scalar_res = crate::approx_powf_f64(base, pwr[i]);
            assert_eq!(
                batch_res[i].to_bits(),
                scalar_res.to_bits(),
                "Mismatch at {} ^{}: batch {} ({:x}), scalar {} ({:x})",
                base,
                pwr[i],
                batch_res[i],
                batch_res[i].to_bits(),
                scalar_res,
                scalar_res.to_bits()
            );
        }
    }

    #[test]
    fn test_batch_fmadd_f32() {
        let input = [1.0, -2.0, 3.5, 0.0, -0.5, 10.0, 100.0, -100.0];
        let m = 2.5;
        let a = 1.25;
        let batch_res = batch_fmadd_f32(input, m, a);
        for i in 0..8 {
            let scalar_res = input[i].mul_add(m, a);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_asymmetric_fma_f32() {
        let input = [1.0, -2.0, 3.5, 0.0, -0.5, 10.0, 100.0, -100.0];
        let mode = 5.0;
        let sigma_lo = 0.5;
        let sigma_hi = 2.0;
        let batch_res = batch_asymmetric_fma_f32(input, mode, sigma_lo, sigma_hi);
        for i in 0..8 {
            let sigma = if input[i] < 0.0 { sigma_lo } else { sigma_hi };
            let scalar_res = input[i].mul_add(sigma, mode);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_fmadd_f64() {
        let input = [1.0, -2.0, 3.5, 0.0];
        let m = 2.5;
        let a = 1.25;
        let batch_res = batch_fmadd_f64(input, m, a);
        for i in 0..4 {
            let scalar_res = input[i].mul_add(m, a);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_asymmetric_fma_f64() {
        let input = [1.0, -2.0, 3.5, 0.0];
        let mode = 5.0;
        let sigma_lo = 0.5;
        let sigma_hi = 2.0;
        let batch_res = batch_asymmetric_fma_f64(input, mode, sigma_lo, sigma_hi);
        for i in 0..4 {
            let sigma = if input[i] < 0.0 { sigma_lo } else { sigma_hi };
            let scalar_res = input[i].mul_add(sigma, mode);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_fmadd_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0];
        let a = [10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let batch_res = batch_fmadd_cols_f32(x, m, a);
        for i in 0..8 {
            let scalar_res = x[i].mul_add(m[i], a[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_fmadd_cols_f64() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let m = [0.5, 0.5, 2.0, 2.0];
        let a = [10.0, 10.0, 0.0, 0.0];
        let batch_res = batch_fmadd_cols_f64(x, m, a);
        for i in 0..4 {
            let scalar_res = x[i].mul_add(m[i], a[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }
    #[test]
    fn test_batch_approx_cbrt_f32() {
        let input = [0.0, 1.0, -1.0, 8.0, -27.0, 1000.0, 0.5, -0.001];
        let batch_res = batch_approx_cbrt_f32(input);
        for i in 0..8 {
            let scalar_res = crate::approx_cbrt_f32(input[i]);
            assert!(
                (batch_res[i] - scalar_res).abs() < 1e-6,
                "Mismatch at {}: batch {}, scalar {}",
                input[i],
                batch_res[i],
                scalar_res
            );
        }
    }

    #[test]
    fn test_batch_approx_cbrt_f64() {
        let input = [0.0, 1.0, -8.0, 1000.0];
        let batch_res = batch_approx_cbrt_f64(input);
        for i in 0..4 {
            let scalar_res = crate::approx_cbrt_f64(input[i]);
            assert!(
                (batch_res[i] - scalar_res).abs() < 1e-12,
                "Mismatch at {}: batch {}, scalar {}",
                input[i],
                batch_res[i],
                scalar_res
            );
        }
    }

    #[test]
    fn test_batch_approx_powf_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [2.0, 0.5, 1.0, 0.0, -1.0, 2.5, 3.0, 0.1];
        let batch_res = {
            let mut out = [0.0; 8];
            chunk_approx_powf_cols_f32(x, y)
        };
        for i in 0..8 {
            let scalar_res = crate::approx_powf_f32(x[i], y[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_powf_cols_f64() {
        let x = [1.0, 2.0, 4.0, 10.0];
        let y = [2.0, 0.5, 1.0, -1.0];
        let batch_res = {
            let mut out = [0.0; 4];
            chunk_approx_powf_cols_f64(x, y)
        };
        for i in 0..4 {
            let scalar_res = crate::approx_powf_f64(x[i], y[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_powi_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 0.5, 10.0];
        let n = [0, 1, 2, 3, -1, -2, 2, 5];
        let batch_res = batch_approx_powi_cols_f32(x, n);
        for i in 0..8 {
            let scalar_res = crate::approx_powi_f32(x[i], n[i]);
            assert_eq!(
                batch_res[i].to_bits(),
                scalar_res.to_bits(),
                "Mismatch at {}^{}: batch {}, scalar {}",
                x[i],
                n[i],
                batch_res[i],
                scalar_res
            );
        }
    }

    #[test]
    fn test_batch_approx_powi_cols_f64() {
        let x = [1.0, 2.0, 3.0, 0.5];
        let n = [0, 2, -1, 3];
        let batch_res = batch_approx_powi_cols_f64(x, n);
        for i in 0..4 {
            let scalar_res = crate::approx_powi_f64(x[i], n[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_asymmetric_fma_cols_f32() {
        let x = [1.0, -2.0, 3.5, 0.0, -0.5, 10.0, 100.0, -100.0];
        let mode = [5.0, 5.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let sigma_lo = [0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4];
        let sigma_hi = [2.0, 2.0, 2.0, 2.0, 0.5, 0.6, 0.7, 0.8];
        let batch_res = batch_asymmetric_fma_cols_f32(x, mode, sigma_lo, sigma_hi);
        for i in 0..8 {
            let sigma = if x[i] < 0.0 { sigma_lo[i] } else { sigma_hi[i] };
            let scalar_res = x[i].mul_add(sigma, mode[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_asymmetric_fma_cols_f64() {
        let x = [1.0, -2.0, 3.5, 0.0];
        let mode = [5.0, 0.0, 1.0, 2.0];
        let sigma_lo = [0.5, 0.5, 0.1, 0.2];
        let sigma_hi = [2.0, 2.0, 0.5, 0.6];
        let batch_res = batch_asymmetric_fma_cols_f64(x, mode, sigma_lo, sigma_hi);
        for i in 0..4 {
            let sigma = if x[i] < 0.0 { sigma_lo[i] } else { sigma_hi[i] };
            let scalar_res = x[i].mul_add(sigma, mode[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_sum_f32() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let batch_total = batch_sum_f32(&input);
        let scalar_total: f32 = input.iter().sum();
        assert!((batch_total - scalar_total).abs() < 1e-4);
    }

    #[test]
    fn test_batch_sum_f64() {
        let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let batch_total = batch_sum_f64(&input);
        let scalar_total: f64 = input.iter().sum();
        assert!((batch_total - scalar_total).abs() < 1e-10);
    }

    #[test]
    fn test_batch_sum_i32() {
        let input: Vec<i32> = (0..100).map(|i| i as i32).collect();
        let batch_total = batch_sum_i32(&input);
        let scalar_total = input.iter().fold(0i32, |acc, &x| acc.wrapping_add(x));
        assert_eq!(batch_total, scalar_total);

        // Test wrapping behavior
        let input_wrap = vec![i32::MAX, 1];
        let total = batch_sum_i32(&input_wrap);
        assert_eq!(total, i32::MIN);
    }

    #[test]
    fn test_batch_sum_u32() {
        let input: Vec<u32> = (0..100).map(|i| i as u32).collect();
        let batch_total = batch_sum_u32(&input);
        let scalar_total = input.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
        assert_eq!(batch_total, scalar_total);

        // Test wrapping behavior
        let input_wrap = vec![u32::MAX, 1];
        let total = batch_sum_u32(&input_wrap);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_batch_sum_i64() {
        let input: Vec<i64> = (0..100).map(|i| i as i64).collect();
        let batch_total = batch_sum_i64(&input);
        let scalar_total: i64 = input.iter().sum();
        assert_eq!(batch_total, scalar_total);
    }

    #[test]
    fn test_batch_sum_u64() {
        let input: Vec<u64> = (0..100).map(|i| i as u64).collect();
        let batch_total = batch_sum_u64(&input);
        let scalar_total: u64 = input.iter().sum();
        assert_eq!(batch_total, scalar_total);
    }
}
