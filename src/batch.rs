use crate::FastFloatFnHaver;

#[inline(always)]
pub fn batch_approx_ln_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm256_loadu_ps(x.as_ptr());
        
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
        
        let mut out = [0.0; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_ps(x.as_ptr());
        
        let bits = _mm256_castps_si256(v_x);
        let shifted = _mm256_srli_epi32(bits, 1);
        let added = _mm256_add_epi32(shifted, _mm256_set1_epi32(0x1fbb4000));
        let guess = _mm256_castsi256_ps(added);

        let div = _mm256_div_ps(v_x, guess);
        let sum = _mm256_add_ps(guess, div);
        let res = _mm256_mul_ps(_mm256_set1_ps(0.5), sum);
        
        let mut out = [0.0; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_ps(x.as_ptr());
        
        let zero = _mm256_setzero_ps();
        let mask_ltz = _mm256_cmp_ps(v_x, zero, _CMP_LT_OQ);
        let is_inf = _mm256_cmp_ps(v_x, _mm256_set1_ps(88.72283), _CMP_GT_OQ);
        let is_z = _mm256_cmp_ps(v_x, _mm256_set1_ps(-87.33654), _CMP_LT_OQ);

        let xv = _mm256_blendv_ps(_mm256_set1_ps(0.5), _mm256_set1_ps(-0.5), mask_ltz);

        let n = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_x, _mm256_set1_ps(core::f32::consts::LOG2_E), xv));
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

        let mut out = [0.0; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_ps(x.as_ptr());
        
        const TWO_PI: f32 = core::f32::consts::PI * 2.0;
        const INV_2PI: f32 = 1.0 / TWO_PI;

        let k = _mm256_round_ps(_mm256_mul_ps(v_x, _mm256_set1_ps(INV_2PI)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let x_red = _mm256_fmadd_ps(k, _mm256_set1_ps(-TWO_PI), v_x);
        let x2 = _mm256_mul_ps(x_red, x_red);

        let s1 = _mm256_fmadd_ps(_mm256_set1_ps(-0.00019841270), x2, _mm256_set1_ps(0.008333333));
        let s2 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(-0.16666667));
        let s3 = _mm256_fmadd_ps(s2, x2, _mm256_set1_ps(1.0));
        let s = _mm256_mul_ps(x_red, s3);

        let c1 = _mm256_fmadd_ps(_mm256_set1_ps(-0.0013888889), x2, _mm256_set1_ps(0.041666668));
        let c2 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(-0.5));
        let c = _mm256_fmadd_ps(c2, x2, _mm256_set1_ps(1.0));

        let mut out_s = [0.0; 8];
        let mut out_c = [0.0; 8];
        _mm256_storeu_ps(out_s.as_mut_ptr(), s);
        _mm256_storeu_ps(out_c.as_mut_ptr(), c);
        
        (out_s, out_c)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let lnx = x.approx_ln();
        let v_lnx = _mm256_set1_ps(lnx);
        let v_y = _mm256_loadu_ps(y.as_ptr());
        
        let y_lnx = _mm256_mul_ps(v_y, v_lnx);
        
        let mut y_lnx_arr = [0.0; 8];
        _mm256_storeu_ps(y_lnx_arr.as_mut_ptr(), y_lnx);
        
        batch_approx_exp_f32(y_lnx_arr)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm256_loadu_pd(x.as_ptr());
        
        let bits = _mm256_castpd_si256(v);
        let exp_shifted = _mm256_srli_epi64(bits, 52);
        let exp_int = _mm256_sub_epi64(exp_shifted, _mm256_set1_epi64x(1023));

        let lower128 = _mm_castsi128_ps(_mm256_castsi256_si128(exp_int));
        let upper128 = _mm_castsi128_ps(_mm256_extracti128_si256(exp_int, 1));
        let packed_ps = _mm_shuffle_ps(lower128, upper128, 136); // _MM_SHUFFLE(2, 0, 2, 0)
        let packed_epi32 = _mm_castps_si128(packed_ps);

        let exp_f64 = _mm256_cvtepi32_pd(packed_epi32);

        let mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
        let c_3ff0 = _mm256_set1_epi64x(0x3FF0000000000000);
        let m_bits = _mm256_or_si256(_mm256_and_si256(bits, mask), c_3ff0);
        let mantissa = _mm256_castsi256_pd(m_bits);

        let m_adj = _mm256_sub_pd(mantissa, _mm256_set1_pd(1.0));
        let neg_one_third = _mm256_set1_pd(-1.0 / 3.0);
        let c_1 = _mm256_set1_pd(1.0);
        let inner = _mm256_fmadd_pd(neg_one_third, m_adj, c_1);
        let ln_mantissa = _mm256_mul_pd(m_adj, inner);

        let ln_2 = _mm256_set1_pd(core::f64::consts::LN_2);
        let res = _mm256_fmadd_pd(exp_f64, ln_2, ln_mantissa);
        
        let mut out = [0.0; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_pd(x.as_ptr());
        
        let bits = _mm256_castpd_si256(v_x);
        let shifted = _mm256_srli_epi64(bits, 1);
        let added = _mm256_add_epi64(shifted, _mm256_set1_epi64x(0x1FF7A00000000000));
        let guess = _mm256_castsi256_pd(added);

        let div = _mm256_div_pd(v_x, guess);
        let sum = _mm256_add_pd(guess, div);
        let res = _mm256_mul_pd(_mm256_set1_pd(0.5), sum);
        
        let mut out = [0.0; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_pd(x.as_ptr());
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

        let mut out = [0.0; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_pd(x.as_ptr());
        
        const TWO_PI: f64 = core::f64::consts::PI * 2.0;
        const INV_2PI: f64 = 1.0 / TWO_PI;

        let k = _mm256_round_pd(_mm256_mul_pd(v_x, _mm256_set1_pd(INV_2PI)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let x_red = _mm256_fmadd_pd(k, _mm256_set1_pd(-TWO_PI), v_x);
        let x2 = _mm256_mul_pd(x_red, x_red);

        let s_p1 = _mm256_fmadd_pd(_mm256_set1_pd(2.75573192239858906e-6), x2, _mm256_set1_pd(-1.984126984126984e-4));
        let s_p2 = _mm256_fmadd_pd(s_p1, x2, _mm256_set1_pd(8.333333333333333e-3));
        let s_p3 = _mm256_fmadd_pd(s_p2, x2, _mm256_set1_pd(-1.666666666666667e-1));
        let s_p4 = _mm256_fmadd_pd(s_p3, x2, _mm256_set1_pd(1.0));
        let s = _mm256_mul_pd(x_red, s_p4);

        let c_p1 = _mm256_fmadd_pd(_mm256_set1_pd(2.48015873015873e-5), x2, _mm256_set1_pd(-1.388888888888889e-3));
        let c_p2 = _mm256_fmadd_pd(c_p1, x2, _mm256_set1_pd(4.166666666666667e-2));
        let c_p3 = _mm256_fmadd_pd(c_p2, x2, _mm256_set1_pd(-5.0e-1));
        let c = _mm256_fmadd_pd(c_p3, x2, _mm256_set1_pd(1.0));

        let mut out_s = [0.0; 4];
        let mut out_c = [0.0; 4];
        _mm256_storeu_pd(out_s.as_mut_ptr(), s);
        _mm256_storeu_pd(out_c.as_mut_ptr(), c);
        
        (out_s, out_c)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let lnx = x.approx_ln();
        let v_lnx = _mm256_set1_pd(lnx);
        let v_y = _mm256_loadu_pd(y.as_ptr());
        
        let y_lnx = _mm256_mul_pd(v_y, v_lnx);
        
        let mut y_lnx_arr = [0.0; 4];
        _mm256_storeu_pd(y_lnx_arr.as_mut_ptr(), y_lnx);
        
        batch_approx_exp_f64(y_lnx_arr)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
pub fn batch_fmadd_f32(x: [f32; 8], m: f32, a: f32) -> [f32; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_ps(x.as_ptr());
        let v_m = _mm256_set1_ps(m);
        let v_a = _mm256_set1_ps(a);
        let res = _mm256_fmadd_ps(v_x, v_m, v_a);
        let mut out = [0.0; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_ps(x.as_ptr());
        let v_mode = _mm256_set1_ps(mode);
        let v_lo = _mm256_set1_ps(sigma_lo);
        let v_hi = _mm256_set1_ps(sigma_hi);
        
        let mask = _mm256_cmp_ps(v_x, _mm256_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_ps(v_hi, v_lo, mask);
        let res = _mm256_fmadd_ps(v_x, sigma, v_mode);
        
        let mut out = [0.0; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
pub fn batch_fmadd_f64(x: [f64; 4], m: f64, a: f64) -> [f64; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_pd(x.as_ptr());
        let v_m = _mm256_set1_pd(m);
        let v_a = _mm256_set1_pd(a);
        let res = _mm256_fmadd_pd(v_x, v_m, v_a);
        let mut out = [0.0; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm256_loadu_pd(x.as_ptr());
        let v_mode = _mm256_set1_pd(mode);
        let v_lo = _mm256_set1_pd(sigma_lo);
        let v_hi = _mm256_set1_pd(sigma_hi);
        
        let mask = _mm256_cmp_pd(v_x, _mm256_setzero_pd(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_pd(v_hi, v_lo, mask);
        let res = _mm256_fmadd_pd(v_x, sigma, v_mode);
        
        let mut out = [0.0; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let sigma = if x[i] < 0.0 { sigma_lo } else { sigma_hi };
            out[i] = x[i].mul_add(sigma, mode);
        }
        out
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
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits(), "Mismatch at {}: batch {}, scalar {}", input[i], batch_res[i], scalar_res);
        }
    }

    #[test]
    fn test_batch_approx_exp_f32() {
        let input = [0.0, 1.0, -1.0, 5.0, -5.0, 10.0, 50.0, -50.0];
        let batch_res = batch_approx_exp_f32(input);
        for i in 0..8 {
            let scalar_res = crate::approx_exp_f32(input[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits(), "Mismatch at {}: batch: {} ({:x}), scalar: {} ({:x})", input[i], batch_res[i], batch_res[i].to_bits(), scalar_res, scalar_res.to_bits());
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
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits(), "Mismatch at {} ^{}: batch {} ({:x}), scalar {} ({:x})", base, pwr[i], batch_res[i], batch_res[i].to_bits(), scalar_res, scalar_res.to_bits());
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
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_exp_f64() {
        let input = [0.0, 1.0, -5.0, 50.0];
        let batch_res = batch_approx_exp_f64(input);
        for i in 0..4 {
            let scalar_res = crate::approx_exp_f64(input[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits(), "Mismatch at {}: batch {} ({:x}), scalar {} ({:x})", input[i], batch_res[i], batch_res[i].to_bits(), scalar_res, scalar_res.to_bits());
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
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits(), "Mismatch at {} ^{}: batch {} ({:x}), scalar {} ({:x})", base, pwr[i], batch_res[i], batch_res[i].to_bits(), scalar_res, scalar_res.to_bits());
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
}
