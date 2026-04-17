use crate::FastFloatFnHaver;
use core::mem::MaybeUninit;
use crate::batch_arith::{batch_approx_inv_f32, batch_approx_inv_f64, batch4_approx_inv_f32};

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
pub fn batch4_approx_ln_f32(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v: __m128 = core::mem::transmute(x);

        let bits = _mm_castps_si128(v);
        let shifted = _mm_srli_epi32(bits, 23);
        let exp_int = _mm_sub_epi32(shifted, _mm_set1_epi32(127));
        let exp_f32 = _mm_cvtepi32_ps(exp_int);

        let mask = _mm_set1_epi32(0x007FFFFF);
        let c_3f80 = _mm_set1_epi32(0x3F800000);
        let m_bits = _mm_or_si128(_mm_and_si128(bits, mask), c_3f80);
        let mantissa = _mm_castsi128_ps(m_bits);

        let m_adj = _mm_sub_ps(mantissa, _mm_set1_ps(1.0));

        let neg_one_third = _mm_set1_ps(-1.0 / 3.0);
        let c_1 = _mm_set1_ps(1.0);
        let inner = _mm_fmadd_ps(neg_one_third, m_adj, c_1);
        let ln_mantissa = _mm_mul_ps(m_adj, inner);

        let ln_2 = _mm_set1_ps(core::f32::consts::LN_2);
        let res = _mm_fmadd_ps(exp_f32, ln_2, ln_mantissa);

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
pub fn batch4_approx_sqrt_f32(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m128 = core::mem::transmute(x);
        let res = _mm_sqrt_ps(v_x);
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

        const LN2_HI: f32 = 0.69314575;
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
pub fn batch4_approx_exp_f32(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m128 = core::mem::transmute(x);

        let zero = _mm_setzero_ps();
        let mask_ltz = _mm_cmp_ps(v_x, zero, _CMP_LT_OQ);
        let is_inf = _mm_cmp_ps(v_x, _mm_set1_ps(88.72283), _CMP_GT_OQ);
        let is_z = _mm_cmp_ps(v_x, _mm_set1_ps(-87.33654), _CMP_LT_OQ);

        let xv = _mm_blendv_ps(_mm_set1_ps(0.5), _mm_set1_ps(-0.5), mask_ltz);

        let n = _mm_cvttps_epi32(_mm_fmadd_ps(
            v_x,
            _mm_set1_ps(core::f32::consts::LOG2_E),
            xv,
        ));
        let nf = _mm_cvtepi32_ps(n);
        let neg_nf = _mm_sub_ps(zero, nf);

        const LN2_HI: f32 = 0.69314575;
        const LN2_LO: f32 = 0.0000014286068;

        let inner = _mm_fmadd_ps(neg_nf, _mm_set1_ps(LN2_HI), v_x);
        let r = _mm_fmadd_ps(neg_nf, _mm_set1_ps(LN2_LO), inner);

        let exponent = _mm_add_epi32(n, _mm_set1_epi32(127));
        let two_n = _mm_castsi128_ps(_mm_slli_epi32(exponent, 23));

        let inv6 = _mm_set1_ps(1.0 / 6.0);
        let c_05 = _mm_set1_ps(0.5);
        let c_1 = _mm_set1_ps(1.0);

        let p1 = _mm_fmadd_ps(inv6, r, c_05);
        let p2 = _mm_fmadd_ps(r, p1, c_1);
        let res_r = _mm_fmadd_ps(r, p2, c_1);

        let rv = _mm_mul_ps(two_n, res_r);

        let v_inf = _mm_set1_ps(f32::INFINITY);

        let rv_masked = _mm_blendv_ps(rv, zero, is_z);
        let res = _mm_blendv_ps(rv_masked, v_inf, is_inf);

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
pub fn batch4_approx_powf_f32(x: f32, y: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let lnx = x.approx_ln();
        let v_lnx = _mm_set1_ps(lnx);
        let v_y: __m128 = core::mem::transmute(y);

        let y_lnx = _mm_mul_ps(v_y, v_lnx);
        let y_lnx_arr: [f32; 4] = core::mem::transmute(y_lnx);

        batch4_approx_exp_f32(y_lnx_arr)
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
        let mut abs_bits_arr = [0u32; 8];
        _mm256_storeu_si256(abs_bits_arr.as_mut_ptr() as *mut __m256i, v_abs_bits);
        for i in 0..8 {
            abs_bits_arr[i] = abs_bits_arr[i] / 3 + 0x2a514067;
        }
        let v_guess =
            _mm256_castsi256_ps(_mm256_loadu_si256(abs_bits_arr.as_ptr() as *const __m256i));

        // Newton-Raphson: refined = 0.6666667 * guess + abs_x / (3.0 * guess * guess)
        let g2 = _mm256_mul_ps(v_guess, v_guess);
        let three_g2 = _mm256_mul_ps(_mm256_set1_ps(3.0), g2);

        let inv_3g2 = _mm256_rcp_ps(three_g2);
        let inv_3g2_refined = _mm256_mul_ps(
            inv_3g2,
            _mm256_fnmadd_ps(three_g2, inv_3g2, _mm256_set1_ps(2.0)),
        );

        let term2 = _mm256_mul_ps(v_abs_x, inv_3g2_refined);
        let refined = _mm256_fmadd_ps(_mm256_set1_ps(0.6666667), v_guess, term2);

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
pub fn batch4_approx_cbrt_f32(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x: __m128 = core::mem::transmute(x);
        let bits = _mm_castps_si128(v_x);

        let sign_mask = _mm_set1_epi32(0x80000000u32 as i32);
        let abs_mask = _mm_set1_epi32(0x7FFFFFFFi32);

        let v_sign = _mm_and_si128(bits, sign_mask);
        let v_abs_bits = _mm_and_si128(bits, abs_mask);
        let v_abs_x = _mm_castsi128_ps(v_abs_bits);

        // guess = bits / 3 + 0x2a514067 (vectorized version of scalar magic)
        let mut abs_bits_arr = [0u32; 4];
        _mm_storeu_si128(abs_bits_arr.as_mut_ptr() as *mut __m128i, v_abs_bits);
        for i in 0..4 {
            abs_bits_arr[i] = abs_bits_arr[i] / 3 + 0x2a514067;
        }
        let v_guess =
            _mm_castsi128_ps(_mm_loadu_si128(abs_bits_arr.as_ptr() as *const __m128i));

        // Newton-Raphson: refined = 0.6666667 * guess + abs_x / (3.0 * guess * guess)
        let g2 = _mm_mul_ps(v_guess, v_guess);
        let three_g2 = _mm_mul_ps(_mm_set1_ps(3.0), g2);

        let inv_3g2 = _mm_rcp_ps(three_g2);
        let inv_3g2_refined = _mm_mul_ps(
            inv_3g2,
            _mm_fnmadd_ps(three_g2, inv_3g2, _mm_set1_ps(2.0)),
        );

        let term2 = _mm_mul_ps(v_abs_x, inv_3g2_refined);
        let refined = _mm_fmadd_ps(_mm_set1_ps(0.6666667), v_guess, term2);

        let res_bits = _mm_or_si128(
            _mm_and_si128(_mm_castps_si128(refined), abs_mask),
            v_sign,
        );
        let res = _mm_castsi128_ps(res_bits);

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
pub fn chunk4_approx_powf_cols_f32(x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_y: __m128 = core::mem::transmute(y);

        let lnx = batch4_approx_ln_f32(x);
        let v_lnx: __m128 = core::mem::transmute(lnx);

        let y_lnx = _mm_mul_ps(v_y, v_lnx);
        let y_lnx_arr: [f32; 4] = core::mem::transmute(y_lnx);

        batch4_approx_exp_f32(y_lnx_arr)
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

        let mut out = [0.0; 8];
        let v_is_neg = _mm256_castsi256_ps(v_is_neg);
        let v_is_zero = _mm256_castsi256_ps(v_is_zero);

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
pub fn batch4_approx_powi_cols_f32(x: [f32; 4], n: [i32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let mut v_base = _mm_loadu_ps(x.as_ptr());
        let v_n = _mm_loadu_si128(n.as_ptr() as *const __m128i);
        let v_is_neg = _mm_cmpgt_epi32(_mm_setzero_si128(), v_n);
        let v_is_zero = _mm_cmpeq_epi32(_mm_setzero_si128(), v_n);

        let mut v_e = _mm_abs_epi32(v_n);
        let mut v_result = _mm_set1_ps(1.0);

        for _ in 0..31 {
            if _mm_test_all_zeros(v_e, v_e) == 1 {
                break;
            }
            let bit_set = _mm_cmpeq_epi32(
                _mm_and_si128(v_e, _mm_set1_epi32(1)),
                _mm_set1_epi32(1),
            );
            let v_mul = _mm_mul_ps(v_result, v_base);
            v_result = _mm_blendv_ps(v_result, v_mul, _mm_castsi128_ps(bit_set));

            v_base = _mm_mul_ps(v_base, v_base);
            v_e = _mm_srli_epi32(v_e, 1);
        }

        let mut out = [0.0; 4];
        let v_is_neg = _mm_castsi128_ps(v_is_neg);
        let v_is_zero = _mm_castsi128_ps(v_is_zero);

        _mm_storeu_ps(out.as_mut_ptr(), v_result);
        let res_inv = batch4_approx_inv_f32(out);
        let v_inv = _mm_loadu_ps(res_inv.as_ptr());

        v_result = _mm_blendv_ps(v_result, v_inv, v_is_neg);
        v_result = _mm_blendv_ps(v_result, _mm_set1_ps(1.0), v_is_zero);

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
pub fn batch_approx_powf_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N]) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f32;
        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let mut ax = [0.0; 8];
            let mut ay = [0.0; 8];
            _mm256_storeu_ps(ax.as_mut_ptr(), vx);
            _mm256_storeu_ps(ay.as_mut_ptr(), vy);
            let o = chunk_approx_powf_cols_f32(ax, ay);
            _mm256_storeu_ps(out_ptr.add(i), _mm256_loadu_ps(o.as_ptr()));
            i += 8;
        }
        while i < len {
            out_ptr.add(i).write(crate::approx_powf_f32(x[i], y[i]));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i].write(crate::approx_powf_f32(x[i], y[i]));
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
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
            let mut ax = [0.0; 8];
            let mut ay = [0.0; 8];
            _mm256_storeu_ps(ax.as_mut_ptr(), vx);
            _mm256_storeu_ps(ay.as_mut_ptr(), vy);
            let o = chunk_approx_powf_cols_f32(ax, ay);
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
pub fn batch_approx_powf_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N]) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f64;
        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let mut ax = [0.0; 4];
            let mut ay = [0.0; 4];
            _mm256_storeu_pd(ax.as_mut_ptr(), vx);
            _mm256_storeu_pd(ay.as_mut_ptr(), vy);
            let o = chunk_approx_powf_cols_f64(ax, ay);
            _mm256_storeu_pd(out_ptr.add(i), _mm256_loadu_pd(o.as_ptr()));
            i += 4;
        }
        while i < len {
            out_ptr.add(i).write(crate::approx_powf_f64(x[i], y[i]));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i].write(crate::approx_powf_f64(x[i], y[i]));
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
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
            let mut ax = [0.0; 4];
            let mut ay = [0.0; 4];
            _mm256_storeu_pd(ax.as_mut_ptr(), vx);
            _mm256_storeu_pd(ay.as_mut_ptr(), vy);
            let o = chunk_approx_powf_cols_f64(ax, ay);
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let x = [1.2, 0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 0.8];
        let y = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
        let batch_res = batch_approx_powf_cols_f32(&x, &y);
        for i in 0..8 {
            let scalar_res = x[i].approx_powf(y[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch_approx_powf_cols_f64() {
        let x = [1.2, 0.8, 1.2, 0.8];
        let y = [2.5, 2.5, 2.5, 2.5];
        let batch_res = batch_approx_powf_cols_f64(&x, &y);
        for i in 0..4 {
            let scalar_res = x[i].approx_powf(y[i]);
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
    fn test_batch4_approx_ln_f32() {
        let input = [1.0, 2.0, 10.0, 0.5];
        let batch_res = batch4_approx_ln_f32(input);
        for i in 0..4 {
            let scalar_res = crate::approx_ln_f32(input[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_approx_sqrt_f32() {
        let input = [1.0, 4.0, 10.0, 0.5];
        let batch_res = batch4_approx_sqrt_f32(input);
        for i in 0..4 {
            let scalar_res = crate::approx_sqrt_f32(input[i]);
            assert!((batch_res[i] - scalar_res).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch4_approx_exp_f32() {
        let input = [0.0, 1.0, -5.0, 50.0];
        let batch_res = batch4_approx_exp_f32(input);
        for i in 0..4 {
            let scalar_res = crate::approx_exp_f32(input[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_approx_powf_f32() {
        let base = 2.0;
        let pwr = [0.0, 1.0, 3.5, -2.5];
        let batch_res = batch4_approx_powf_f32(base, pwr);
        for i in 0..4 {
            let scalar_res = crate::approx_powf_f32(base, pwr[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_approx_cbrt_f32() {
        let input = [1.0, 8.0, 27.0, 0.5];
        let batch_res = batch4_approx_cbrt_f32(input);
        for i in 0..4 {
            let scalar_res = crate::approx_cbrt_f32(input[i]);
            assert!((batch_res[i] - scalar_res).abs() < 1e-6);
        }
    }

    #[test]
    fn test_chunk4_approx_powf_cols_f32() {
        let x = [1.2, 0.8, 1.2, 0.8];
        let y = [2.5, 2.5, 2.5, 2.5];
        let batch_res = chunk4_approx_powf_cols_f32(x, y);
        for i in 0..4 {
            let scalar_res = x[i].approx_powf(y[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_approx_powi_cols_f32() {
        let x = [1.0, 2.0, 3.0, 0.5];
        let n = [0, 2, -1, 3];
        let batch_res = batch4_approx_powi_cols_f32(x, n);
        for i in 0..4 {
            let scalar_res = crate::approx_powi_f32(x[i], n[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }
}
