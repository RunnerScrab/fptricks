use core::arch::x86_64::*;
use crate::raw_batch_arith::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_ln_f32(v: __m256) -> __m256 {
    unsafe {
        let bits = _mm256_castps_si256(v);
        let shifted = _mm256_srli_epi32(bits, 23);
        let exp_int = _mm256_sub_epi32(shifted, _mm256_set1_epi32(127));
        let exp_f32 = _mm256_cvtepi32_ps(exp_int);

        let mask = _mm256_set1_epi32(0x007FFFFF);
        let c_3f80 = _mm256_set1_epi32(0x3F800000);
        let m_bits = _mm256_or_si256(_mm256_and_si256(bits, mask), c_3f80);
        let mantissa = _mm256_castsi256_ps(m_bits);

        let m_adj = _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0));
        
        let neg_one_third = _mm256_set1_ps(-1.0/3.0);
        let c_1 = _mm256_set1_ps(1.0);
        
        let inner = _mm256_fmadd_ps(neg_one_third, m_adj, c_1);
        let ln_mantissa = _mm256_mul_ps(m_adj, inner);

        let ln_2 = _mm256_set1_ps(core::f32::consts::LN_2);
        _mm256_fmadd_ps(exp_f32, ln_2, ln_mantissa)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_ln_f32(v: __m128) -> __m128 {
    unsafe {
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
        _mm_fmadd_ps(exp_f32, ln_2, ln_mantissa)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_sqrt_f32(v_x: __m256) -> __m256 {
    unsafe { _mm256_sqrt_ps(v_x) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_sqrt_f32(v_x: __m128) -> __m128 {
    unsafe { _mm_sqrt_ps(v_x) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_exp_f32(v_x: __m256) -> __m256 {
    unsafe {
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
        _mm256_blendv_ps(rv_masked, v_inf, is_inf)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_exp_f32(v_x: __m128) -> __m128 {
    unsafe {
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
        _mm_blendv_ps(rv_masked, v_inf, is_inf)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_powf_f32(x: f32, v_y: __m256) -> __m256 {
    unsafe {
        let lnx = crate::logarithmic::approx_ln_f32(x);
        let v_lnx = _mm256_set1_ps(lnx);
        let y_lnx = _mm256_mul_ps(v_y, v_lnx);
        raw_batch_approx_exp_f32(y_lnx)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_powf_f32(x: f32, v_y: __m128) -> __m128 {
    unsafe {
        let lnx = crate::logarithmic::approx_ln_f32(x);
        let v_lnx = _mm_set1_ps(lnx);
        let y_lnx = _mm_mul_ps(v_y, v_lnx);
        raw_batch4_approx_exp_f32(y_lnx)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_ln_f64(v: __m256d) -> __m256d {
    unsafe {
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
        _mm256_fmadd_pd(exp_f64, ln_2, ln_mantissa)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_sqrt_f64(v_x: __m256d) -> __m256d {
    unsafe { _mm256_sqrt_pd(v_x) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_exp_f64(v_x: __m256d) -> __m256d {
    unsafe {
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
        _mm256_blendv_pd(rv_masked, v_inf, is_inf)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_powf_f64(x: f64, v_y: __m256d) -> __m256d {
    unsafe {
        let lnx = crate::logarithmic::approx_ln_f64(x);
        let v_lnx = _mm256_set1_pd(lnx);
        let y_lnx = _mm256_mul_pd(v_y, v_lnx);
        raw_batch_approx_exp_f64(y_lnx)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_cbrt_f32(v_x: __m256) -> __m256 {
    unsafe {
        let bits = _mm256_castps_si256(v_x);

        let sign_mask = _mm256_set1_epi32(0x80000000u32 as i32);
        let abs_mask = _mm256_set1_epi32(0x7FFFFFFFi32);

        let v_sign = _mm256_and_si256(bits, sign_mask);
        let v_abs_bits = _mm256_and_si256(bits, abs_mask);
        let v_abs_x = _mm256_castsi256_ps(v_abs_bits);

        let mut abs_bits_arr = [0u32; 8];
        _mm256_storeu_si256(abs_bits_arr.as_mut_ptr() as *mut __m256i, v_abs_bits);
        for i in 0..8 {
            abs_bits_arr[i] = abs_bits_arr[i] / 3 + 0x2a514067;
        }
        let v_guess =
            _mm256_castsi256_ps(_mm256_loadu_si256(abs_bits_arr.as_ptr() as *const __m256i));

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
        _mm256_castsi256_ps(res_bits)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_cbrt_f32(v_x: __m128) -> __m128 {
    unsafe {
        let bits = _mm_castps_si128(v_x);

        let sign_mask = _mm_set1_epi32(0x80000000u32 as i32);
        let abs_mask = _mm_set1_epi32(0x7FFFFFFFi32);

        let v_sign = _mm_and_si128(bits, sign_mask);
        let v_abs_bits = _mm_and_si128(bits, abs_mask);
        let v_abs_x = _mm_castsi128_ps(v_abs_bits);

        let mut abs_bits_arr = [0u32; 4];
        _mm_storeu_si128(abs_bits_arr.as_mut_ptr() as *mut __m128i, v_abs_bits);
        for i in 0..4 {
            abs_bits_arr[i] = abs_bits_arr[i] / 3 + 0x2a514067;
        }
        let v_guess =
            _mm_castsi128_ps(_mm_loadu_si128(abs_bits_arr.as_ptr() as *const __m128i));

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
        _mm_castsi128_ps(res_bits)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_chunk_approx_powf_cols_f32(v_x: __m256, v_y: __m256) -> __m256 {
    unsafe {
        let v_lnx = raw_batch_approx_ln_f32(v_x);
        let y_lnx = _mm256_mul_ps(v_y, v_lnx);
        raw_batch_approx_exp_f32(y_lnx)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_chunk4_approx_powf_cols_f32(v_x: __m128, v_y: __m128) -> __m128 {
    unsafe {
        let v_lnx = raw_batch4_approx_ln_f32(v_x);
        let y_lnx = _mm_mul_ps(v_y, v_lnx);
        raw_batch4_approx_exp_f32(y_lnx)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_powi_cols_f32(mut v_base: __m256, v_n: __m256i) -> __m256 {
    unsafe {
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

        let v_is_neg = _mm256_castsi256_ps(v_is_neg);
        let v_is_zero = _mm256_castsi256_ps(v_is_zero);

        let v_inv = raw_batch_approx_inv_f32(v_result);

        v_result = _mm256_blendv_ps(v_result, v_inv, v_is_neg);
        _mm256_blendv_ps(v_result, _mm256_set1_ps(1.0), v_is_zero)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch4_approx_powi_cols_f32(mut v_base: __m128, v_n: __m128i) -> __m128 {
    unsafe {
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

        let v_is_neg = _mm_castsi128_ps(v_is_neg);
        let v_is_zero = _mm_castsi128_ps(v_is_zero);

        let v_inv = raw_batch4_approx_inv_f32(v_result);

        v_result = _mm_blendv_ps(v_result, v_inv, v_is_neg);
        _mm_blendv_ps(v_result, _mm_set1_ps(1.0), v_is_zero)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_cbrt_f64(v_x: __m256d) -> __m256d {
    unsafe {
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
        _mm256_castsi256_pd(res_bits)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_chunk_approx_powf_cols_f64(v_x: __m256d, v_y: __m256d) -> __m256d {
    unsafe {
        let v_lnx = raw_batch_approx_ln_f64(v_x);
        let y_lnx = _mm256_mul_pd(v_y, v_lnx);
        raw_batch_approx_exp_f64(y_lnx)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub unsafe fn raw_batch_approx_powi_cols_f64(mut v_base: __m256d, v_n: __m128i) -> __m256d {
    unsafe {
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

        let v_inv = raw_batch_approx_inv_f64(v_result);

        v_result = _mm256_blendv_pd(v_result, v_inv, _mm256_castsi256_pd(v_is_neg));
        _mm256_blendv_pd(
            v_result,
            _mm256_set1_pd(1.0),
            _mm256_castsi256_pd(v_is_zero),
        )
    }
}
