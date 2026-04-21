use core::arch::x86_64::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_approx_inv_f32(v_x: __m256) -> __m256 {
    unsafe {
        let bits = _mm256_castps_si256(v_x);
        let magic = _mm256_set1_epi32(0x7EF127EA_u32 as i32);
        let y0_bits = _mm256_sub_epi32(magic, bits);
        let v_y0 = _mm256_castsi256_ps(y0_bits);

        _mm256_mul_ps(v_y0, _mm256_fnmadd_ps(v_x, v_y0, _mm256_set1_ps(2.0)))
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch4_approx_inv_f32(v_x: __m128) -> __m128 {
    unsafe {
        let bits = _mm_castps_si128(v_x);
        let magic = _mm_set1_epi32(0x7EF127EA_u32 as i32);
        let y0_bits = _mm_sub_epi32(magic, bits);
        let v_y0 = _mm_castsi128_ps(y0_bits);

        _mm_mul_ps(v_y0, _mm_fnmadd_ps(v_x, v_y0, _mm_set1_ps(2.0)))
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_approx_inv_f64(v_x: __m256d) -> __m256d {
    unsafe {
        let bits = _mm256_castpd_si256(v_x);
        let magic = _mm256_set1_epi64x(0x7FDE623822835EEA_u64 as i64);
        let y0_bits = _mm256_sub_epi64(magic, bits);
        let v_y0 = _mm256_castsi256_pd(y0_bits);

        let two = _mm256_set1_pd(2.0);
        let v_y1 = _mm256_mul_pd(v_y0, _mm256_fnmadd_pd(v_x, v_y0, two));
        _mm256_mul_pd(v_y1, _mm256_fnmadd_pd(v_x, v_y1, two))
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_fmadd_cols_f32(v_x: __m256, v_m: __m256, v_a: __m256) -> __m256 {
    unsafe { _mm256_fmadd_ps(v_x, v_m, v_a) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch4_fmadd_cols_f32(v_x: __m128, v_m: __m128, v_a: __m128) -> __m128 {
    unsafe { _mm_fmadd_ps(v_x, v_m, v_a) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_fmadd_cols_f64(v_x: __m256d, v_m: __m256d, v_a: __m256d) -> __m256d {
    unsafe { _mm256_fmadd_pd(v_x, v_m, v_a) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_fmadd_f32(v_x: __m256, m: f32, a: f32) -> __m256 {
    unsafe {
        let v_m = _mm256_set1_ps(m);
        let v_a = _mm256_set1_ps(a);
        _mm256_fmadd_ps(v_x, v_m, v_a)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch4_fmadd_f32(v_x: __m128, m: f32, a: f32) -> __m128 {
    unsafe {
        let v_m = _mm_set1_ps(m);
        let v_a = _mm_set1_ps(a);
        _mm_fmadd_ps(v_x, v_m, v_a)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_asymmetric_fma_f32(v_x: __m256, mode: f32, sigma_lo: f32, sigma_hi: f32) -> __m256 {
    unsafe {
        let v_mode = _mm256_set1_ps(mode);
        let v_lo = _mm256_set1_ps(sigma_lo);
        let v_hi = _mm256_set1_ps(sigma_hi);

        let mask = _mm256_cmp_ps(v_x, _mm256_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_ps(v_hi, v_lo, mask);
        _mm256_fmadd_ps(v_x, sigma, v_mode)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch4_asymmetric_fma_f32(v_x: __m128, mode: f32, sigma_lo: f32, sigma_hi: f32) -> __m128 {
    unsafe {
        let v_mode = _mm_set1_ps(mode);
        let v_lo = _mm_set1_ps(sigma_lo);
        let v_hi = _mm_set1_ps(sigma_hi);

        let mask = _mm_cmp_ps(v_x, _mm_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm_blendv_ps(v_hi, v_lo, mask);
        _mm_fmadd_ps(v_x, sigma, v_mode)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_asymmetric_fma_cols_f32(v_x: __m256, v_mode: __m256, v_lo: __m256, v_hi: __m256) -> __m256 {
    unsafe {
        let mask = _mm256_cmp_ps(v_x, _mm256_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_ps(v_hi, v_lo, mask);
        _mm256_fmadd_ps(v_x, sigma, v_mode)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch4_asymmetric_fma_cols_f32(v_x: __m128, v_mode: __m128, v_lo: __m128, v_hi: __m128) -> __m128 {
    unsafe {
        let mask = _mm_cmp_ps(v_x, _mm_setzero_ps(), _CMP_LT_OQ);
        let sigma = _mm_blendv_ps(v_hi, v_lo, mask);
        _mm_fmadd_ps(v_x, sigma, v_mode)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_fmadd_f64(v_x: __m256d, m: f64, a: f64) -> __m256d {
    unsafe {
        let v_m = _mm256_set1_pd(m);
        let v_a = _mm256_set1_pd(a);
        _mm256_fmadd_pd(v_x, v_m, v_a)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_asymmetric_fma_f64(v_x: __m256d, mode: f64, sigma_lo: f64, sigma_hi: f64) -> __m256d {
    unsafe {
        let v_mode = _mm256_set1_pd(mode);
        let v_lo = _mm256_set1_pd(sigma_lo);
        let v_hi = _mm256_set1_pd(sigma_hi);

        let mask = _mm256_cmp_pd(v_x, _mm256_setzero_pd(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_pd(v_hi, v_lo, mask);
        _mm256_fmadd_pd(v_x, sigma, v_mode)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch_asymmetric_fma_cols_f64(v_x: __m256d, v_mode: __m256d, v_lo: __m256d, v_hi: __m256d) -> __m256d {
    unsafe {
        let mask = _mm256_cmp_pd(v_x, _mm256_setzero_pd(), _CMP_LT_OQ);
        let sigma = _mm256_blendv_pd(v_hi, v_lo, mask);
        _mm256_fmadd_pd(v_x, sigma, v_mode)
    }
}


#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline(always)]
pub fn raw_batch_div_cols_f32(v_x: __m128, v_y: __m128) -> __m128 {
    unsafe { _mm_div_ps(v_x, v_y) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline(always)]
pub fn raw_batch_div_cols_f64(v_x: __m256d, v_y: __m256d) -> __m256d {
    unsafe { _mm256_div_pd(v_x, v_y) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline(always)]
pub fn raw_batch4_mul_cols_f32(v_x: __m128, v_y: __m128) -> __m128 {
    unsafe { _mm_mul_ps(v_x, v_y) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline(always)]
pub fn raw_batch4_add_cols_f32(v_x: __m128, v_y: __m128) -> __m128 {
    unsafe { _mm_add_ps(v_x, v_y) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
pub fn raw_batch4_fma_cols_f32(v_x: __m128, v_y: __m128, v_z: __m128) -> __m128 {
    unsafe { _mm_fmadd_ps(v_x, v_y, v_z) }
}
