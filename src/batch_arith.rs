use crate::FastFloatFnHaver;
use core::mem::MaybeUninit;

#[inline(always)]
pub fn batch_approx_inv_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_approx_inv_f32(v_x);
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
pub fn batch4_approx_inv_f32(x: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_approx_inv_f32(v_x);
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
pub fn batch_approx_inv_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_approx_inv_f64(v_x);
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
        let res = crate::raw_batch_fmadd_cols_f32(v_x, v_m, v_a);
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
pub fn batch4_fmadd_cols_f32(x: [f32; 4], m: [f32; 4], a: [f32; 4]) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let v_x = _mm_loadu_ps(x.as_ptr());
        let v_m = _mm_loadu_ps(m.as_ptr());
        let v_a = _mm_loadu_ps(a.as_ptr());
        let res = crate::raw_batch4_fmadd_cols_f32(v_x, v_m, v_a);
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
        let res = crate::raw_batch_fmadd_cols_f64(v_x, v_m, v_a);
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
pub fn batch_fmadd_f32(x: [f32; 8], m: f32, a: f32) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_fmadd_f32(v_x, m, a);
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
pub fn batch4_fmadd_f32(x: [f32; 4], m: f32, a: f32) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_fmadd_f32(v_x, m, a);
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
pub fn batch_asymmetric_fma_f32(x: [f32; 8], mode: f32, sigma_lo: f32, sigma_hi: f32) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_asymmetric_fma_f32(v_x, mode, sigma_lo, sigma_hi);
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
pub fn batch4_asymmetric_fma_f32(x: [f32; 4], mode: f32, sigma_lo: f32, sigma_hi: f32) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_asymmetric_fma_f32(v_x, mode, sigma_lo, sigma_hi);
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
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let v_mode: core::arch::x86_64::__m256 = core::mem::transmute(mode);
        let v_lo: core::arch::x86_64::__m256 = core::mem::transmute(sigma_lo);
        let v_hi: core::arch::x86_64::__m256 = core::mem::transmute(sigma_hi);
        let res = crate::raw_batch_asymmetric_fma_cols_f32(v_x, v_mode, v_lo, v_hi);
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
pub fn batch4_asymmetric_fma_cols_f32(
    x: [f32; 4],
    mode: [f32; 4],
    sigma_lo: [f32; 4],
    sigma_hi: [f32; 4],
) -> [f32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let v_mode: core::arch::x86_64::__m128 = core::mem::transmute(mode);
        let v_lo: core::arch::x86_64::__m128 = core::mem::transmute(sigma_lo);
        let v_hi: core::arch::x86_64::__m128 = core::mem::transmute(sigma_hi);
        let res = crate::raw_batch4_asymmetric_fma_cols_f32(v_x, v_mode, v_lo, v_hi);
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
pub fn batch_fmadd_f64(x: [f64; 4], m: f64, a: f64) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_fmadd_f64(v_x, m, a);
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
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_asymmetric_fma_f64(v_x, mode, sigma_lo, sigma_hi);
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
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let v_mode: core::arch::x86_64::__m256d = core::mem::transmute(mode);
        let v_lo: core::arch::x86_64::__m256d = core::mem::transmute(sigma_lo);
        let v_hi: core::arch::x86_64::__m256d = core::mem::transmute(sigma_hi);
        let res = crate::raw_batch_asymmetric_fma_cols_f64(v_x, v_mode, v_lo, v_hi);
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
pub fn batch_div_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N]) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f32;

        while i + 31 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            _mm256_storeu_ps(out_ptr.add(i), _mm256_div_ps(vx0, vy0));

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            _mm256_storeu_ps(out_ptr.add(i + 8), _mm256_div_ps(vx1, vy1));

            let vx2 = _mm256_loadu_ps(x_ptr.add(i + 16));
            let vy2 = _mm256_loadu_ps(y_ptr.add(i + 16));
            _mm256_storeu_ps(out_ptr.add(i + 16), _mm256_div_ps(vx2, vy2));

            let vx3 = _mm256_loadu_ps(x_ptr.add(i + 24));
            let vy3 = _mm256_loadu_ps(y_ptr.add(i + 24));
            _mm256_storeu_ps(out_ptr.add(i + 24), _mm256_div_ps(vx3, vy3));

            i += 32;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            _mm256_storeu_ps(out_ptr.add(i), _mm256_div_ps(vx, vy));
            i += 8;
        }

        while i < len {
            out_ptr.add(i).write(*x_ptr.add(i) / *y_ptr.add(i));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i].write(x[i] / y[i]);
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_mul_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N]) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f32;

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
            out[i].write(x[i] * y[i]);
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch4_mul_cols_f32(x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        let vx = core::arch::x86_64::_mm_loadu_ps(x.as_ptr());
        let vy = core::arch::x86_64::_mm_loadu_ps(y.as_ptr());
        let res = crate::raw_batch4_mul_cols_f32(vx, vy);
        core::mem::transmute(res)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i] * y[i];
        }
        out
    }
}

#[inline(always)]
pub fn batch_add_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N]) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f32;

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
            out[i].write(x[i] + y[i]);
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_sub_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N]) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f32;

        while i + 31 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.sub(i));
            let vy0 = _mm256_loadu_ps(y_ptr.sub(i));
            let vz0 = _mm256_sub_ps(vx0, vy0);
            _mm256_storeu_ps(out_ptr.sub(i), vz0);

            let vx1 = _mm256_loadu_ps(x_ptr.sub(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.sub(i + 8));
            let vz1 = _mm256_sub_ps(vx1, vy1);
            _mm256_storeu_ps(out_ptr.sub(i + 8), vz1);

            let vx2 = _mm256_loadu_ps(x_ptr.sub(i + 16));
            let vy2 = _mm256_loadu_ps(y_ptr.sub(i + 16));
            let vz2 = _mm256_sub_ps(vx2, vy2);
            _mm256_storeu_ps(out_ptr.sub(i + 16), vz2);

            let vx3 = _mm256_loadu_ps(x_ptr.sub(i + 24));
            let vy3 = _mm256_loadu_ps(y_ptr.sub(i + 24));
            let vz3 = _mm256_sub_ps(vx3, vy3);
            _mm256_storeu_ps(out_ptr.sub(i + 24), vz3);

            i += 32;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.sub(i));
            let vy = _mm256_loadu_ps(y_ptr.sub(i));
            let vz = _mm256_sub_ps(vx, vy);
            _mm256_storeu_ps(out_ptr.sub(i), vz);
            i += 8;
        }

        while i < len {
            out_ptr.sub(i).write(*x_ptr.sub(i) + *y_ptr.sub(i));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i].write(x[i] - y[i]);
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch4_add_cols_f32(x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        let vx = core::arch::x86_64::_mm_loadu_ps(x.as_ptr());
        let vy = core::arch::x86_64::_mm_loadu_ps(y.as_ptr());
        let res = crate::raw_batch4_add_cols_f32(vx, vy);
        core::mem::transmute(res)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i] + y[i];
        }
        out
    }
}

#[inline(always)]
pub fn batch_fma_cols_f32<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        out[i].write(x[i].mul_add(y[i], z[i]));
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch4_fma_cols_f32(x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        let vx = core::arch::x86_64::_mm_loadu_ps(x.as_ptr());
        let vy = core::arch::x86_64::_mm_loadu_ps(y.as_ptr());
        let vz = core::arch::x86_64::_mm_loadu_ps(z.as_ptr());
        let res = crate::raw_batch4_fma_cols_f32(vx, vy, vz);
        core::mem::transmute(res)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = x[i].mul_add(y[i], z[i]);
        }
        out
    }
}

#[inline(always)]
pub fn batch_mul_3_cols_f32_into<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    out: &mut [f32; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 31 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            let vz0 = _mm256_loadu_ps(z_ptr.add(i));
            let res0 = _mm256_mul_ps(_mm256_mul_ps(vx0, vy0), vz0);
            _mm256_storeu_ps(out_ptr.add(i), res0);

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            let vz1 = _mm256_loadu_ps(z_ptr.add(i + 8));
            let res1 = _mm256_mul_ps(_mm256_mul_ps(vx1, vy1), vz1);
            _mm256_storeu_ps(out_ptr.add(i + 8), res1);

            let vx2 = _mm256_loadu_ps(x_ptr.add(i + 16));
            let vy2 = _mm256_loadu_ps(y_ptr.add(i + 16));
            let vz2 = _mm256_loadu_ps(z_ptr.add(i + 16));
            let res2 = _mm256_mul_ps(_mm256_mul_ps(vx2, vy2), vz2);
            _mm256_storeu_ps(out_ptr.add(i + 16), res2);

            let vx3 = _mm256_loadu_ps(x_ptr.add(i + 24));
            let vy3 = _mm256_loadu_ps(y_ptr.add(i + 24));
            let vz3 = _mm256_loadu_ps(z_ptr.add(i + 24));
            let res3 = _mm256_mul_ps(_mm256_mul_ps(vx3, vy3), vz3);
            _mm256_storeu_ps(out_ptr.add(i + 24), res3);

            i += 32;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let vz = _mm256_loadu_ps(z_ptr.add(i));
            let res = _mm256_mul_ps(_mm256_mul_ps(vx, vy), vz);
            _mm256_storeu_ps(out_ptr.add(i), res);
            i += 8;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) * *y_ptr.add(i) * *z_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] * y[i] * z[i];
        }
    }
}

#[inline(always)]
pub fn batch_mul_3_cols_f32<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_mul_3_cols_f32_into(x, y, z, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f32; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_mul_4_cols_f32_into<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    w: &[f32; N],
    out: &mut [f32; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let w_ptr = w.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 15 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            let vz0 = _mm256_loadu_ps(z_ptr.add(i));
            let vw0 = _mm256_loadu_ps(w_ptr.add(i));
            let res0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(vx0, vy0), vz0), vw0);
            _mm256_storeu_ps(out_ptr.add(i), res0);

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            let vz1 = _mm256_loadu_ps(z_ptr.add(i + 8));
            let vw1 = _mm256_loadu_ps(w_ptr.add(i + 8));
            let res1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(vx1, vy1), vz1), vw1);
            _mm256_storeu_ps(out_ptr.add(i + 8), res1);

            i += 16;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let vz = _mm256_loadu_ps(z_ptr.add(i));
            let vw = _mm256_loadu_ps(w_ptr.add(i));
            let res = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(vx, vy), vz), vw);
            _mm256_storeu_ps(out_ptr.add(i), res);
            i += 8;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) * *y_ptr.add(i) * *z_ptr.add(i) * *w_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] * y[i] * z[i] * w[i];
        }
    }
}

#[inline(always)]
pub fn batch_mul_4_cols_f32<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    w: &[f32; N],
) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_mul_4_cols_f32_into(x, y, z, w, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f32; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_add_3_cols_f32_into<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    out: &mut [f32; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 31 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            let vz0 = _mm256_loadu_ps(z_ptr.add(i));
            let res0 = _mm256_add_ps(_mm256_add_ps(vx0, vy0), vz0);
            _mm256_storeu_ps(out_ptr.add(i), res0);

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            let vz1 = _mm256_loadu_ps(z_ptr.add(i + 8));
            let res1 = _mm256_add_ps(_mm256_add_ps(vx1, vy1), vz1);
            _mm256_storeu_ps(out_ptr.add(i + 8), res1);

            let vx2 = _mm256_loadu_ps(x_ptr.add(i + 16));
            let vy2 = _mm256_loadu_ps(y_ptr.add(i + 16));
            let vz2 = _mm256_loadu_ps(z_ptr.add(i + 16));
            let res2 = _mm256_add_ps(_mm256_add_ps(vx2, vy2), vz2);
            _mm256_storeu_ps(out_ptr.add(i + 16), res2);

            let vx3 = _mm256_loadu_ps(x_ptr.add(i + 24));
            let vy3 = _mm256_loadu_ps(y_ptr.add(i + 24));
            let vz3 = _mm256_loadu_ps(z_ptr.add(i + 24));
            let res3 = _mm256_add_ps(_mm256_add_ps(vx3, vy3), vz3);
            _mm256_storeu_ps(out_ptr.add(i + 24), res3);

            i += 32;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let vz = _mm256_loadu_ps(z_ptr.add(i));
            let res = _mm256_add_ps(_mm256_add_ps(vx, vy), vz);
            _mm256_storeu_ps(out_ptr.add(i), res);
            i += 8;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) + *y_ptr.add(i) + *z_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] + y[i] + z[i];
        }
    }
}

#[inline(always)]
pub fn batch_add_3_cols_f32<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_add_3_cols_f32_into(x, y, z, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f32; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_add_4_cols_f32_into<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    w: &[f32; N],
    out: &mut [f32; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let w_ptr = w.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 15 < len {
            let vx0 = _mm256_loadu_ps(x_ptr.add(i));
            let vy0 = _mm256_loadu_ps(y_ptr.add(i));
            let vz0 = _mm256_loadu_ps(z_ptr.add(i));
            let vw0 = _mm256_loadu_ps(w_ptr.add(i));
            let res0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vx0, vy0), vz0), vw0);
            _mm256_storeu_ps(out_ptr.add(i), res0);

            let vx1 = _mm256_loadu_ps(x_ptr.add(i + 8));
            let vy1 = _mm256_loadu_ps(y_ptr.add(i + 8));
            let vz1 = _mm256_loadu_ps(z_ptr.add(i + 8));
            let vw1 = _mm256_loadu_ps(w_ptr.add(i + 8));
            let res1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vx1, vy1), vz1), vw1);
            _mm256_storeu_ps(out_ptr.add(i + 8), res1);

            i += 16;
        }

        while i + 7 < len {
            let vx = _mm256_loadu_ps(x_ptr.add(i));
            let vy = _mm256_loadu_ps(y_ptr.add(i));
            let vz = _mm256_loadu_ps(z_ptr.add(i));
            let vw = _mm256_loadu_ps(w_ptr.add(i));
            let res = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vx, vy), vz), vw);
            _mm256_storeu_ps(out_ptr.add(i), res);
            i += 8;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) + *y_ptr.add(i) + *z_ptr.add(i) + *w_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] + y[i] + z[i] + w[i];
        }
    }
}

#[inline(always)]
pub fn batch_add_4_cols_f32<const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
    w: &[f32; N],
) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_add_4_cols_f32_into(x, y, z, w, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f32; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_mul_3_cols_f64_into<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    out: &mut [f64; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
            _mm256_storeu_pd(out_ptr.add(i), _mm256_mul_pd(_mm256_mul_pd(vx0, vy0), vz0));

            let vx1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            let vz1 = _mm256_loadu_pd(z_ptr.add(i + 4));
            _mm256_storeu_pd(out_ptr.add(i + 4), _mm256_mul_pd(_mm256_mul_pd(vx1, vy1), vz1));

            let vx2 = _mm256_loadu_pd(x_ptr.add(i + 8));
            let vy2 = _mm256_loadu_pd(y_ptr.add(i + 8));
            let vz2 = _mm256_loadu_pd(z_ptr.add(i + 8));
            _mm256_storeu_pd(out_ptr.add(i + 8), _mm256_mul_pd(_mm256_mul_pd(vx2, vy2), vz2));

            let vx3 = _mm256_loadu_pd(x_ptr.add(i + 12));
            let vy3 = _mm256_loadu_pd(y_ptr.add(i + 12));
            let vz3 = _mm256_loadu_pd(z_ptr.add(i + 12));
            _mm256_storeu_pd(out_ptr.add(i + 12), _mm256_mul_pd(_mm256_mul_pd(vx3, vy3), vz3));

            i += 16;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let vz = _mm256_loadu_pd(z_ptr.add(i));
            let res = _mm256_mul_pd(_mm256_mul_pd(vx, vy), vz);
            _mm256_storeu_pd(out_ptr.add(i), res);
            i += 4;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) * *y_ptr.add(i) * *z_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] * y[i] * z[i];
        }
    }
}

pub fn batch_mul_3_cols_f64<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_mul_3_cols_f64_into(x, y, z, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f64; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_mul_4_cols_f64_into<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    w: &[f64; N],
    out: &mut [f64; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let w_ptr = w.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 7 < len {
            let vx0 = _mm256_loadu_pd(x_ptr.add(i));
            let vy0 = _mm256_loadu_pd(y_ptr.add(i));
            let vz0 = _mm256_loadu_pd(z_ptr.add(i));
            let vw0 = _mm256_loadu_pd(w_ptr.add(i));
            _mm256_storeu_pd(out_ptr.add(i), _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(vx0, vy0), vz0), vw0));

            let vx1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            let vz1 = _mm256_loadu_pd(z_ptr.add(i + 4));
            let vw1 = _mm256_loadu_pd(w_ptr.add(i + 4));
            _mm256_storeu_pd(out_ptr.add(i + 4), _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(vx1, vy1), vz1), vw1));

            i += 8;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let vz = _mm256_loadu_pd(z_ptr.add(i));
            let vw = _mm256_loadu_pd(w_ptr.add(i));
            let res = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(vx, vy), vz), vw);
            _mm256_storeu_pd(out_ptr.add(i), res);
            i += 4;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) * *y_ptr.add(i) * *z_ptr.add(i) * *w_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] * y[i] * z[i] * w[i];
        }
    }
}

#[inline(always)]
pub fn batch_mul_4_cols_f64<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    w: &[f64; N],
) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_mul_4_cols_f64_into(x, y, z, w, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f64; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_mul_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N]) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        out[i].write(x[i] * y[i]);
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_add_3_cols_f64_into<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    out: &mut [f64; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
            _mm256_storeu_pd(out_ptr.add(i), _mm256_add_pd(_mm256_add_pd(vx0, vy0), vz0));

            let vx1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            let vz1 = _mm256_loadu_pd(z_ptr.add(i + 4));
            _mm256_storeu_pd(out_ptr.add(i + 4), _mm256_add_pd(_mm256_add_pd(vx1, vy1), vz1));

            let vx2 = _mm256_loadu_pd(x_ptr.add(i + 8));
            let vy2 = _mm256_loadu_pd(y_ptr.add(i + 8));
            let vz2 = _mm256_loadu_pd(z_ptr.add(i + 8));
            _mm256_storeu_pd(out_ptr.add(i + 8), _mm256_add_pd(_mm256_add_pd(vx2, vy2), vz2));

            let vx3 = _mm256_loadu_pd(x_ptr.add(i + 12));
            let vy3 = _mm256_loadu_pd(y_ptr.add(i + 12));
            let vz3 = _mm256_loadu_pd(z_ptr.add(i + 12));
            _mm256_storeu_pd(out_ptr.add(i + 12), _mm256_add_pd(_mm256_add_pd(vx3, vy3), vz3));

            i += 16;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let vz = _mm256_loadu_pd(z_ptr.add(i));
            let res = _mm256_add_pd(_mm256_add_pd(vx, vy), vz);
            _mm256_storeu_pd(out_ptr.add(i), res);
            i += 4;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) + *y_ptr.add(i) + *z_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] + y[i] + z[i];
        }
    }
}

pub fn batch_add_3_cols_f64<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_add_3_cols_f64_into(x, y, z, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f64; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_add_4_cols_f64_into<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    w: &[f64; N],
    out: &mut [f64; N],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let z_ptr = z.as_ptr();
        let w_ptr = w.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i + 7 < len {
            let vx0 = _mm256_loadu_pd(x_ptr.add(i));
            let vy0 = _mm256_loadu_pd(y_ptr.add(i));
            let vz0 = _mm256_loadu_pd(z_ptr.add(i));
            let vw0 = _mm256_loadu_pd(w_ptr.add(i));
            _mm256_storeu_pd(out_ptr.add(i), _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(vx0, vy0), vz0), vw0));

            let vx1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            let vz1 = _mm256_loadu_pd(z_ptr.add(i + 4));
            let vw1 = _mm256_loadu_pd(w_ptr.add(i + 4));
            _mm256_storeu_pd(out_ptr.add(i + 4), _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(vx1, vy1), vz1), vw1));

            i += 8;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.add(i));
            let vy = _mm256_loadu_pd(y_ptr.add(i));
            let vz = _mm256_loadu_pd(z_ptr.add(i));
            let vw = _mm256_loadu_pd(w_ptr.add(i));
            let res = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(vx, vy), vz), vw);
            _mm256_storeu_pd(out_ptr.add(i), res);
            i += 4;
        }

        while i < len {
            *out_ptr.add(i) = *x_ptr.add(i) + *y_ptr.add(i) + *z_ptr.add(i) + *w_ptr.add(i);
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i] = x[i] + y[i] + z[i] + w[i];
        }
    }
}

pub fn batch_add_4_cols_f64<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
    w: &[f64; N],
) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    batch_add_4_cols_f64_into(x, y, z, w, unsafe {
        &mut *(out.as_mut_ptr() as *mut [f64; N])
    });
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_add_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N]) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f64;

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
            out[i].write(x[i] + y[i]);
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_sub_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N]) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        let len = N;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = out.as_mut_ptr() as *mut f64;

        while i + 15 < len {
            let vx0 = _mm256_loadu_pd(x_ptr.add(i));
            let vy0 = _mm256_loadu_pd(y_ptr.add(i));
            _mm256_storeu_pd(out_ptr.sub(i), _mm256_sub_pd(vx0, vy0));

            let vx1 = _mm256_loadu_pd(x_ptr.sub(i + 4));
            let vy1 = _mm256_loadu_pd(y_ptr.sub(i + 4));
            _mm256_storeu_pd(out_ptr.sub(i + 4), _mm256_sub_pd(vx1, vy1));

            let vx2 = _mm256_loadu_pd(x_ptr.sub(i + 8));
            let vy2 = _mm256_loadu_pd(y_ptr.sub(i + 8));
            _mm256_storeu_pd(out_ptr.sub(i + 8), _mm256_sub_pd(vx2, vy2));

            let vx3 = _mm256_loadu_pd(x_ptr.sub(i + 12));
            let vy3 = _mm256_loadu_pd(y_ptr.sub(i + 12));
            _mm256_storeu_pd(out_ptr.sub(i + 12), _mm256_sub_pd(vx3, vy3));

            i += 16;
        }

        while i + 3 < len {
            let vx = _mm256_loadu_pd(x_ptr.sub(i));
            let vy = _mm256_loadu_pd(y_ptr.sub(i));
            _mm256_storeu_pd(out_ptr.sub(i), _mm256_sub_pd(vx, vy));
            i += 4;
        }

        while i < len {
            out_ptr.sub(i).write(*x_ptr.sub(i) + *y_ptr.sub(i));
            i += 1;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for i in 0..N {
            out[i].write(x[i] - y[i]);
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}


#[inline(always)]
pub fn batch_fma_cols_f64<const N: usize>(
    x: &[f64; N],
    y: &[f64; N],
    z: &[f64; N],
) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
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
        let out_ptr = out.as_mut_ptr() as *mut f64;

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
            out[i].write(x[i].mul_add(y[i], z[i]));
        }
    }
    unsafe { core::mem::transmute_copy(&out) }
}

// Slice versions

#[inline(always)]
pub fn batch_mul_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i] * y[i];
    }
}

#[inline(always)]
pub fn batch_add_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i] + y[i];
    }
}

#[inline(always)]
pub fn batch_fma_vec_f32(x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == z.len() && z.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i].mul_add(y[i], z[i]);
    }
}

#[inline(always)]
pub fn batch_mul_vec_f64(x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i] * y[i];
    }
}

#[inline(always)]
pub fn batch_add_vec_f64(x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i] + y[i];
    }
}

#[inline(always)]
pub fn batch_fma_vec_f64(x: &[f64], y: &[f64], z: &[f64], out: &mut [f64]) {
    assert!(x.len() == y.len() && y.len() == z.len() && z.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i].mul_add(y[i], z[i]);
    }
}

#[inline(always)]
pub fn batch_powf_cols_f32<const N: usize>(x: &[f32; N], y: &[f32; N]) -> [f32; N] {
    let mut out: [MaybeUninit<f32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        out[i].write(x[i].powf(y[i]));
    }
    unsafe { core::mem::transmute_copy(&out) }
}

#[inline(always)]
pub fn batch_powf_vec_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len() && y.len() == out.len());
    for i in 0..x.len() {
        out[i] = x[i].powf(y[i]);
    }
}

#[inline(always)]
pub fn batch_powf_cols_f64<const N: usize>(x: &[f64; N], y: &[f64; N]) -> [f64; N] {
    let mut out: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        out[i].write(x[i].powf(y[i]));
    }
    unsafe { core::mem::transmute_copy(&out) }
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

    #[test]
    fn test_batch_mul_multi_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0];
        let z = [10.0, 10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1];
        let w = [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];

        let res3 = batch_mul_3_cols_f32(&x, &y, &z);
        for i in 0..8 {
            assert_eq!(res3[i], x[i] * y[i] * z[i]);
        }

        let res4 = batch_mul_4_cols_f32(&x, &y, &z, &w);
        for i in 0..8 {
            assert_eq!(res4[i], x[i] * y[i] * z[i] * w[i]);
        }
    }

    #[test]
    fn test_batch_mul_multi_cols_f64() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.5, 0.5, 2.0, 2.0];
        let z = [10.0, 10.0, 0.1, 0.1];
        let w = [2.0, 2.0, 1.0, 1.0];

        let res3 = batch_mul_3_cols_f64(&x, &y, &z);
        for i in 0..4 {
            assert_eq!(res3[i], x[i] * y[i] * z[i]);
        }

        let res4 = batch_mul_4_cols_f64(&x, &y, &z, &w);
        for i in 0..4 {
            assert_eq!(res4[i], x[i] * y[i] * z[i] * w[i]);
        }
    }

    #[test]
    fn test_batch_add_multi_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let z = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let w = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];

        let res3 = batch_add_3_cols_f32(&x, &y, &z);
        for i in 0..8 {
            assert_eq!(res3[i], x[i] + y[i] + z[i]);
        }

        let res4 = batch_add_4_cols_f32(&x, &y, &z, &w);
        for i in 0..8 {
            assert_eq!(res4[i], x[i] + y[i] + z[i] + w[i]);
        }
    }

    #[test]
    fn test_batch_add_multi_cols_f64() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.1, 0.2, 0.3, 0.4];
        let z = [10.0, 20.0, 30.0, 40.0];
        let w = [100.0, 200.0, 300.0, 400.0];

        let res3 = batch_add_3_cols_f64(&x, &y, &z);
        for i in 0..4 {
            assert_eq!(res3[i], x[i] + y[i] + z[i]);
        }

        let res4 = batch_add_4_cols_f64(&x, &y, &z, &w);
        for i in 0..4 {
            assert_eq!(res4[i], x[i] + y[i] + z[i] + w[i]);
        }
    }

    #[test]
    fn test_batch4_approx_inv_f32() {
        let input = [1.0, 2.0, 4.0, 8.0];
        let batch_res = batch4_approx_inv_f32(input);
        for i in 0..4 {
            let scalar_res = input[i].approx_inv();
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_fmadd_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let m = [0.5, 0.5, 2.0, 2.0];
        let a = [10.0, 10.0, 0.1, 0.1];
        let batch_res = batch4_fmadd_cols_f32(x, m, a);
        for i in 0..4 {
            let scalar_res = x[i].mul_add(m[i], a[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_fmadd_f32() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let m = 0.5;
        let a = 10.0;
        let batch_res = batch4_fmadd_f32(x, m, a);
        for i in 0..4 {
            let scalar_res = x[i].mul_add(m, a);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_asymmetric_fma_f32() {
        let x = [-1.0, 1.0, -2.0, 2.0];
        let mode = 10.0;
        let lo = 0.5;
        let hi = 2.0;
        let batch_res = batch4_asymmetric_fma_f32(x, mode, lo, hi);
        for i in 0..4 {
            let sigma = if x[i] < 0.0 { lo } else { hi };
            let scalar_res = x[i].mul_add(sigma, mode);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_asymmetric_fma_cols_f32() {
        let x = [-1.0, 1.0, -2.0, 2.0];
        let mode = [10.0, 20.0, 30.0, 40.0];
        let lo = [0.5, 0.6, 0.7, 0.8];
        let hi = [2.0, 3.0, 4.0, 5.0];
        let batch_res = batch4_asymmetric_fma_cols_f32(x, mode, lo, hi);
        for i in 0..4 {
            let sigma = if x[i] < 0.0 { lo[i] } else { hi[i] };
            let scalar_res = x[i].mul_add(sigma, mode[i]);
            assert_eq!(batch_res[i].to_bits(), scalar_res.to_bits());
        }
    }

    #[test]
    fn test_batch4_mul_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.5, 0.5, 2.0, 2.0];
        let batch_res = batch4_mul_cols_f32(x, y);
        for i in 0..4 {
            assert_eq!(batch_res[i].to_bits(), (x[i] * y[i]).to_bits());
        }
    }

    #[test]
    fn test_batch4_add_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.5, 0.5, 2.0, 2.0];
        let batch_res = batch4_add_cols_f32(x, y);
        for i in 0..4 {
            assert_eq!(batch_res[i].to_bits(), (x[i] + y[i]).to_bits());
        }
    }

    #[test]
    fn test_batch4_fma_cols_f32() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.5, 0.5, 2.0, 2.0];
        let z = [10.0, 10.0, 0.1, 0.1];
        let batch_res = batch4_fma_cols_f32(x, y, z);
        for i in 0..4 {
            assert_eq!(batch_res[i].to_bits(), x[i].mul_add(y[i], z[i]).to_bits());
        }
    }
}
