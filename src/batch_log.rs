use crate::FastFloatFnHaver;
use core::mem::MaybeUninit;
use crate::batch_arith::{batch_approx_inv_f32, batch_approx_inv_f64, batch4_approx_inv_f32};
use core::arch::x86_64::*;

// Local raw_batch_ln removed, replaced by crate-level raw_batch_log

#[inline(always)]
pub fn batch_approx_ln_f32(x: [f32; 8]) -> [f32; 8] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_approx_ln_f32(v);
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
        let v: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_approx_ln_f32(v);
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
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_approx_sqrt_f32(v_x);
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
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_approx_sqrt_f32(v_x);
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
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_approx_exp_f32(v_x);
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
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_approx_exp_f32(v_x);
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
        let v_y: core::arch::x86_64::__m256 = core::mem::transmute(y);
        let res = crate::raw_batch_approx_powf_f32(x, v_y);
        core::mem::transmute(res)
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
        let v_y: core::arch::x86_64::__m128 = core::mem::transmute(y);
        let res = crate::raw_batch4_approx_powf_f32(x, v_y);
        core::mem::transmute(res)
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

// Local raw_batch_ln_f64 removed, replaced by crate-level raw_batch_log

#[inline(always)]
pub fn batch_approx_ln_f64(x: [f64; 4]) -> [f64; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_approx_ln_f64(v);
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
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_approx_sqrt_f64(v_x);
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
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_approx_exp_f64(v_x);
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
        let v_y: core::arch::x86_64::__m256d = core::mem::transmute(y);
        let res = crate::raw_batch_approx_powf_f64(x, v_y);
        core::mem::transmute(res)
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
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let res = crate::raw_batch_approx_cbrt_f32(v_x);
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
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let res = crate::raw_batch4_approx_cbrt_f32(v_x);
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
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let v_y: core::arch::x86_64::__m256 = core::mem::transmute(y);
        let res = crate::raw_chunk_approx_powf_cols_f32(v_x, v_y);
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
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let v_y: core::arch::x86_64::__m128 = core::mem::transmute(y);
        let res = crate::raw_chunk4_approx_powf_cols_f32(v_x, v_y);
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
        let v_base: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let v_n: core::arch::x86_64::__m256i = core::mem::transmute(n);
        let res = crate::raw_batch_approx_powi_cols_f32(v_base, v_n);
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
        let v_base: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let v_n: core::arch::x86_64::__m128i = core::mem::transmute(n);
        let res = crate::raw_batch4_approx_powi_cols_f32(v_base, v_n);
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
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let res = crate::raw_batch_approx_cbrt_f64(v_x);
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
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let v_y: core::arch::x86_64::__m256d = core::mem::transmute(y);
        let res = crate::raw_chunk_approx_powf_cols_f64(v_x, v_y);
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
        let v_base: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let v_n: core::arch::x86_64::__m128i = core::mem::transmute(n);
        let res = crate::raw_batch_approx_powi_cols_f64(v_base, v_n);
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
