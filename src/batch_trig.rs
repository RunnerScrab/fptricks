use crate::FastFloatFnHaver;

#[inline(always)]
pub fn batch_approx_sin_cos_f32(x: [f32; 8]) -> ([f32; 8], [f32; 8]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256 = core::mem::transmute(x);
        let (s, c) = crate::raw_batch_approx_sin_cos_f32(v_x);
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
pub fn batch4_approx_sin_cos_f32(x: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m128 = core::mem::transmute(x);
        let (s, c) = crate::raw_batch4_approx_sin_cos_f32(v_x);
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
pub fn batch_approx_sin_cos_f64(x: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        let v_x: core::arch::x86_64::__m256d = core::mem::transmute(x);
        let (s, c) = crate::raw_batch_approx_sin_cos_f64(v_x);
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

    #[test]
    fn test_batch4_approx_sin_cos_f32() {
        let input = [0.0, 1.0, 3.14159, -10.0];
        let (batch_sin, batch_cos) = batch4_approx_sin_cos_f32(input);
        for i in 0..4 {
            let (scalar_sin, scalar_cos) = crate::approx_sin_cos_f32(input[i]);
            assert_eq!(batch_sin[i].to_bits(), scalar_sin.to_bits());
            assert_eq!(batch_cos[i].to_bits(), scalar_cos.to_bits());
        }
    }
}
