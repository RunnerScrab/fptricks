#![allow(unused)]
mod arithmetic;
mod logarithmic;
mod trigonometric;
mod batch;

pub use batch::*;
pub use arithmetic::*;
pub use logarithmic::*;
pub use trigonometric::*;

pub trait FastFloatFnHaver: Sized {
    fn fast_mul2(self) -> Self;
    fn fast_div2(self) -> Self;
    fn fast_mul3(self) -> Self;
    fn fast_mul4(self) -> Self;
    fn fast_mul8(self) -> Self;
    fn approx_exp(self) -> Self;
    fn approx_ln(self) -> Self;
    ///Faster only for f64
    fn approx_sqrt(self) -> Self;
    fn approx_cbrt(self) -> Self;
    fn approx_sin(self) -> Self;
    fn approx_cos(self) -> Self;
    fn approx_sin_cos(self) -> (Self, Self);
    fn approx_acos(self) -> Self;
    fn approx_asin(self) -> Self;
    fn approx_inv(self) -> Self;
    ///Worse in all cases
    fn approx_powi(self, n: i32) -> Self;
    ///Faster only for f64
    fn approx_powf(self, y: Self) -> Self;
    fn approx_atan2(self, x: Self) -> Self;
}

impl FastFloatFnHaver for f32 {
    #[inline(always)]
    fn fast_mul2(self) -> Self {
        arithmetic::fast_mul2_f32(self)
    }

    #[inline(always)]
    fn fast_div2(self) -> Self {
        arithmetic::fast_div2_f32(self)
    }

    #[inline(always)]
    fn fast_mul3(self) -> Self {
        arithmetic::fast_mul3_f32(self)
    }

    #[inline(always)]
    fn fast_mul4(self) -> Self {
        arithmetic::fast_mul4_f32(self)
    }

    #[inline(always)]
    fn fast_mul8(self) -> Self {
        arithmetic::fast_mul8_f32(self)
    }

    #[inline(always)]
    fn approx_exp(self) -> Self {
        // self.exp() was considered faster in some cases, but we keep the option
        logarithmic::approx_exp_f32(self)
    }

    #[inline(always)]
    fn approx_ln(self) -> Self {
        logarithmic::approx_ln_f32(self)
    }

    #[inline(always)]
    fn approx_sqrt(self) -> Self {
        logarithmic::approx_sqrt_f32(self)
    }

    #[inline(always)]
    fn approx_cbrt(self) -> Self {
        logarithmic::approx_cbrt_f32(self)
    }

    #[inline(always)]
    fn approx_sin(self) -> Self {
        trigonometric::approx_sin_f32(self)
    }

    #[inline(always)]
    fn approx_cos(self) -> Self {
        trigonometric::approx_cos_f32(self)
    }

    #[inline(always)]
    fn approx_sin_cos(self) -> (Self, Self) {
        trigonometric::approx_sin_cos_f32(self)
    }

    #[inline(always)]
    fn approx_acos(self) -> Self {
        trigonometric::approx_acos_f32(self)
    }

    #[inline(always)]
    fn approx_asin(self) -> Self {
        trigonometric::approx_asin_f32(self)
    }

    #[inline(always)]
    fn approx_inv(self) -> Self {
        arithmetic::approx_inv_f32(self)
    }

    #[inline(always)]
    fn approx_powi(self, n: i32) -> Self {
        logarithmic::approx_powi_f32(self, n)
    }

    #[inline(always)]
    fn approx_powf(self, y: Self) -> Self {
        logarithmic::approx_powf_f32(self, y)
    }

    #[inline(always)]
    fn approx_atan2(self, x: Self) -> Self {
        trigonometric::approx_atan2_f32(self, x)
    }
}

impl FastFloatFnHaver for f64 {
    #[inline(always)]
    fn fast_mul2(self) -> Self {
        arithmetic::fast_mul2_f64(self)
    }

    #[inline(always)]
    fn fast_div2(self) -> Self {
        arithmetic::fast_div2_f64(self)
    }

    #[inline(always)]
    fn fast_mul3(self) -> Self {
        arithmetic::fast_mul3_f64(self)
    }

    #[inline(always)]
    fn fast_mul4(self) -> Self {
        arithmetic::fast_mul4_f64(self)
    }

    #[inline(always)]
    fn fast_mul8(self) -> Self {
        arithmetic::fast_mul8_f64(self)
    }

    #[inline(always)]
    fn approx_exp(self) -> Self {
        logarithmic::approx_exp_f64(self)
    }

    #[inline(always)]
    fn approx_ln(self) -> Self {
        logarithmic::approx_ln_f64(self)
    }

    #[inline(always)]
    fn approx_sqrt(self) -> Self {
        logarithmic::approx_sqrt_f64(self)
    }

    #[inline(always)]
    fn approx_cbrt(self) -> Self {
        logarithmic::approx_cbrt_f64(self)
    }

    #[inline(always)]
    fn approx_sin(self) -> Self {
        trigonometric::approx_sin_f64(self)
    }

    #[inline(always)]
    fn approx_cos(self) -> Self {
        trigonometric::approx_cos_f64(self)
    }

    #[inline(always)]
    fn approx_sin_cos(self) -> (Self, Self) {
        trigonometric::approx_sin_cos_f64(self)
    }

    #[inline(always)]
    fn approx_acos(self) -> Self {
        trigonometric::approx_acos_f64(self)
    }

    #[inline(always)]
    fn approx_asin(self) -> Self {
        trigonometric::approx_asin_f64(self)
    }

    #[inline(always)]
    fn approx_inv(self) -> Self {
        arithmetic::approx_inv_f64(self)
    }

    #[inline(always)]
    fn approx_powi(self, n: i32) -> Self {
        logarithmic::approx_powi_f64(self, n)
    }

    #[inline(always)]
    fn approx_powf(self, y: Self) -> Self {
        logarithmic::approx_powf_f64(self, y)
    }

    #[inline(always)]
    fn approx_atan2(self, x: Self) -> Self {
        trigonometric::approx_atan2_f64(self, x)
    }
}
