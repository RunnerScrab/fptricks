#[inline(always)]
pub fn fast_mul2_f32(x: f32) -> f32 {
    const ONEEXP: u32 = 1<<23;
    f32::from_bits(x.to_bits().saturating_add(ONEEXP))
}

#[inline(always)]
pub fn fast_div2_f32(x: f32) -> f32 {
    const ONEEXP: u32 = 1<<23;
    f32::from_bits(x.to_bits().saturating_sub(ONEEXP))
}

#[inline(always)]
pub fn fast_mul2_f64(x: f64) -> f64 {
    const ONEEXP: u64 = 1<<52;
    f64::from_bits(x.to_bits().saturating_add(ONEEXP))
}

#[inline(always)]
pub fn fast_div2_f64(x: f64) -> f64 {
    const ONEEXP: u64 = 1<<52;
    f64::from_bits(x.to_bits().saturating_sub(ONEEXP))
}

#[inline(always)]
pub fn fast_mul3_f64(x: f64) -> f64 {
   fast_mul2_f64(x) + x 
}

#[inline(always)]
pub fn fast_mul3_f32(x: f32) -> f32 {
   fast_mul2_f32(x) + x 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
    }
}
