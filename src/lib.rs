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

#[inline(always)]
///Good to 3-6 significant digits
pub fn approx_exp_f32(x: f32) -> f32 {
    let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
    let xgeqz: u32 = ((x >= 0.0) as u32).wrapping_neg();
    let is_inf: u32 = ((x > 88.72283) as u32).wrapping_neg();
    let is_z: u32 = ((x < -87.33654) as u32).wrapping_neg();

    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.69314575; 
    const LN2_LO: f32 = 0.0000014286068;

    //Range reduction x = n*ln2 + r
    let xv = (xltz & (-0.5_f32).to_bits()) | (xgeqz & (0.5_f32).to_bits());
    let n = (x * INV_LN2 + (f32::from_bits(xv))) as i32;
    let r = x - (n as f32) * LN2_HI - (n as f32) * LN2_LO;

    let is_good: u32 = !is_inf.wrapping_neg() & !is_z.wrapping_neg();
    //Approximate e^r on [-0.35, 0.35]
    let exponent = (n + 127) as u32;
    let res_r = ((0.16666667 * r + 0.5) * r + 1.0) * r + 1.0;

    let two_n = f32::from_bits(exponent.wrapping_shl(23));

    let rv = two_n * res_r;
    f32::from_bits(is_inf & f32::INFINITY.to_bits() 
        | 0.0_f32.to_bits() & is_z 
        | rv.to_bits() & is_good)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
    }
}
