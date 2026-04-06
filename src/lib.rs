pub trait FastFloatFnHaver {
    fn fast_mul2(self) -> Self;
    fn fast_div2(self) -> Self;
    fn fast_mul3(self) -> Self;
    fn approx_exp(self) -> Self;
    fn approx_ln(self) -> Self;
}

impl FastFloatFnHaver for f32 {
    fn fast_mul2(self) -> Self {
        fast_mul2_f32(self)
    }

    fn fast_div2(self) -> Self {
        fast_div2_f32(self)
    }

    fn fast_mul3(self) -> Self {
        fast_mul3_f32(self)
    }

    fn approx_exp(self) -> Self {
        approx_exp_f32(self)
    }

    fn approx_ln(self) -> Self {
        approx_ln_f32(self)
    }
}

impl FastFloatFnHaver for f64 {
    fn fast_mul2(self) -> Self {
        fast_mul2_f64(self)
    }

    fn fast_div2(self) -> Self {
        fast_div2_f64(self)
    }

    fn fast_mul3(self) -> Self {
        fast_mul3_f64(self)
    }

    fn approx_exp(self) -> Self {
        approx_exp_f64(self)
    }

    fn approx_ln(self) -> Self {
        approx_ln_f64(self)
    }
}

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


    //We are pulling e from the const LOG2_E
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = std::f32::consts::LN_2;//0.69314575; 
    const LN2_LO: f32 = 0.0000014286068;
    const INV6: f32 = 1.0/6.0;

    //Range reduction x = n*ln2 + r
    let xv = (xltz & (-0.5_f32).to_bits()) | (xgeqz & (0.5_f32).to_bits());

    //mul_add will run like shit unless target-cpu=native is used
    //and the ISA has an FMA
    let n = x.mul_add(INV_LN2, f32::from_bits(xv)) as i32;
    //let r = x - (n as f32) * LN2_HI - (n as f32) * LN2_LO;
    let r = (-n as f32).mul_add(LN2_LO, (-n as f32).mul_add(LN2_HI, x));

    let is_good: u32 = !is_inf.wrapping_neg() & !is_z.wrapping_neg();

    //Approximate e^r on [-0.35, 0.35] using the Taylor series
    let exponent = (n + 127) as u32;
    //let res_r = ((INV6 * r + 0.5) * r + 1.0) * r + 1.0;
    let res_r = r.mul_add(r.mul_add(INV6.mul_add(r, 0.5), 1.0), 1.0);
    let two_n = f32::from_bits(exponent.wrapping_shl(23));

    let rv = two_n * res_r;
    f32::from_bits(is_inf & f32::INFINITY.to_bits() 
        | 0.0_f32.to_bits() & is_z 
        | rv.to_bits() & is_good)
}

#[inline(always)]
pub fn approx_exp_f64(x: f64) -> f64 {
    let xltz: u64 = ((x < 0.0) as u64).wrapping_neg();
    let xgeqz: u64 = ((x >= 0.0) as u64).wrapping_neg();
    let is_inf: u64 = ((x > 709.782712893384) as u64).wrapping_neg();
    let is_z: u64 = ((x < -708.3964185322641) as u64).wrapping_neg();

    // Range reduction uses log2(e) = 1/ln(2)
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = std::f64::consts::LN_2;
    const LN2_LO: f64 = 1.9082149292705877e-10;
    const INV6: f64 = 1.0/6.0;

    // Bias for range reduction: add/sub 0.5 depending on sign of x
    let xv = (xltz & (-0.5_f64).to_bits()) | (xgeqz & (0.5_f64).to_bits());
    let n = (x * INV_LN2 + f64::from_bits(xv)) as i32;

    //let r = x - (n as f64) * LN2_HI - (n as f64) * LN2_LO;
    //These mul_adds will run like shit unless -C target-cpu=native is used
    //and the CPU has a fused multiply add instruction!
    let r = (-n as f64).mul_add(LN2_LO, (-n as f64).mul_add(LN2_HI, x));

    let is_good: u64 = !is_inf.wrapping_neg() & !is_z.wrapping_neg();

    // Polynomial approximation for e^r on roughly [-0.35, 0.35]:
    //let res_r = ((INV6 * r + 0.5) * r + 1.0) * r + 1.0;
    let res_r = r.mul_add(r.mul_add(INV6.mul_add(r, 0.5), 1.0), 1.0);


    // Build 2^n via exponent bias (1023 for f64)
    let exponent = (n + 1023) as u64;
    let two_n = f64::from_bits(exponent.wrapping_shl(52));

    let rv = two_n * res_r;

    f64::from_bits(
        is_inf & f64::INFINITY.to_bits()
            | 0.0_f64.to_bits() & is_z
            | rv.to_bits() & is_good
    )
}

#[inline(always)]
pub fn approx_ln_f32(x: f32) -> f32 {
    const ONE_THIRD: f32 = 1.0/3.0;
    let bits = x.to_bits();
    let exponent = ((bits >> 23) as i32 - 127) as f32;
    
    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);

    // Linear approximation of ln(mantissa) for mantissa in [1, 2]
    let m_adj = mantissa - 1.0;
    //let ln_mantissa = m_adj * (1.0 - ONE_THIRD * m_adj); 
    let ln_mantissa = m_adj * (-ONE_THIRD).mul_add(m_adj, 1.0); 

    // ln(x) = ln(2^exp * mantissa) = exp * ln(2) + ln(mantissa)
    exponent.mul_add(std::f32::consts::LN_2, ln_mantissa)
}

#[inline(always)]
pub fn approx_ln_f64(x: f64) -> f64 {
    const ONE_THIRD: f64 = 1.0 / 3.0;
    let bits = x.to_bits();
    let exponent = ((bits >> 52) as i64 - 1023) as f64;
    let mantissa_bits = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    let mantissa = f64::from_bits(mantissa_bits);

    let m_adj = mantissa - 1.0;
    
    let ln_mantissa = m_adj * (-ONE_THIRD).mul_add(m_adj, 1.0);

    // ln(x) = exp * ln(2) + ln(mantissa)
    exponent.mul_add(std::f64::consts::LN_2, ln_mantissa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64() {
        const testvalues: [f64; 200] = {
            let mut table: [f64; 200] = [0.0; 200];
            let mut idx = 0;
            let mut val: f64 = 0.0;
            while idx < table.len() {
                table[idx] = val;
                val += 1.0;
                idx += 1;
            }
            table
        };
        for val in testvalues {
            let r = approx_exp_f64(val);
            let lnr = approx_ln_f64(r);
            println!("e^{} = {}; {}; ln(approx) = {}", val, val.exp(), r, lnr);
        }
    }
}
