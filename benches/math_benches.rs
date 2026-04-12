use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32");
    let val: f32 = 1.23456;
    let int_pow: i32 = 5;

    // Fast multiplication/division
    group.bench_function("mul2_std", |b| b.iter(|| black_box(val) * 2.0));
    group.bench_function("mul2_fast", |b| b.iter(|| black_box(val).fast_mul2()));

    group.bench_function("div2_std", |b| b.iter(|| black_box(val) / 2.0));
    group.bench_function("div2_fast", |b| b.iter(|| black_box(val).fast_div2()));

    group.bench_function("mul3_std", |b| b.iter(|| black_box(val) * 3.0));
    group.bench_function("mul3_fast", |b| b.iter(|| black_box(val).fast_mul3()));

    group.bench_function("mul4_std", |b| b.iter(|| black_box(val) * 4.0));
    group.bench_function("mul4_fast", |b| b.iter(|| black_box(val).fast_mul4()));

    group.bench_function("mul8_std", |b| b.iter(|| black_box(val) * 8.0));
    group.bench_function("mul8_fast", |b| b.iter(|| black_box(val).fast_mul8()));

    // Exponential/logarithmic
    group.bench_function("exp_std", |b| b.iter(|| black_box(val).exp()));
    group.bench_function("exp_fast", |b| b.iter(|| black_box(val).approx_exp()));

    group.bench_function("ln_std", |b| b.iter(|| black_box(val).ln()));
    group.bench_function("ln_fast", |b| b.iter(|| black_box(val).approx_ln()));

    // Roots
    group.bench_function("sqrt_std", |b| b.iter(|| black_box(val).sqrt()));
    group.bench_function("sqrt_fast", |b| b.iter(|| black_box(val).approx_sqrt()));

    group.bench_function("cbrt_std", |b| b.iter(|| black_box(val).cbrt()));
    group.bench_function("cbrt_fast", |b| b.iter(|| black_box(val).approx_cbrt()));

    // Trigonometry
    group.bench_function("sin_std", |b| b.iter(|| black_box(val).sin()));
    group.bench_function("sin_fast", |b| b.iter(|| black_box(val).approx_sin()));

    group.bench_function("cos_std", |b| b.iter(|| black_box(val).cos()));
    group.bench_function("cos_fast", |b| b.iter(|| black_box(val).approx_cos()));

    group.bench_function("sin_cos_std", |b| b.iter(|| black_box(val).sin_cos()));
    group.bench_function("sin_cos_fast", |b| {
        b.iter(|| black_box(val).approx_sin_cos())
    });
    group.bench_function("sin_cos_fast_sep", |b| {
        b.iter(|| {
            let v = black_box(val);
            (v.approx_sin(), v.approx_cos())
        })
    });

    let trig_val: f32 = 0.5;
    group.bench_function("acos_std", |b| b.iter(|| black_box(trig_val).acos()));
    group.bench_function("acos_fast", |b| {
        b.iter(|| black_box(trig_val).approx_acos())
    });

    group.bench_function("asin_std", |b| b.iter(|| black_box(trig_val).asin()));
    group.bench_function("asin_fast", |b| {
        b.iter(|| black_box(trig_val).approx_asin())
    });

    // Inverse
    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    // Power functions
    group.bench_function("powi_std", |b| {
        b.iter(|| black_box(val).powi(black_box(int_pow)))
    });
    group.bench_function("powi_fast", |b| {
        b.iter(|| black_box(val).approx_powi(black_box(int_pow)))
    });

    let float_pow: f32 = 2.5;
    group.bench_function("powf_std", |b| {
        b.iter(|| black_box(val).powf(black_box(float_pow)))
    });
    group.bench_function("powf_fast", |b| {
        b.iter(|| black_box(val).approx_powf(black_box(float_pow)))
    });

    group.finish();
}

fn bench_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64");
    let val: f64 = 1.23456;
    let int_pow: i32 = 5;

    // Fast multiplication/division
    group.bench_function("mul2_std", |b| b.iter(|| black_box(val) * 2.0));
    group.bench_function("mul2_fast", |b| b.iter(|| black_box(val).fast_mul2()));

    group.bench_function("div2_std", |b| b.iter(|| black_box(val) / 2.0));
    group.bench_function("div2_fast", |b| b.iter(|| black_box(val).fast_div2()));

    group.bench_function("mul3_std", |b| b.iter(|| black_box(val) * 3.0));
    group.bench_function("mul3_fast", |b| b.iter(|| black_box(val).fast_mul3()));

    group.bench_function("mul4_std", |b| b.iter(|| black_box(val) * 4.0));
    group.bench_function("mul4_fast", |b| b.iter(|| black_box(val).fast_mul4()));

    group.bench_function("mul8_std", |b| b.iter(|| black_box(val) * 8.0));
    group.bench_function("mul8_fast", |b| b.iter(|| black_box(val).fast_mul8()));

    // Exponential/logarithmic
    group.bench_function("exp_std", |b| b.iter(|| black_box(val).exp()));
    group.bench_function("exp_fast", |b| b.iter(|| black_box(val).approx_exp()));
    group.bench_function("exp_fastb", |b| b.iter(|| approx_exp_f64b(black_box(val))));

    group.bench_function("ln_std", |b| b.iter(|| black_box(val).ln()));
    group.bench_function("ln_fast", |b| b.iter(|| black_box(val).approx_ln()));

    // Roots
    group.bench_function("sqrt_std", |b| b.iter(|| black_box(val).sqrt()));
    group.bench_function("sqrt_fast", |b| b.iter(|| black_box(val).approx_sqrt()));

    group.bench_function("cbrt_std", |b| b.iter(|| black_box(val).cbrt()));
    group.bench_function("cbrt_fast", |b| b.iter(|| black_box(val).approx_cbrt()));

    // Trigonometry
    group.bench_function("sin_std", |b| b.iter(|| black_box(val).sin()));
    group.bench_function("sin_fast", |b| b.iter(|| black_box(val).approx_sin()));

    group.bench_function("cos_std", |b| b.iter(|| black_box(val).cos()));
    group.bench_function("cos_fast", |b| b.iter(|| black_box(val).approx_cos()));

    group.bench_function("sin_cos_std", |b| b.iter(|| black_box(val).sin_cos()));
    group.bench_function("sin_cos_fast", |b| {
        b.iter(|| black_box(val).approx_sin_cos())
    });
    group.bench_function("sin_cos_fast_sep", |b| {
        b.iter(|| {
            let v = black_box(val);
            (v.approx_sin(), v.approx_cos())
        })
    });

    let trig_val: f64 = 0.5;
    group.bench_function("acos_std", |b| b.iter(|| black_box(trig_val).acos()));
    group.bench_function("acos_fast", |b| {
        b.iter(|| black_box(trig_val).approx_acos())
    });

    group.bench_function("asin_std", |b| b.iter(|| black_box(trig_val).asin()));
    group.bench_function("asin_fast", |b| {
        b.iter(|| black_box(trig_val).approx_asin())
    });

    // Inverse
    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    // Power functions
    group.bench_function("powi_std", |b| {
        b.iter(|| black_box(val).powi(black_box(int_pow)))
    });
    group.bench_function("powi_fast", |b| {
        b.iter(|| black_box(val).approx_powi(black_box(int_pow)))
    });

    let float_pow: f64 = 2.5;
    group.bench_function("powf_std", |b| {
        b.iter(|| black_box(val).powf(black_box(float_pow)))
    });
    group.bench_function("powf_fast", |b| {
        b.iter(|| black_box(val).approx_powf(black_box(float_pow)))
    });

    group.finish();
}

fn bench_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f32");
    let x = [1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2]; // Mixed signs
    let y = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
    let z = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let n = [5i32; 8];

    group.bench_function("ln_batch", |b| b.iter(|| batch_approx_ln_f32(black_box(x))));
    group.bench_function("ln_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = val[i].approx_ln();
            }
            out
        })
    });

    group.bench_function("inv_batch", |b| {
        b.iter(|| batch_approx_inv_f32(black_box(x)))
    });
    group.bench_function("inv_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = val[i].approx_inv();
            }
            out
        })
    });

    group.bench_function("exp_batch", |b| {
        b.iter(|| batch_approx_exp_f32(black_box(x)))
    });
    group.bench_function("exp_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = val[i].approx_exp();
            }
            out
        })
    });

    group.bench_function("sqrt_batch", |b| {
        b.iter(|| batch_approx_sqrt_f32(black_box(x)))
    });
    group.bench_function("sqrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = val[i].approx_sqrt();
            }
            out
        })
    });

    group.bench_function("cbrt_batch", |b| {
        b.iter(|| batch_approx_cbrt_f32(black_box(x)))
    });
    group.bench_function("cbrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = val[i].approx_cbrt();
            }
            out
        })
    });

    group.bench_function("sin_cos_batch", |b| {
        b.iter(|| batch_approx_sin_cos_f32(black_box(x)))
    });
    group.bench_function("sin_cos_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out_s = [0.0; 8];
            let mut out_c = [0.0; 8];
            for i in 0..8 {
                let (s, c) = val[i].approx_sin_cos();
                out_s[i] = s;
                out_c[i] = c;
            }
            (out_s, out_c)
        })
    });

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| chunk_approx_powf_cols_f32(black_box(x), black_box(y)))
    });
    group.bench_function("powf_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = vx[i].approx_powf(vy[i]);
            }
            out
        })
    });

    group.bench_function("powi_cols_batch", |b| {
        b.iter(|| batch_approx_powi_cols_f32(black_box(x), black_box(n)))
    });
    group.bench_function("powi_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vn = black_box(n);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = vx[i].approx_powi(vn[i]);
            }
            out
        })
    });

    group.bench_function("fmadd_cols_batch", |b| {
        b.iter(|| batch_fmadd_cols_f32(black_box(x), black_box(y), black_box(z)))
    });
    group.bench_function("fmadd_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let vz = black_box(z);
            let mut out = [0.0; 8];
            for i in 0..8 {
                out[i] = vx[i].mul_add(vy[i], vz[i]);
            }
            out
        })
    });

    group.bench_function("asymmetric_fma_cols_batch", |b| {
        b.iter(|| {
            batch_asymmetric_fma_cols_f32(black_box(x), black_box(z), black_box(y), black_box(z))
        })
    });
    group.bench_function("asymmetric_fma_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vz = black_box(z);
            let v_lo = black_box(y);
            let v_hi = black_box(z);
            let mut out = [0.0; 8];
            for i in 0..8 {
                let sigma = if vx[i] < 0.0 { v_lo[i] } else { v_hi[i] };
                out[i] = vx[i].mul_add(sigma, vz[i]);
            }
            out
        })
    });

    group.finish();
}

fn bench_batch_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f64");
    let x = [1.2, -1.2, 1.2, -1.2];
    let y = [2.5, 2.5, 2.5, 2.5];
    let z = [0.5, 0.5, 0.5, 0.5];
    let n = [5i32; 4];

    group.bench_function("ln_batch", |b| b.iter(|| batch_approx_ln_f64(black_box(x))));
    group.bench_function("ln_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = val[i].approx_ln();
            }
            out
        })
    });

    group.bench_function("inv_batch", |b| {
        b.iter(|| batch_approx_inv_f64(black_box(x)))
    });
    group.bench_function("inv_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = val[i].approx_inv();
            }
            out
        })
    });

    group.bench_function("exp_batch", |b| {
        b.iter(|| batch_approx_exp_f64(black_box(x)))
    });
    group.bench_function("exp_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = val[i].approx_exp();
            }
            out
        })
    });

    group.bench_function("sqrt_batch", |b| {
        b.iter(|| batch_approx_sqrt_f64(black_box(x)))
    });
    group.bench_function("sqrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = val[i].approx_sqrt();
            }
            out
        })
    });

    group.bench_function("cbrt_batch", |b| {
        b.iter(|| batch_approx_cbrt_f64(black_box(x)))
    });
    group.bench_function("cbrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = val[i].approx_cbrt();
            }
            out
        })
    });

    group.bench_function("sin_cos_batch", |b| {
        b.iter(|| batch_approx_sin_cos_f64(black_box(x)))
    });
    group.bench_function("sin_cos_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out_s = [0.0; 4];
            let mut out_c = [0.0; 4];
            for i in 0..4 {
                let (s, c) = val[i].approx_sin_cos();
                out_s[i] = s;
                out_c[i] = c;
            }
            (out_s, out_c)
        })
    });

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| chunk_approx_powf_cols_f64(black_box(x), black_box(y)))
    });
    group.bench_function("powf_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = vx[i].approx_powf(vy[i]);
            }
            out
        })
    });

    group.bench_function("powi_cols_batch", |b| {
        b.iter(|| batch_approx_powi_cols_f64(black_box(x), black_box(n)))
    });
    group.bench_function("powi_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vn = black_box(n);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = vx[i].approx_powi(vn[i]);
            }
            out
        })
    });

    group.bench_function("fmadd_cols_batch", |b| {
        b.iter(|| batch_fmadd_cols_f64(black_box(x), black_box(y), black_box(z)))
    });
    group.bench_function("fmadd_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let vz = black_box(z);
            let mut out = [0.0; 4];
            for i in 0..4 {
                out[i] = vx[i].mul_add(vy[i], vz[i]);
            }
            out
        })
    });

    group.bench_function("asymmetric_fma_cols_batch", |b| {
        b.iter(|| {
            batch_asymmetric_fma_cols_f64(black_box(x), black_box(z), black_box(y), black_box(z))
        })
    });
    group.bench_function("asymmetric_fma_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vz = black_box(z);
            let v_lo = black_box(y);
            let v_hi = black_box(z);
            let mut out = [0.0; 4];
            for i in 0..4 {
                let sigma = if vx[i] < 0.0 { v_lo[i] } else { v_hi[i] };
                out[i] = vx[i].mul_add(sigma, vz[i]);
            }
            out
        })
    });
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let data_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let data_f64: Vec<f64> = (0..1024).map(|i| i as f64 * 0.001).collect();
    let data_i32: Vec<i32> = (0..1024).map(|i| i as i32).collect();
    let data_i64: Vec<i64> = (0..1024).map(|i| i as i64).collect();

    let mut group = c.benchmark_group("sum");

    group.bench_function("f32_batch", |b| {
        b.iter(|| batch_sum_f32(black_box(&data_f32)))
    });
    group.bench_function("f32_scalar", |b| {
        b.iter(|| black_box(&data_f32).iter().sum::<f32>())
    });

    group.bench_function("f64_batch", |b| {
        b.iter(|| batch_sum_f64(black_box(&data_f64)))
    });
    group.bench_function("f64_scalar", |b| {
        b.iter(|| black_box(&data_f64).iter().sum::<f64>())
    });

    group.bench_function("i32_batch", |b| {
        b.iter(|| batch_sum_i32(black_box(&data_i32)))
    });
    group.bench_function("i32_scalar", |b| {
        b.iter(|| {
            black_box(&data_i32)
                .iter()
                .fold(0i32, |acc, &x| acc.wrapping_add(x))
        })
    });

    group.bench_function("i64_batch", |b| {
        b.iter(|| batch_sum_i64(black_box(&data_i64)))
    });
    group.bench_function("i64_scalar", |b| {
        b.iter(|| black_box(&data_i64).iter().sum::<i64>())
    });

    group.finish();
}

fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");
    let vec_x_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 1.001).collect();
    let vec_y_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 0.999).collect();
    let vec_z_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let vec_x_f64: Vec<f64> = (0..1024).map(|i| i as f64 * 1.001).collect();
    let vec_y_f64: Vec<f64> = (0..1024).map(|i| i as f64 * 0.999).collect();
    let vec_z_f64: Vec<f64> = (0..1024).map(|i| i as f64 * 0.5).collect();

    let x_f32: &[f32; 1024] = vec_x_f32.as_slice().try_into().unwrap();
    let y_f32: &[f32; 1024] = vec_y_f32.as_slice().try_into().unwrap();
    let z_f32: &[f32; 1024] = vec_z_f32.as_slice().try_into().unwrap();
    let x_f64: &[f64; 1024] = vec_x_f64.as_slice().try_into().unwrap();
    let y_f64: &[f64; 1024] = vec_y_f64.as_slice().try_into().unwrap();
    let z_f64: &[f64; 1024] = vec_z_f64.as_slice().try_into().unwrap();

    // f32 Mul
    group.bench_function("f32_mul_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            batch_mul_cols_f32(black_box(x_f32), black_box(y_f32), &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f32_mul_scalar", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xb = black_box(x_f32);
            let yb = black_box(y_f32);
            for i in 0..1024 { out[i] = xb[i] * yb[i]; }
            black_box(out[0])
        })
    });

    // f32 Add
    group.bench_function("f32_add_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            batch_add_cols_f32(black_box(x_f32), black_box(y_f32), &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f32_add_scalar", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xb = black_box(x_f32);
            let yb = black_box(y_f32);
            for i in 0..1024 { out[i] = xb[i] + yb[i]; }
            black_box(out[0])
        })
    });

    // f32 FMA
    group.bench_function("f32_fma_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            batch_fma_cols_f32(black_box(x_f32), black_box(y_f32), black_box(z_f32), &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f32_fma_scalar", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xb = black_box(x_f32);
            let yb = black_box(y_f32);
            let zb = black_box(z_f32);
            for i in 0..1024 { out[i] = xb[i].mul_add(yb[i], zb[i]); }
            black_box(out[0])
        })
    });

    // f64 Mul
    group.bench_function("f64_mul_batch", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            batch_mul_cols_f64(black_box(x_f64), black_box(y_f64), &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f64_mul_scalar", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xb = black_box(x_f64);
            let yb = black_box(y_f64);
            for i in 0..1024 { out[i] = xb[i] * yb[i]; }
            black_box(out[0])
        })
    });

    // f64 Add
    group.bench_function("f64_add_batch", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            batch_add_cols_f64(black_box(x_f64), black_box(y_f64), &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f64_add_scalar", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xb = black_box(x_f64);
            let yb = black_box(y_f64);
            for i in 0..1024 { out[i] = xb[i] + yb[i]; }
            black_box(out[0])
        })
    });

    // f64 FMA
    group.bench_function("f64_fma_batch", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            batch_fma_cols_f64(black_box(x_f64), black_box(y_f64), black_box(z_f64), &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f64_fma_scalar", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xb = black_box(x_f64);
            let yb = black_box(y_f64);
            let zb = black_box(z_f64);
            for i in 0..1024 { out[i] = xb[i].mul_add(yb[i], zb[i]); }
            black_box(out[0])
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_f32,
    bench_f64,
    bench_batch_f32,
    bench_batch_f64,
    bench_sum,
    bench_arithmetic
);
criterion_main!(benches);
