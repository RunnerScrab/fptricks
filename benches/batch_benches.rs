use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f32");
    let x = [1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2];
    let y = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
    let z = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let n = [5i32; 8];

    group.bench_function("ln_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_ln_f32(val);
            out
        })
    });
    group.bench_function("ln_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = val[i].approx_ln(); }
            out
        })
    });

    group.bench_function("inv_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_inv_f32(val);
            out
        })
    });
    group.bench_function("inv_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = val[i].approx_inv(); }
            out
        })
    });

    group.bench_function("exp_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_exp_f32(val);
            out
        })
    });
    group.bench_function("exp_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = val[i].approx_exp(); }
            out
        })
    });

    group.bench_function("sqrt_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_sqrt_f32(val);
            out
        })
    });
    group.bench_function("sqrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = val[i].approx_sqrt(); }
            out
        })
    });

    group.bench_function("cbrt_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_cbrt_f32(val);
            out
        })
    });
    group.bench_function("cbrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = val[i].approx_cbrt(); }
            out
        })
    });

    group.bench_function("sin_cos_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_sin_cos_f32(val);
            out
        })
    });
    group.bench_function("sin_cos_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out_s = [0.0f32; 8];
            let mut out_c = [0.0f32; 8];
            for i in 0..8 {
                let (s, c) = val[i].approx_sin_cos();
                out_s[i] = s;
                out_c[i] = c;
            }
            (out_s, out_c)
        })
    });

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let out = chunk_approx_powf_cols_f32(vx, vy);
            out
        })
    });
    group.bench_function("powf_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = vx[i].approx_powf(vy[i]); }
            out
        })
    });

    group.bench_function("powi_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vn = black_box(n);
            let out = batch_approx_powi_cols_f32(vx, vn);
            out
        })
    });
    group.bench_function("powi_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vn = black_box(n);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = vx[i].approx_powi(vn[i]); }
            out
        })
    });

    group.bench_function("fmadd_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let vz = black_box(z);
            let out = batch_fmadd_cols_f32(vx, vy, vz);
            out
        })
    });
    group.bench_function("fmadd_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let vz = black_box(z);
            let mut out = [0.0f32; 8];
            for i in 0..8 { out[i] = vx[i].mul_add(vy[i], vz[i]); }
            out
        })
    });

    group.bench_function("asymmetric_fma_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vm = black_box(z);
            let v_lo = black_box(y);
            let v_hi = black_box(z);
            let out = batch_asymmetric_fma_cols_f32(vx, vm, v_lo, v_hi);
            out
        })
    });
    group.bench_function("asymmetric_fma_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vz = black_box(z);
            let v_lo = black_box(y);
            let v_hi = black_box(z);
            let mut out = [0.0f32; 8];
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

    group.bench_function("ln_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_ln_f64(val);
            out
        })
    });
    group.bench_function("ln_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = val[i].approx_ln(); }
            out
        })
    });

    group.bench_function("inv_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_inv_f64(val);
            out
        })
    });
    group.bench_function("inv_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = val[i].approx_inv(); }
            out
        })
    });

    group.bench_function("exp_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_exp_f64(val);
            out
        })
    });
    group.bench_function("exp_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = val[i].approx_exp(); }
            out
        })
    });

    group.bench_function("sqrt_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_sqrt_f64(val);
            out
        })
    });
    group.bench_function("sqrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = val[i].approx_sqrt(); }
            out
        })
    });

    group.bench_function("cbrt_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_cbrt_f64(val);
            out
        })
    });
    group.bench_function("cbrt_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = val[i].approx_cbrt(); }
            out
        })
    });

    group.bench_function("sin_cos_batch", |b| {
        b.iter(|| {
            let val = black_box(x);
            let out = batch_approx_sin_cos_f64(val);
            out
        })
    });
    group.bench_function("sin_cos_scalar_loop", |b| {
        b.iter(|| {
            let val = black_box(x);
            let mut out_s = [0.0f64; 4];
            let mut out_c = [0.0f64; 4];
            for i in 0..4 {
                let (s, c) = val[i].approx_sin_cos();
                out_s[i] = s;
                out_c[i] = c;
            }
            (out_s, out_c)
        })
    });

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let out = chunk_approx_powf_cols_f64(vx, vy);
            out
        })
    });
    group.bench_function("powf_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = vx[i].approx_powf(vy[i]); }
            out
        })
    });

    group.bench_function("powi_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vn = black_box(n);
            let out = batch_approx_powi_cols_f64(vx, vn);
            out
        })
    });
    group.bench_function("powi_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vn = black_box(n);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = vx[i].approx_powi(vn[i]); }
            out
        })
    });

    group.bench_function("fmadd_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let vz = black_box(z);
            let out = batch_fmadd_cols_f64(vx, vy, vz);
            out
        })
    });
    group.bench_function("fmadd_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vy = black_box(y);
            let vz = black_box(z);
            let mut out = [0.0f64; 4];
            for i in 0..4 { out[i] = vx[i].mul_add(vy[i], vz[i]); }
            out
        })
    });

    group.bench_function("asymmetric_fma_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vm = black_box(z);
            let v_lo = black_box(y);
            let v_hi = black_box(z);
            let out = batch_asymmetric_fma_cols_f64(vx, vm, v_lo, v_hi);
            out
        })
    });
    group.bench_function("asymmetric_fma_cols_scalar_loop", |b| {
        b.iter(|| {
            let vx = black_box(x);
            let vz = black_box(z);
            let v_lo = black_box(y);
            let v_hi = black_box(z);
            let mut out = [0.0f64; 4];
            for i in 0..4 {
                let sigma = if vx[i] < 0.0 { v_lo[i] } else { v_hi[i] };
                out[i] = vx[i].mul_add(sigma, vz[i]);
            }
            out
        })
    });

    group.finish();
}

fn bench_sums(c: &mut Criterion) {
    let data_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let data_f64: Vec<f64> = (0..1024).map(|i| i as f64 * 0.001).collect();
    let data_i32: Vec<i32> = (0..1024).map(|i| i as i32).collect();
    let data_i64: Vec<i64> = (0..1024).map(|i| i as i64).collect();

    let mut group = c.benchmark_group("sum");
    group.bench_function("f32_batch", |b| {
        b.iter(|| {
            let input = black_box(&data_f32);
            batch_sum_f32(input)
        })
    });
    group.bench_function("f32_scalar", |b| {
        b.iter(|| {
            let input = black_box(&data_f32);
            input.iter().sum::<f32>()
        })
    });

    group.bench_function("f64_batch", |b| {
        b.iter(|| {
            let input = black_box(&data_f64);
            batch_sum_f64(input)
        })
    });
    group.bench_function("f64_scalar", |b| {
        b.iter(|| {
            let input = black_box(&data_f64);
            input.iter().sum::<f64>()
        })
    });

    group.bench_function("i32_batch", |b| {
        b.iter(|| {
            let input = black_box(&data_i32);
            batch_sum_i32(input)
        })
    });
    group.bench_function("i32_scalar", |b| {
        b.iter(|| {
            let input = black_box(&data_i32);
            input.iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
        })
    });

    group.bench_function("i64_batch", |b| {
        b.iter(|| {
            let input = black_box(&data_i64);
            batch_sum_i64(input)
        })
    });
    group.bench_function("i64_scalar", |b| {
        b.iter(|| {
            let input = black_box(&data_i64);
            input.iter().sum::<i64>()
        })
    });
    group.finish();
}

fn bench_batch_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_arithmetic");
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

    // f32
    group.bench_function("f32_mul_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xi = black_box(x_f32);
            let yi = black_box(y_f32);
            batch_mul_cols_f32(xi, yi, &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f32_mul_scalar", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xi = black_box(x_f32);
            let yi = black_box(y_f32);
            for i in 0..1024 { out[i] = xi[i] * yi[i]; }
            black_box(out[0])
        })
    });

    group.bench_function("f32_add_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xi = black_box(x_f32);
            let yi = black_box(y_f32);
            batch_add_cols_f32(xi, yi, &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f32_add_scalar", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xi = black_box(x_f32);
            let yi = black_box(y_f32);
            for i in 0..1024 { out[i] = xi[i] + yi[i]; }
            black_box(out[0])
        })
    });

    group.bench_function("f32_fma_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xi = black_box(x_f32);
            let yi = black_box(y_f32);
            let zi = black_box(z_f32);
            batch_fma_cols_f32(xi, yi, zi, &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f32_fma_scalar", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            let xi = black_box(x_f32);
            let yi = black_box(y_f32);
            let zi = black_box(z_f32);
            for i in 0..1024 { out[i] = xi[i].mul_add(yi[i], zi[i]); }
            black_box(out[0])
        })
    });

    // f64
    group.bench_function("f64_mul_batch", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xi = black_box(x_f64);
            let yi = black_box(y_f64);
            batch_mul_cols_f64(xi, yi, &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f64_mul_scalar", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xi = black_box(x_f64);
            let yi = black_box(y_f64);
            for i in 0..1024 { out[i] = xi[i] * yi[i]; }
            black_box(out[0])
        })
    });

    group.bench_function("f64_add_batch", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xi = black_box(x_f64);
            let yi = black_box(y_f64);
            batch_add_cols_f64(xi, yi, &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f64_add_scalar", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xi = black_box(x_f64);
            let yi = black_box(y_f64);
            for i in 0..1024 { out[i] = xi[i] + yi[i]; }
            black_box(out[0])
        })
    });

    group.bench_function("f64_fma_batch", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xi = black_box(x_f64);
            let yi = black_box(y_f64);
            let zi = black_box(z_f64);
            batch_fma_cols_f64(xi, yi, zi, &mut out);
            black_box(out[0])
        })
    });
    group.bench_function("f64_fma_scalar", |b| {
        let mut out = [0.0f64; 1024];
        b.iter(|| {
            let xi = black_box(x_f64);
            let yi = black_box(y_f64);
            let zi = black_box(z_f64);
            for i in 0..1024 { out[i] = xi[i].mul_add(yi[i], zi[i]); }
            black_box(out[0])
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_f32,
    bench_batch_f64,
    bench_sums,
    bench_batch_arithmetic
);
criterion_main!(benches);
