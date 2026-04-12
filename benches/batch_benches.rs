use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f32");
    let x = [1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2];
    let y = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
    let z = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let n = [5i32; 8];

    group.bench_function("ln_batch", |b| b.iter(|| batch_approx_ln_f32(black_box(x))));
    group.bench_function("ln_scalar_loop", |b| {
        let mut out = [0.0f32; 8];
        b.iter(|| {
            let val = black_box(x);
            for i in 0..8 { out[i] = val[i].approx_ln(); }
            black_box(out[0])
        })
    });

    group.bench_function("inv_batch", |b| {
        b.iter(|| batch_approx_inv_f32(black_box(x)))
    });
    group.bench_function("inv_scalar_loop", |b| {
        let mut out = [0.0f32; 8];
        b.iter(|| {
            let val = black_box(x);
            for i in 0..8 { out[i] = val[i].approx_inv(); }
            black_box(out[0])
        })
    });

    group.bench_function("exp_batch", |b| {
        b.iter(|| batch_approx_exp_f32(black_box(x)))
    });
    group.bench_function("exp_scalar_loop", |b| {
        let mut out = [0.0f32; 8];
        b.iter(|| {
            let val = black_box(x);
            for i in 0..8 { out[i] = val[i].approx_exp(); }
            black_box(out[0])
        })
    });

    group.bench_function("sqrt_batch", |b| {
        b.iter(|| batch_approx_sqrt_f32(black_box(x)))
    });

    group.bench_function("cbrt_batch", |b| {
        b.iter(|| batch_approx_cbrt_f32(black_box(x)))
    });

    group.bench_function("sin_cos_batch", |b| {
        b.iter(|| batch_approx_sin_cos_f32(black_box(x)))
    });

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| chunk_approx_powf_cols_f32(black_box(x), black_box(y)))
    });

    group.bench_function("powi_cols_batch", |b| {
        b.iter(|| batch_approx_powi_cols_f32(black_box(x), black_box(n)))
    });

    group.bench_function("fmadd_cols_batch", |b| {
        b.iter(|| batch_fmadd_cols_f32(black_box(x), black_box(y), black_box(z)))
    });

    group.bench_function("asymmetric_fma_cols_batch", |b| {
        b.iter(|| {
            batch_asymmetric_fma_cols_f32(black_box(x), black_box(z), black_box(y), black_box(z))
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
    group.bench_function("inv_batch", |b| {
        b.iter(|| batch_approx_inv_f64(black_box(x)))
    });
    group.bench_function("exp_batch", |b| {
        b.iter(|| batch_approx_exp_f64(black_box(x)))
    });
    group.bench_function("sqrt_batch", |b| {
        b.iter(|| batch_approx_sqrt_f64(black_box(x)))
    });
    group.bench_function("cbrt_batch", |b| {
        b.iter(|| batch_approx_cbrt_f64(black_box(x)))
    });
    group.bench_function("sin_cos_batch", |b| {
        b.iter(|| batch_approx_sin_cos_f64(black_box(x)))
    });
    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| chunk_approx_powf_cols_f64(black_box(x), black_box(y)))
    });
    group.bench_function("powi_cols_batch", |b| {
        b.iter(|| batch_approx_powi_cols_f64(black_box(x), black_box(n)))
    });
    group.bench_function("fmadd_cols_batch", |b| {
        b.iter(|| batch_fmadd_cols_f64(black_box(x), black_box(y), black_box(z)))
    });
    group.bench_function("asymmetric_fma_cols_batch", |b| {
        b.iter(|| {
            batch_asymmetric_fma_cols_f64(black_box(x), black_box(z), black_box(y), black_box(z))
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
    group.bench_function("f32_batch", |b| b.iter(|| batch_sum_f32(black_box(&data_f32))));
    group.bench_function("f64_batch", |b| b.iter(|| batch_sum_f64(black_box(&data_f64))));
    group.bench_function("i32_batch", |b| b.iter(|| batch_sum_i32(black_box(&data_i32))));
    group.bench_function("i64_batch", |b| b.iter(|| batch_sum_i64(black_box(&data_i64))));
    group.finish();
}

fn bench_batch_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_arithmetic");
    let vec_x_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 1.001).collect();
    let vec_y_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 0.999).collect();
    let vec_z_f32: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    
    let x_f32: &[f32; 1024] = vec_x_f32.as_slice().try_into().unwrap();
    let y_f32: &[f32; 1024] = vec_y_f32.as_slice().try_into().unwrap();
    let z_f32: &[f32; 1024] = vec_z_f32.as_slice().try_into().unwrap();

    group.bench_function("f32_mul_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            batch_mul_cols_f32(black_box(x_f32), black_box(y_f32), &mut out);
            black_box(out[0])
        })
    });

    group.bench_function("f32_add_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            batch_add_cols_f32(black_box(x_f32), black_box(y_f32), &mut out);
            black_box(out[0])
        })
    });

    group.bench_function("f32_fma_batch", |b| {
        let mut out = [0.0f32; 1024];
        b.iter(|| {
            batch_fma_cols_f32(black_box(x_f32), black_box(y_f32), black_box(z_f32), &mut out);
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
