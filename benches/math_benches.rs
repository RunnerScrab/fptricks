use criterion::{black_box, criterion_group, criterion_main, Criterion};
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
    group.bench_function("sin_cos_fast", |b| b.iter(|| black_box(val).approx_sin_cos()));
    group.bench_function("sin_cos_fast_sep", |b| {
        b.iter(|| {
            let v = black_box(val);
            (v.approx_sin(), v.approx_cos())
        })
    });

    // Inverse
    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    // Power functions
    group.bench_function("powi_std", |b| b.iter(|| black_box(val).powi(black_box(int_pow))));
    group.bench_function("powi_fast", |b| b.iter(|| black_box(val).approx_powi(black_box(int_pow))));

    let float_pow: f32 = 2.5;
    group.bench_function("powf_std", |b| b.iter(|| black_box(val).powf(black_box(float_pow))));
    group.bench_function("powf_fast", |b| b.iter(|| black_box(val).approx_powf(black_box(float_pow))));

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
    group.bench_function("sin_cos_fast", |b| b.iter(|| black_box(val).approx_sin_cos()));
    group.bench_function("sin_cos_fast_sep", |b| {
        b.iter(|| {
            let v = black_box(val);
            (v.approx_sin(), v.approx_cos())
        })
    });

    // Inverse
    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    // Power functions
    group.bench_function("powi_std", |b| b.iter(|| black_box(val).powi(black_box(int_pow))));
    group.bench_function("powi_fast", |b| b.iter(|| black_box(val).approx_powi(black_box(int_pow))));

    let float_pow: f64 = 2.5;
    group.bench_function("powf_std", |b| b.iter(|| black_box(val).powf(black_box(float_pow))));
    group.bench_function("powf_fast", |b| b.iter(|| black_box(val).approx_powf(black_box(float_pow))));

    group.finish();
}

fn bench_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f32");
    let x = [1.23456f32; 8];
    let y = [2.5f32; 8];
    let n = [5i32; 8];

    group.bench_function("ln_batch", |b| b.iter(|| batch_approx_ln_f32(black_box(x))));
    group.bench_function("ln_scalar_loop", |b| b.iter(|| {
        let val = black_box(x);
        let mut out = [0.0; 8];
        for i in 0..8 { out[i] = val[i].approx_ln(); }
        out
    }));

    group.bench_function("exp_batch", |b| b.iter(|| batch_approx_exp_f32(black_box(x))));
    group.bench_function("sqrt_batch", |b| b.iter(|| batch_approx_sqrt_f32(black_box(x))));
    group.bench_function("cbrt_batch", |b| b.iter(|| batch_approx_cbrt_f32(black_box(x))));
    group.bench_function("sin_cos_batch", |b| b.iter(|| batch_approx_sin_cos_f32(black_box(x))));
    
    group.bench_function("powf_cols_batch", |b| b.iter(|| batch_approx_powf_cols_f32(black_box(x), black_box(y))));
    group.bench_function("powi_cols_batch", |b| b.iter(|| batch_approx_powi_cols_f32(black_box(x), black_box(n))));

    group.bench_function("fmadd_cols_batch", |b| b.iter(|| batch_fmadd_cols_f32(black_box(x), black_box(y), black_box(y))));
    group.bench_function("asymmetric_fma_cols_batch", |b| b.iter(|| batch_asymmetric_fma_cols_f32(black_box(x), black_box(y), black_box(y), black_box(y))));

    group.finish();
}

fn bench_batch_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f64");
    let x = [1.23456f64; 4];
    let y = [2.5f64; 4];
    let n = [5i32; 4];

    group.bench_function("ln_batch", |b| b.iter(|| batch_approx_ln_f64(black_box(x))));
    group.bench_function("ln_scalar_loop", |b| b.iter(|| {
        let val = black_box(x);
        let mut out = [0.0; 4];
        for i in 0..4 { out[i] = val[i].approx_ln(); }
        out
    }));

    group.bench_function("exp_batch", |b| b.iter(|| batch_approx_exp_f64(black_box(x))));
    group.bench_function("sqrt_batch", |b| b.iter(|| batch_approx_sqrt_f64(black_box(x))));
    group.bench_function("cbrt_batch", |b| b.iter(|| batch_approx_cbrt_f64(black_box(x))));
    group.bench_function("sin_cos_batch", |b| b.iter(|| batch_approx_sin_cos_f64(black_box(x))));

    group.bench_function("powf_cols_batch", |b| b.iter(|| batch_approx_powf_cols_f64(black_box(x), black_box(y))));
    group.bench_function("powi_cols_batch", |b| b.iter(|| batch_approx_powi_cols_f64(black_box(x), black_box(n))));

    group.bench_function("fmadd_cols_batch", |b| b.iter(|| batch_fmadd_cols_f64(black_box(x), black_box(y), black_box(y))));
    group.bench_function("asymmetric_fma_cols_batch", |b| b.iter(|| batch_asymmetric_fma_cols_f64(black_box(x), black_box(y), black_box(y), black_box(y))));

    group.finish();
}

criterion_group!(benches, bench_f32, bench_f64, bench_batch_f32, bench_batch_f64);
criterion_main!(benches);
