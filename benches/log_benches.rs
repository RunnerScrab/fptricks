use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_log_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_f32");
    let val: f32 = 1.23456;

    group.bench_function("exp_std", |b| b.iter(|| black_box(val).exp()));
    group.bench_function("exp_fast", |b| b.iter(|| black_box(val).approx_exp()));

    group.bench_function("ln_std", |b| b.iter(|| black_box(val).ln()));
    group.bench_function("ln_fast", |b| b.iter(|| black_box(val).approx_ln()));

    group.bench_function("sqrt_std", |b| b.iter(|| black_box(val).sqrt()));
    group.bench_function("sqrt_fast", |b| b.iter(|| black_box(val).approx_sqrt()));

    group.bench_function("cbrt_std", |b| b.iter(|| black_box(val).cbrt()));
    group.bench_function("cbrt_fast", |b| b.iter(|| black_box(val).approx_cbrt()));

    let int_pow: i32 = 5;
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

fn bench_log_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_f64");
    let val: f64 = 1.23456;

    group.bench_function("exp_std", |b| b.iter(|| black_box(val).exp()));
    group.bench_function("exp_fast", |b| b.iter(|| black_box(val).approx_exp()));
    group.bench_function("exp_fastb", |b| b.iter(|| approx_exp_f64b(black_box(val))));

    group.bench_function("ln_std", |b| b.iter(|| black_box(val).ln()));
    group.bench_function("ln_fast", |b| b.iter(|| black_box(val).approx_ln()));

    group.bench_function("sqrt_std", |b| b.iter(|| black_box(val).sqrt()));
    group.bench_function("sqrt_fast", |b| b.iter(|| black_box(val).approx_sqrt()));

    group.bench_function("cbrt_std", |b| b.iter(|| black_box(val).cbrt()));
    group.bench_function("cbrt_fast", |b| b.iter(|| black_box(val).approx_cbrt()));

    let int_pow: i32 = 5;
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

criterion_group!(benches, bench_log_f32, bench_log_f64);
criterion_main!(benches);
