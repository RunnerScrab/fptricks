use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_trig_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("trig_f32");
    let val: f32 = 1.23456;
    let trig_val: f32 = 0.5;

    group.bench_function("sin_std", |b| b.iter(|| black_box(val).sin()));
    group.bench_function("sin_fast", |b| b.iter(|| black_box(val).approx_sin()));

    group.bench_function("cos_std", |b| b.iter(|| black_box(val).cos()));
    group.bench_function("cos_fast", |b| b.iter(|| black_box(val).approx_cos()));

    group.bench_function("sin_cos_std", |b| b.iter(|| black_box(val).sin_cos()));
    group.bench_function("sin_cos_fast", |b| b.iter(|| black_box(val).approx_sin_cos()));

    group.bench_function("acos_std", |b| b.iter(|| black_box(trig_val).acos()));
    group.bench_function("acos_fast", |b| b.iter(|| black_box(trig_val).approx_acos()));

    group.bench_function("asin_std", |b| b.iter(|| black_box(trig_val).asin()));
    group.bench_function("asin_fast", |b| b.iter(|| black_box(trig_val).approx_asin()));

    let y_atan: f32 = 0.5;
    let x_atan: f32 = 0.8;
    group.bench_function("atan2_std", |b| {
        b.iter(|| black_box(y_atan).atan2(black_box(x_atan)))
    });
    group.bench_function("atan2_fast", |b| {
        b.iter(|| black_box(y_atan).approx_atan2(black_box(x_atan)))
    });

    group.finish();
}

fn bench_trig_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("trig_f64");
    let val: f64 = 1.23456;
    let trig_val: f64 = 0.5;

    group.bench_function("sin_std", |b| b.iter(|| black_box(val).sin()));
    group.bench_function("sin_fast", |b| b.iter(|| black_box(val).approx_sin()));

    group.bench_function("cos_std", |b| b.iter(|| black_box(val).cos()));
    group.bench_function("cos_fast", |b| b.iter(|| black_box(val).approx_cos()));

    group.bench_function("sin_cos_std", |b| b.iter(|| black_box(val).sin_cos()));
    group.bench_function("sin_cos_fast", |b| b.iter(|| black_box(val).approx_sin_cos()));

    group.bench_function("acos_std", |b| b.iter(|| black_box(trig_val).acos()));
    group.bench_function("acos_fast", |b| b.iter(|| black_box(trig_val).approx_acos()));

    group.bench_function("asin_std", |b| b.iter(|| black_box(trig_val).asin()));
    group.bench_function("asin_fast", |b| b.iter(|| black_box(trig_val).approx_asin()));

    let y_atan: f64 = 0.5;
    let x_atan: f64 = 0.8;
    group.bench_function("atan2_std", |b| {
        b.iter(|| black_box(y_atan).atan2(black_box(x_atan)))
    });
    group.bench_function("atan2_fast", |b| {
        b.iter(|| black_box(y_atan).approx_atan2(black_box(x_atan)))
    });

    group.finish();
}

criterion_group!(benches, bench_trig_f32, bench_trig_f64);
criterion_main!(benches);
