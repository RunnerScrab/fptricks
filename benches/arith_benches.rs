use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_arith_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("arith_f32");
    let val: f32 = 1.23456;

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

    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    group.finish();
}

fn bench_arith_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("arith_f64");
    let val: f64 = 1.23456;

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

    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    group.finish();
}

criterion_group!(benches, bench_arith_f32, bench_arith_f64);
criterion_main!(benches);
