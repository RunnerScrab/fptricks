use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_arith_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("arith_f32");
    let val: f32 = 1.23456;

    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    group.finish();
}

fn bench_arith_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("arith_f64");
    let val: f64 = 1.23456;

    group.bench_function("inv_std", |b| b.iter(|| 1.0 / black_box(val)));
    group.bench_function("inv_fast", |b| b.iter(|| black_box(val).approx_inv()));

    group.finish();
}

criterion_group!(benches, bench_arith_f32, bench_arith_f64);
criterion_main!(benches);
