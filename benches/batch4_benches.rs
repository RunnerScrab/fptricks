use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_batch4_arith(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch4_arith");
    let x4 = [1.2, -1.2, 1.2, -1.2];
    let x8 = [1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2];
    let y4 = [2.5, 2.5, 2.5, 2.5];
    let y8 = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
    let z4 = [0.5, 0.5, 0.5, 0.5];
    let z8 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    
    // INV
    group.bench_function("inv_batch4_f32", |b| {
        b.iter(|| batch4_approx_inv_f32(black_box(x4)))
    });
    group.bench_function("inv_batch_f64", |b| {
        let x_f64 = [1.2, -1.2, 1.2, -1.2];
        b.iter(|| batch_approx_inv_f64(black_box(x_f64)))
    });
    group.bench_function("inv_scalar4_f32", |b| {
        b.iter(|| {
            let val = black_box(x4);
            let mut out = [0.0f32; 4];
            for i in 0..4 { out[i] = val[i].approx_inv(); }
            out
        })
    });
    
    // FMADD
    group.bench_function("fmadd_batch4_f32", |b| {
        b.iter(|| batch4_fmadd_cols_f32(black_box(x4), black_box(y4), black_box(z4)))
    });
    group.bench_function("fmadd_batch_f64", |b| {
        let x_f64 = [1.2, -1.2, 1.2, -1.2];
        let y_f64 = [2.5, 2.5, 2.5, 2.5];
        let z_f64 = [0.5, 0.5, 0.5, 0.5];
        b.iter(|| batch_fmadd_cols_f64(black_box(x_f64), black_box(y_f64), black_box(z_f64)))
    });
    
    // Comparison 8-wide vs 2x4-wide
    group.bench_function("fmadd_batch8_f32", |b| {
        b.iter(|| batch_fmadd_cols_f32(black_box(x8), black_box(y8), black_box(z8)))
    });
    group.bench_function("fmadd_2x_batch4_f32", |b| {
        b.iter(|| {
            let vx = black_box(x8);
            let vy = black_box(y8);
            let vz = black_box(z8);
            let mut out = [0.0f32; 8];
            let low = [vx[0], vx[1], vx[2], vx[3]];
            let low_y = [vy[0], vy[1], vy[2], vy[3]];
            let low_z = [vz[0], vz[1], vz[2], vz[3]];
            let high = [vx[4], vx[5], vx[6], vx[7]];
            let high_y = [vy[4], vy[5], vy[6], vy[7]];
            let high_z = [vz[4], vz[5], vz[6], vz[7]];
            
            let r1 = batch4_fmadd_cols_f32(low, low_y, low_z);
            let r2 = batch4_fmadd_cols_f32(high, high_y, high_z);
            
            out[0..4].copy_from_slice(&r1);
            out[4..8].copy_from_slice(&r2);
            out
        })
    });

    group.finish();
}

fn bench_batch4_transcendental(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch4_trans");
    let x4 = [1.2, 0.8, 1.5, 0.5];
    let y4 = [2.5, 2.5, 2.5, 2.5];
    let n4 = [5i32; 4];

    // LN
    group.bench_function("ln_batch4_f32", |b| {
        b.iter(|| batch4_approx_ln_f32(black_box(x4)))
    });
    group.bench_function("ln_batch_f64", |b| {
        let x_f64 = [1.2, 0.8, 1.5, 0.5];
        b.iter(|| batch_approx_ln_f64(black_box(x_f64)))
    });

    // EXP
    group.bench_function("exp_batch4_f32", |b| {
        b.iter(|| batch4_approx_exp_f32(black_box(x4)))
    });
    group.bench_function("exp_batch_f64", |b| {
        let x_f64 = [1.2, 0.8, 1.5, 0.5];
        b.iter(|| batch_approx_exp_f64(black_box(x_f64)))
    });

    // POWI
    group.bench_function("powi_batch4_f32", |b| {
        b.iter(|| batch4_approx_powi_cols_f32(black_box(x4), black_box(n4)))
    });
    group.bench_function("powi_batch_f64", |b| {
        let x_f64 = [1.2, 0.8, 1.5, 0.5];
        let n_f64 = [5i32; 4];
        b.iter(|| batch_approx_powi_cols_f64(black_box(x_f64), black_box(n_f64)))
    });

    group.finish();
}

fn bench_batch4_trig(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch4_trig");
    let x4 = [0.0, 1.0, 3.14159, -1.0];

    group.bench_function("sin_cos_batch4_f32", |b| {
        b.iter(|| batch4_approx_sin_cos_f32(black_box(x4)))
    });
    group.bench_function("sin_cos_batch_f64", |b| {
        let x_f64 = [0.0, 1.0, 3.14159, -1.0];
        b.iter(|| batch_approx_sin_cos_f64(black_box(x_f64)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_batch4_arith,
    bench_batch4_transcendental,
    bench_batch4_trig
);
criterion_main!(benches);
