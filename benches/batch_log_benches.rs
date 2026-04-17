use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_log_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_log_f32");
    let x = [1.2, 0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 0.8];
    let y = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
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

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(&x);
            let vy = black_box(&y);
            let out = batch_approx_powf_cols_f32(vx, vy);
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

    group.finish();
}

fn bench_log_batch_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_log_f64");
    let x = [1.2, 0.8, 1.2, 0.8];
    let y = [2.5, 2.5, 2.5, 2.5];
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

    group.bench_function("powf_cols_batch", |b| {
        b.iter(|| {
            let vx = black_box(&x);
            let vy = black_box(&y);
            let out = batch_approx_powf_cols_f64(vx, vy);
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

    group.finish();
}

criterion_group!(benches, bench_log_batch_f32, bench_log_batch_f64);
criterion_main!(benches);
