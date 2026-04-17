use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fptricks::*;

fn bench_trig_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_trig_f32");
    let x = [1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2];

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

    group.finish();
}

fn bench_trig_batch_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_trig_f64");
    let x = [1.2, -1.2, 1.2, -1.2];

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

    group.finish();
}

criterion_group!(benches, bench_trig_batch_f32, bench_trig_batch_f64);
criterion_main!(benches);
