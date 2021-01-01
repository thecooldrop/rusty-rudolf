use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array3};
use rusty_rudolf::kalman;

pub fn kalman_predict_benchmark(c: &mut Criterion) {
    let kf = kalman::KalmanFilter::<f64>::new(
        &Array2::eye(8),
        &Array2::eye(8),
        &Array2::eye(8),
        &Array2::eye(8),
    ).unwrap();
    c.bench_function("Kalman filter predict for 1000 inputs", |b| {
        b.iter(|| {
            kf.predict(
                &black_box(Array2::zeros([1000, 8])),
                &black_box(Array3::zeros([1000, 8, 8])),
            )
        })
    });
}

pub fn kalman_update_benchmark(c: &mut Criterion) {
    let kf = kalman::KalmanFilter::<f64>::new(
        &Array2::eye(8),
        &Array2::eye(8),
        &Array2::eye(8),
        &Array2::eye(8),
    ).unwrap();
    c.bench_function(
        "Kalman update for updating 1000 inputs with 10 measurements, totaling in 10000 updates",
        |b| {
            b.iter_with_large_drop(|| {
                kf.update(
                    &black_box(Array2::zeros([1000, 8])),
                    &black_box(Array3::zeros([1000, 8, 8])),
                    &black_box(Array2::zeros([10, 8])),
                )
            })
        },
    );
}

criterion_group!(predict, kalman_predict_benchmark);
criterion_group!(update, kalman_update_benchmark);
criterion_main!(predict, update);
