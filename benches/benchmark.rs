use camera_intrinsic_calibration::optimization::{
    factors::ReprojectionFactor, homography::homography_to_focal,
};
use camera_intrinsic_model::{GenericModel, UCM};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glam::Vec2;
use nalgebra as na;

fn bench_homography_solve(c: &mut Criterion) {
    let f: f32 = 1000.0f32;
    let k: na::Matrix3<f32> = na::Matrix3::new(f, 0.0f32, 0.0f32, 0.0f32, f, 0.0f32, 0.0f32, 0.0f32, 1.0f32);
    let k_inv = k.try_inverse().unwrap();

    let axis = na::Unit::new_normalize(na::Vector3::<f32>::new(1.0f32, 1.0f32, 0.5f32));
    let angle = 0.2f32;
    let r_mat: na::Matrix3<f32> = na::Rotation3::from_axis_angle(&axis, angle).into_inner();
    let h = k * r_mat * k_inv;

    c.bench_function("homography_to_focal", |b| {
        b.iter(|| homography_to_focal(black_box(&h)))
    });
}

fn bench_reprojection_residual(c: &mut Criterion) {
    let params = na::dvector![500.0, 500.0, 320.0, 240.0, 0.5];
    let model: GenericModel<f64> = GenericModel::UCM(UCM::new(&params, 640, 480));

    let p3d = glam::Vec3::new(1.0, 2.0, 10.0);
    let p2d = Vec2::new(320.0, 240.0);
    let factor = ReprojectionFactor::new(&model, &p3d, &p2d, false);

    let rvec = na::dvector![0.0, 0.0, 0.0];
    let tvec = na::dvector![0.0, 0.0, 0.0];
    let all_params = vec![params.clone(), rvec, tvec];

    c.bench_function("reprojection_residual", |b| {
        b.iter(|| factor.residual_func(black_box(&all_params)))
    });
}

criterion_group!(benches, bench_homography_solve, bench_reprojection_residual);
criterion_main!(benches);
