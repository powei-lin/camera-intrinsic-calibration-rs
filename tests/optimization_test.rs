use camera_intrinsic_calibration::detected_points::{FeaturePoint, FrameFeature};
use camera_intrinsic_calibration::optimization::{
    factors::ReprojectionFactor, homography::homography_to_focal, linear::init_pose,
};
use camera_intrinsic_model::{GenericModel, UCM};
use glam::{Vec2, Vec3};
use nalgebra as na;
use std::collections::HashMap;
use tiny_solver::factors::Factor;

#[test]
fn test_homography_to_focal_general() {
    let f = 1000.0;
    let k = na::Matrix3::new(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0);
    let k_inv = k.try_inverse().unwrap();

    // Rotate around a general axis
    let axis = na::Unit::new_normalize(na::Vector3::new(1.0, 1.0, 0.5));
    let angle = 0.2f32; // rad
    let r_mat: na::Matrix3<f32> = na::Rotation3::from_axis_angle(&axis, angle).into_inner();

    let h = k * r_mat * k_inv;

    let solved_f = homography_to_focal(&h).expect("Failed to solve focal");

    // Accuracy might depend on angle size or float precision
    assert!((solved_f - f).abs() < 10.0, "Focal mismatch: got {}, expected {}", solved_f, f);
}

#[test]
fn test_reprojection_factor() {
    // Setup UCM model
    let w = 640;
    let h = 480;
    // Params: [gamma_x, gamma_y, cx, cy, alpha]
    let cam_params = na::dvector![500.0, 500.0, 320.0, 240.0, 0.5];
    let model: GenericModel<f64> = GenericModel::UCM(UCM::new(&cam_params, w, h));

    let p3d = glam::Vec3::new(1.0, 2.0, 10.0);

    // Project using model
    let p3d_na = na::Vector3::new(p3d.x as f64, p3d.y as f64, p3d.z as f64);
    let p2d_na = model.project_one(&p3d_na);
    let p2d = glam::Vec2::new(p2d_na[0] as f32, p2d_na[1] as f32);

    // Create factor
    let factor = ReprojectionFactor::new(
        &model, &p3d, &p2d, false, // xy_same_focal
    );

    let rvec = na::dvector![0.0, 0.0, 0.0];
    let tvec = na::dvector![0.0, 0.0, 0.0];

    // Ensure params match what residual_func expects.
    // residual_func uses params[0].
    let all_params = vec![cam_params.clone(), rvec.clone(), tvec.clone()];

    let residual = factor.residual_func(&all_params);

    assert!(residual.norm() < 1e-4, "Residual should be zero at GT. Got {}", residual.norm());

    // Perturb camera translation
    let tvec_bad = na::dvector![0.1, 0.0, 0.0];
    let bad_params = vec![cam_params.clone(), rvec, tvec_bad];
    let residual_bad = factor.residual_func(&bad_params);

    assert!(residual_bad.norm() > 1e-3, "Residual should be non-zero for bad params");
}

#[test]
fn test_init_pose() {
    // Construct 3D points in a known configuration
    let mut points_3d = Vec::new();
    points_3d.push(Vec3::new(0.0, 0.0, 0.0));
    points_3d.push(Vec3::new(1.0, 0.0, 0.0));
    points_3d.push(Vec3::new(0.0, 1.0, 0.0));
    points_3d.push(Vec3::new(1.0, 1.0, 0.0));

    // Construct a known pose (Identity)
    // Camera at origin, looking down +Z (or depending on convention).
    // Let's assume standard camera frame.
    // If points are at Z=5.0
    let z_offset = 5.0;
    let p3d_in_cam: Vec<_> = points_3d.iter().map(|p| *p + Vec3::Z * z_offset).collect();

    // Project to normalized coords (x/z, y/z)
    let lambda = 0.0; // No distortion

    // Construct FrameFeature
    let mut features = HashMap::new();
    let w = 1000;
    let h = 1000;
    let cx = 500.0;
    let cy = 500.0;
    // half_img_size = 500.0

    for (i, p) in p3d_in_cam.iter().enumerate() {
        let x_norm = p.x / p.z;
        let y_norm = p.y / p.z;

        // Un-normalize to pixel
        // code: xy = (p2d - cxcy) / half_img_size
        // p2d = xy * half_img_size + cxcy
        let half = 500.0;
        let u = x_norm * half + cx;
        let v = y_norm * half + cy;

        features.insert(
            i as u32,
            FeaturePoint {
                p2d: Vec2::new(u, v),
                p3d: points_3d[i], // The 3D point in world frame (we assumed world = cam translated by -Z)
            },
        );
    }

    let frame_feature = FrameFeature { time_ns: 0, img_w_h: (w, h), features };

    let (r_vec, t_vec) = init_pose(&frame_feature, lambda);

    // Our world points were (0,0,0) etc.
    // Camera sees them at (0,0,5)...
    // So Transform World to Cam: P_c = R * P_w + t
    // (0,0,5) = R * (0,0,0) + t  => t = (0,0,5).
    // R should be Identity.

    println!("Estimated R: {:?}, t: {:?}", r_vec, t_vec);

    // Check t
    assert!((t_vec.0 - 0.0).abs() < 0.1);
    assert!((t_vec.1 - 0.0).abs() < 0.1);
    assert!((t_vec.2 - z_offset as f64).abs() < 0.1);

    // Check R (rotation vector for identity is 0)
    assert!(r_vec.0.abs() < 0.1);
    assert!(r_vec.1.abs() < 0.1);
    assert!(r_vec.2.abs() < 0.1);
}
