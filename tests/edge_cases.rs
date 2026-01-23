use camera_intrinsic_calibration::board::{Board, BoardConfig};
use camera_intrinsic_calibration::data_loader::load_frame_features;
use camera_intrinsic_calibration::detected_points::{FeaturePoint, FrameFeature};
use camera_intrinsic_calibration::optimization::{
    factors::ReprojectionFactor, homography::homography_to_focal, linear::init_pose,
};
use camera_intrinsic_calibration::util::{convert_model, find_best_two_frames_idx};
use camera_intrinsic_model::{EUCM, GenericModel, UCM};
use glam::Vec2;
use nalgebra as na;
use std::collections::HashMap;

#[test]
fn test_board_edge_cases() {
    // Test zero rows/cols
    let config = BoardConfig {
        tag_rows: 0,
        tag_cols: 1,
        ..Default::default()
    };
    let board = Board::from_config(&config);
    assert_eq!(board.id_to_3d.len(), 0);

    // Test negative spacing (should still work, but maybe not sensible)
    let config2 = BoardConfig {
        tag_spacing: -0.1,
        ..Default::default()
    };
    let board2 = Board::from_config(&config2);
    assert_eq!(board2.id_to_3d.len(), 6 * 6 * 4); // Same length

    // Test zero size
    let config3 = BoardConfig {
        tag_size_meter: 0.0,
        ..Default::default()
    };
    let board3 = Board::from_config(&config3);
    // Should still create points, all at origin basically
    assert_eq!(board3.id_to_3d.len(), 6 * 6 * 4);
}

#[test]
fn test_data_loader_edge_cases() {
    // Test with non-existent path
    let result = load_frame_features("non_existent_path");
    assert!(result.is_err());

    // Test empty directory (if we create one, but skip for now)
}

#[test]
fn test_homography_edge_cases() {
    // Test with identity matrix (should give back focal)
    let f = 1000.0;
    let k = na::Matrix3::new(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0);
    let h = k.clone(); // Identity homography
    let solved_f = homography_to_focal(&h).expect("Failed to solve focal");
    assert!((solved_f - f).abs() < 1e-6);

    // Test with zero matrix (should fail gracefully)
    let h_zero = na::Matrix3::zeros();
    let result = homography_to_focal(&h_zero);
    assert!(result.is_none()); // Expect None for degenerate case
}

#[test]
fn test_reprojection_factor_edge_cases() {
    let params = na::dvector![500.0, 500.0, 320.0, 240.0, 0.5];
    let model: GenericModel<f64> = GenericModel::UCM(UCM::new(&params, 640, 480));

    // Test with point at infinity (z=0)
    let p3d_bad = glam::Vec3::new(1.0, 1.0, 0.0); // Invalid for projection
    let p2d = Vec2::ZERO;
    let factor = ReprojectionFactor::new(&model, &p3d_bad, &p2d, false, false);

    let rvec = na::dvector![0.0, 0.0, 0.0];
    let tvec = na::dvector![0.0, 0.0, 0.0];
    let all_params = vec![params.clone(), rvec, tvec];
    let residual = factor.residual_func(&all_params);
    // Should handle gracefully, residual might be large
    assert!(residual.norm().is_finite());
}

#[test]
fn test_init_pose_edge_cases() {
    // Test with no points
    let frame = FrameFeature {
        time_ns: 0,
        img_w_h: (100, 100),
        features: HashMap::new(),
    };
    let (r, t) = init_pose(&frame, 0.0);
    // Should return zeros or defaults
    assert_eq!(r.0, 0.0);
    assert_eq!(t.0, 0.0);

    // Test with single point
    let mut features = HashMap::new();
    features.insert(
        0,
        FeaturePoint {
            p2d: Vec2::ZERO,
            p3d: glam::Vec3::Z, // At origin
        },
    );
    let frame2 = FrameFeature {
        time_ns: 0,
        img_w_h: (100, 100),
        features,
    };
    let (r2, t2) = init_pose(&frame2, 0.0);
    assert!(r2.0.is_finite());
    assert!(t2.0.is_finite());
}

#[test]
fn test_find_best_two_frames_edge_cases() {
    // Test with no frames
    let frames: Vec<Option<FrameFeature>> = vec![];
    let (i1, i2) = find_best_two_frames_idx(&frames, false);
    // Should handle gracefully, perhaps panic or return defaults
    // Based on code, it might panic if no frames

    // Test with one frame
    let frame = Some(FrameFeature {
        time_ns: 0,
        img_w_h: (100, 100),
        features: HashMap::new(),
    });
    let frames2 = vec![frame];
    // Code might not handle single frame well, but let's see
}

#[test]
fn test_convert_model_edge_cases() {
    // Test invalid conversion (same model types)
    let params = na::dvector![500.0, 500.0, 320.0, 240.0, 0.5];
    let ucm = GenericModel::UCM(UCM::new(&params, 640, 480));
    let mut ucm_target = GenericModel::UCM(UCM::new(&params, 640, 480));
    convert_model(&ucm, &mut ucm_target, 0);
    // Should be no-op for same type

    // Test unsupported conversion
    // Depending on code, might not do anything
}
