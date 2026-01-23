use camera_intrinsic_calibration::types::RvecTvec;
use camera_intrinsic_calibration::util::{analyze_extrinsic_outliers, validate_extrinsics};
use camera_intrinsic_model::GenericModel;

#[test]
fn test_validate_extrinsics_no_errors() {
    let intrinsics = vec![GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
        &[500.0, 500.0, 320.0, 240.0, 0.5],
        640,
        480,
    ))];
    let extrinsics = vec![RvecTvec::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0])];
    let cam_rtvecs = vec![std::collections::HashMap::new()];

    let errors = validate_extrinsics(&intrinsics, &extrinsics, &cam_rtvecs);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_extrinsics_with_errors() {
    let intrinsics = vec![
        GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
            &[500.0, 500.0, 320.0, 240.0, 0.5],
            640,
            480,
        )),
        GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
            &[500.0, 500.0, 320.0, 240.0, 0.5],
            640,
            480,
        )),
    ];

    let extrinsics = vec![
        RvecTvec::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]),
        RvecTvec::new(&[0.1, 0.0, 0.0], &[0.5, 0.0, 0.0]),
    ];

    let mut cam_rtvecs = vec![
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    ];

    // Add some frame data
    cam_rtvecs[0].insert(0, RvecTvec::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]));
    cam_rtvecs[1].insert(0, RvecTvec::new(&[0.05, 0.0, 0.0], &[0.2, 0.0, 0.0]));

    let errors = validate_extrinsics(&intrinsics, &extrinsics, &cam_rtvecs);
    // Should detect some inconsistencies
    assert!(!errors.is_empty());
}

#[test]
fn test_analyze_extrinsic_outliers() {
    let errors = vec![
        (0, 1, 0.01),
        (0, 1, 5.0), // Outlier
        (1, 2, 0.02),
    ];

    let (outliers, recommendations) = analyze_extrinsic_outliers(&errors);

    assert!(!outliers.is_empty());
    assert!(outliers.contains(&(0, 1)));
    assert!(!recommendations.is_empty());
}

#[test]
fn test_analyze_extrinsic_no_outliers() {
    let errors = vec![(0, 1, 0.01), (0, 1, 0.015), (1, 2, 0.02)];

    let (outliers, recommendations) = analyze_extrinsic_outliers(&errors);

    assert!(outliers.is_empty());
    assert!(recommendations.is_empty());
}

#[test]
fn test_pose_difference_calculation() {
    use nalgebra as na;

    let pose1 = na::Isometry3::from_parts(
        na::Translation3::new(0.0, 0.0, 0.0),
        na::UnitQuaternion::identity(),
    );

    let pose2 = na::Isometry3::from_parts(
        na::Translation3::new(1.0, 0.0, 0.0),
        na::UnitQuaternion::from_euler_angles(0.1, 0.0, 0.0),
    );

    // This tests the internal calculation, but since it's private,
    // we test through the public API
    let intrinsics = vec![
        GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
            &[500.0, 500.0, 320.0, 240.0, 0.5],
            640,
            480,
        )),
        GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
            &[500.0, 500.0, 320.0, 240.0, 0.5],
            640,
            480,
        )),
    ];

    let extrinsics = vec![
        RvecTvec::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]),
        RvecTvec::new(&[0.1, 0.0, 0.0], &[1.0, 0.0, 0.0]),
    ];

    let mut cam_rtvecs = vec![
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    ];
    cam_rtvecs[0].insert(0, RvecTvec::new(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]));
    cam_rtvecs[1].insert(0, RvecTvec::new(&[0.05, 0.0, 0.0], &[0.5, 0.0, 0.0]));

    let errors = validate_extrinsics(&intrinsics, &extrinsics, &cam_rtvecs);
    assert!(!errors.is_empty());
    // Verify error values are reasonable
    for (_, _, error) in errors {
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }
}
