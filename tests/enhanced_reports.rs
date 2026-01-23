use camera_intrinsic_calibration::io::{write_detailed_report, CameraReport};
use serde_json::json;
use tempfile::TempDir;

#[test]
fn test_write_detailed_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.json");

    let intrinsics = vec![camera_intrinsic_model::GenericModel::EUCM(
        camera_intrinsic_model::EUCM::new(&[500.0, 500.0, 320.0, 240.0, 0.5], 640, 480),
    )];

    let rep_rms = vec![(0.5, 100), (0.8, 80)];

    write_detailed_report(output_path.to_str().unwrap(), &intrinsics, None, &rep_rms).unwrap();

    // Check file exists and contains expected content
    let content = std::fs::read_to_string(&output_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(json["overall_rms"].is_number());
    assert!(json["quality_score"].is_number());
    assert!(json["cameras"].is_array());
    assert_eq!(json["cameras"].as_array().unwrap().len(), 2);
}

#[test]
fn test_quality_assessment() {
    // Test various quality scenarios
    let cameras = vec![
        CameraReport {
            id: 0,
            intrinsics: json!([500.0, 500.0, 320.0, 240.0, 0.5]),
            rms_error: 0.3,
            extrinsic_to_cam0: None,
        },
        CameraReport {
            id: 1,
            intrinsics: json!([480.0, 520.0, 310.0, 250.0, 0.6]),
            rms_error: 2.5,
            extrinsic_to_cam0: None,
        },
    ];

    // This would test the quality assessment logic
    // Since it's internal, we test through the public API
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("quality_test.json");

    let intrinsics = vec![
        camera_intrinsic_model::GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
            &[500.0, 500.0, 320.0, 240.0, 0.5],
            640,
            480,
        )),
        camera_intrinsic_model::GenericModel::EUCM(camera_intrinsic_model::EUCM::new(
            &[480.0, 520.0, 310.0, 250.0, 0.6],
            640,
            480,
        )),
    ];

    let rep_rms = vec![(0.3, 200), (2.5, 50)];

    write_detailed_report(output_path.to_str().unwrap(), &intrinsics, None, &rep_rms).unwrap();

    let content = std::fs::read_to_string(&output_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Check quality score is reasonable
    let quality_score = json["quality_score"].as_f64().unwrap();
    assert!(quality_score >= 0.0 && quality_score <= 100.0);

    // Check warnings exist for poor quality
    assert!(json["warnings"].is_array());
    assert!(!json["warnings"].as_array().unwrap().is_empty());

    // Check recommendations exist
    assert!(json["recommendations"].is_array());
}
