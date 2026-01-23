use std::io::Write;

use serde::{de::DeserializeOwned, Serialize};

/// Serializes an object to a JSON file.
pub fn object_to_json<T: Serialize>(output_path: &str, object: &T) {
    let j = serde_json::to_string_pretty(object).unwrap();
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(j.as_bytes()).unwrap();
}
/// Deserializes an object from a JSON file.
pub fn object_from_json<T: DeserializeOwned>(file_path: &str) -> T {
    let contents =
        std::fs::read_to_string(file_path).expect("Should have been able to read the file");
    serde_json::from_str(&contents).unwrap()
}

/// Writes a calibration report to a text file.
///
/// Detailed report includes average and median reprojection errors for each camera.
#[derive(serde::Serialize)]
struct CalibrationReport {
    timestamp: String,
    extrinsic_calibrated: bool,
    cameras: Vec<CameraReport>,
    overall_rms: f64,
    total_points: usize,
}

#[derive(serde::Serialize)]
struct CameraReport {
    id: usize,
    intrinsics: serde_json::Value,
    rms_error: f64,
    point_count: usize,
    extrinsic_to_cam0: Option<serde_json::Value>,
}

pub fn write_detailed_report(
    output_path: &str,
    intrinsics: &[camera_intrinsic_model::GenericModel<f64>],
    extrinsics: Option<&[RvecTvec]>,
    rep_rms: &[(f64, f64)],
) -> std::io::Result<()> {
    use std::time::SystemTime;

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let cameras = intrinsics
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let (rms, points) = rep_rms.get(i).copied().unwrap_or((0.0, 0));
            let extrinsic = extrinsics.and_then(|ext| ext.get(i)).map(|rt| {
                serde_json::json!({
                    "rvec": rt.na_rvec().as_slice(),
                    "tvec": rt.na_tvec().as_slice()
                })
            });

            CameraReport {
                id: i,
                intrinsics: model.params().iter().cloned().collect(),
                rms_error: rms,
                point_count: points,
                extrinsic_to_cam0: extrinsic,
            }
        })
        .collect();

    let overall_rms = rep_rms.iter().map(|(rms, _)| rms).sum::<f64>() / rep_rms.len() as f64;
    let total_points = rep_rms.iter().map(|(_, pts)| pts).sum();

    let report = CalibrationReport {
        timestamp: timestamp.to_string(),
        extrinsic_calibrated: extrinsics.is_some(),
        cameras,
        overall_rms,
        total_points,
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(output_path, json)?;
    Ok(())
}

pub fn write_report(output_path: &str, with_extrinsic: bool, rep_rms: &[(f64, f64)]) {
    let mut s = String::new();
    s += format!("Calibrate with extrinsics: {}\n\n", with_extrinsic).as_str();
    for (cam_idx, &(avg_rep, med_rep)) in rep_rms.iter().enumerate() {
        s += format!("cam{}:\n", cam_idx).as_str();
        s += format!("    average reprojection error: {:.5} px\n", avg_rep).as_str();
        s += format!("    median  reprojection error: {:.5} px\n\n", med_rep).as_str();
    }
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(s.as_bytes()).unwrap();
}
