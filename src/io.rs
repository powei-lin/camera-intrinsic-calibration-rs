use std::io::Write;

use crate::types::Extrinsics;

pub fn extrinsics_to_json(output_path: &str, extrinsic: &Extrinsics) {
    let j = serde_json::to_string_pretty(extrinsic).unwrap();
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(j.as_bytes()).unwrap();
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
