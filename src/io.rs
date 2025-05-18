use std::io::Write;

use serde::{Serialize, de::DeserializeOwned};

pub fn object_to_json<T: Serialize>(output_path: &str, object: &T) {
    let j = serde_json::to_string_pretty(object).unwrap();
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(j.as_bytes()).unwrap();
}
pub fn object_from_json<T: DeserializeOwned>(file_path: &str) -> T {
    let contents =
        std::fs::read_to_string(file_path).expect("Should have been able to read the file");
    serde_json::from_str(&contents).unwrap()
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
