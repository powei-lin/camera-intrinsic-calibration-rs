use std::io::Write;

use super::generic::GenericModel;

pub fn model_to_json(output_path: &str, camera_model: &GenericModel<f64>) {
    let j = serde_json::to_string_pretty(camera_model).unwrap();
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(j.as_bytes()).unwrap();
}

pub fn model_from_json(file_path: &str) -> GenericModel<f64> {
    let contents =
        std::fs::read_to_string(file_path).expect("Should have been able to read the file");
    let model: GenericModel<f64> = serde_json::from_str(&contents).unwrap();
    model
}
