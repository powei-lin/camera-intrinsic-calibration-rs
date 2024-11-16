use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily::T36H11;
use camera_intrinsic::board::create_default_6x6_board;
use camera_intrinsic::data_loader::load_euroc;
use std::time::Instant;

fn main() {
    let detector = TagDetector::new(&T36H11, None);
    let board = create_default_6x6_board();
    let dataset_root = "/Users/powei/Documents/dataset/EuRoC/calibration/";
    let now = Instant::now();
    let detected_feature_frames = load_euroc(&dataset_root, &detector, &board);
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!(
        "avg: {} sec",
        duration_sec / detected_feature_frames.len() as f64
    );
}
