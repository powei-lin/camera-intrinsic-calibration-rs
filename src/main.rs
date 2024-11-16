use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily::T36H11;
use camera_intrinsic::board::create_default_6x6_board;
use camera_intrinsic::data_loader::load_euroc;
use camera_intrinsic::visualization::*;
use std::time::Instant;

fn main() {
    let detector = TagDetector::new(&T36H11, None);
    let board = create_default_6x6_board();
    let dataset_root = "/Users/powei/Documents/dataset/EuRoC/calibration/";
    let now = Instant::now();
    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .spawn()
        .unwrap();
    let detected_feature_frames = load_euroc(dataset_root, &detector, &board, Some(&recording));
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!(
        "avg: {} sec",
        duration_sec / detected_feature_frames.len() as f64
    );
    for f in &detected_feature_frames {
        let (pts, colors): (Vec<_>, Vec<_>) = f
            .features
            .iter()
            .map(|(id, p)| {
                let color = id_to_color(*id as usize);
                ((p.p2d.x, p.p2d.y), color)
            })
            .unzip();
        let pts = rerun_shift(&pts);

        let topic = "/cam0";
        recording.set_time_nanos("stable", f.time_ns);
        recording
            .log(
                format!("{}/pts", topic),
                &rerun::Points2D::new(pts)
                    .with_colors(colors)
                    .with_radii([rerun::Radius::new_ui_points(5.0)]),
            )
            .unwrap();
    }
}
