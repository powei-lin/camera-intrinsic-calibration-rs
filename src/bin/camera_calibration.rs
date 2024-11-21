use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic::board::create_default_6x6_board;
use camera_intrinsic::data_loader::load_euroc;
use camera_intrinsic::visualization::*;
use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(version, about, author)]
struct CCRSCli {
    /// path to image folder
    path: String,

    /// tag_family: ["t16h5", "t25h7", "t25h9", "t36h11", "t36h11b1"]
    #[arg(value_enum, default_value = "t36h11")]
    tag_family: TagFamily,
}

fn main() {
    let cli = CCRSCli::parse();
    let detector = TagDetector::new(&cli.tag_family, None);
    let board = create_default_6x6_board();
    let dataset_root = &cli.path;
    let now = Instant::now();
    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save("output.rrd")
        .unwrap();
    let detected_feature_frames = load_euroc(dataset_root, &detector, &board, Some(&recording));
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!(
        "avg: {} sec",
        duration_sec / detected_feature_frames.len() as f64
    );
    for f in &detected_feature_frames {
        let (pts, colors_labels): (Vec<_>, Vec<_>) = f
            .features
            .iter()
            .map(|(id, p)| {
                let color = id_to_color(*id as usize);
                (
                    (p.p2d.x, p.p2d.y),
                    (color, format!("{:?}", p.p3d).to_string()),
                )
            })
            .unzip();
        let (colors, labels): (Vec<_>, Vec<_>) = colors_labels.iter().cloned().unzip();
        let pts = rerun_shift(&pts);

        let topic = "/cam0";
        recording.set_time_nanos("stable", f.time_ns);
        recording
            .log(
                format!("{}/pts", topic),
                &rerun::Points2D::new(pts)
                    .with_colors(colors)
                    .with_labels(labels)
                    .with_radii([rerun::Radius::new_ui_points(5.0)]),
            )
            .unwrap();
    }
}
