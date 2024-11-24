use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic::board::create_default_6x6_board;
use camera_intrinsic::data_loader::load_euroc;
use camera_intrinsic::detected_points::{FeaturePoint, FrameFeature};
use camera_intrinsic::visualization::*;
use clap::Parser;
use log::trace;
use rerun::RecordingStream;
use std::collections::HashMap;
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

fn log_frames(recording: &RecordingStream, detected_feature_frames: &[FrameFeature]) {
    for f in detected_feature_frames {
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

fn features_avg_center(features: &HashMap<u32, FeaturePoint>) -> glam::Vec2 {
    features
        .iter()
        .map(|(_, p)| p.p2d)
        .reduce(|acc, e| acc + e)
        .unwrap()
        / features.len() as f32
}

fn vec2_distance2(v0: &glam::Vec2, v1: &glam::Vec2) -> f32 {
    let v = v0 - v1;
    v.x * v.x + v.y * v.y
}

fn find_best_two_frames(detected_feature_frames: &[FrameFeature]) -> (usize, usize) {
    let mut max_detection = 0;
    let mut max_detection_idxs = Vec::new();
    for (i, f) in detected_feature_frames.iter().enumerate() {
        if f.features.len() > max_detection {
            max_detection = f.features.len();
            max_detection_idxs = vec![i];
        } else if f.features.len() == max_detection {
            max_detection_idxs.push(i);
        }
    }
    let mut v: Vec<_> = max_detection_idxs
        .iter()
        .map(|i| {
            let p_avg = features_avg_center(&detected_feature_frames[*i].features);
            (i, p_avg)
        })
        .collect();

    let avg_all = v.iter().map(|(_, p)| *p).reduce(|acc, e| acc + e).unwrap() / v.len() as f32;
    v.sort_by(|a, b| {
        vec2_distance2(&a.1, &avg_all)
            .partial_cmp(&vec2_distance2(&b.1, &avg_all))
            .unwrap()
    });
    (*v[0].0, *v.last().unwrap().0)
}

fn main() {
    env_logger::init();
    let cli = CCRSCli::parse();
    let detector = TagDetector::new(&cli.tag_family, None);
    let board = create_default_6x6_board();
    let dataset_root = &cli.path;
    let now = Instant::now();
    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save("output.rrd")
        .unwrap();
    trace!("Start loading data");
    let detected_feature_frames = load_euroc(dataset_root, &detector, &board, Some(&recording));
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!(
        "avg: {} sec",
        duration_sec / detected_feature_frames.len() as f64
    );
    // log_frames(&recording, &detected_feature_frames);
    let (frame0, frame1) = find_best_two_frames(&detected_feature_frames);
    let key_frames = vec![
        detected_feature_frames[frame0].clone(),
        detected_feature_frames[frame1].clone(),
    ];
    log_frames(&recording, &key_frames);
}
