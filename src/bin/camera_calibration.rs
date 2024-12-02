use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic_calibration::board::create_default_6x6_board;
use camera_intrinsic_calibration::data_loader::load_euroc;
use camera_intrinsic_calibration::optimization::*;
use camera_intrinsic_calibration::types::RvecTvec;
use camera_intrinsic_calibration::util::*;
use camera_intrinsic_calibration::visualization::*;
use camera_intrinsic_model::*;
use clap::Parser;
use log::trace;
use std::time::Instant;

#[derive(Parser)]
#[command(version, about, author)]
struct CCRSCli {
    /// path to image folder
    path: String,

    /// tag_family: ["t16h5", "t25h7", "t25h9", "t36h11", "t36h11b1"]
    #[arg(long, value_enum, default_value = "t36h11")]
    tag_family: TagFamily,

    /// model: ["ucm", "eucm", "kb4", "opencv5"]
    #[arg(short, long, value_enum, default_value = "eucm")]
    model: GenericModel<f64>,

    #[arg(long, default_value_t = 0)]
    start_idx: usize,

    #[arg(long, default_value_t = 1)]
    step: usize,

    #[arg(long, default_value_t = 600)]
    max_images: usize,

    #[arg(short, long, default_value = "output.json")]
    output_json: String,
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
    let mut detected_feature_frames = load_euroc(
        dataset_root,
        &detector,
        &board,
        cli.start_idx,
        cli.step,
        Some(&recording),
    );
    detected_feature_frames.truncate(cli.max_images);
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!(
        "avg: {} sec",
        duration_sec / detected_feature_frames.len() as f64
    );
    log_frames(&recording, &detected_feature_frames);
    let (frame0, frame1) = find_best_two_frames(&detected_feature_frames);
    let key_frames = vec![
        detected_feature_frames[frame0].clone(),
        detected_feature_frames[frame1].clone(),
    ];
    log_frames(&recording, &key_frames);

    // initialize focal length and undistorted p2d for init poses
    let (lambda, h_mat) = radial_distortion_homography(
        &detected_feature_frames[frame0],
        &detected_feature_frames[frame1],
    );
    // focal
    let f_option = homography_to_focal(&h_mat);
    if f_option.is_none() {
        return;
    }
    let focal = f_option.unwrap();
    println!("focal {}", focal);

    // poses
    let frame_feature0 = &detected_feature_frames[frame0];
    let frame_feature1 = &detected_feature_frames[frame1];
    let (rvec0, tvec0) = rtvec_to_na_dvec(init_pose(frame_feature0, lambda));
    let (rvec1, tvec1) = rtvec_to_na_dvec(init_pose(frame_feature1, lambda));
    let rtvec0 = RvecTvec::new(rvec0, tvec0);
    let rtvec1 = RvecTvec::new(rvec1, tvec1);

    let half_w = frame_feature0.img_w_h.0 as f64 / 2.0;
    let half_h = frame_feature0.img_w_h.1 as f64 / 2.0;
    let half_img_size = half_h.max(half_w);
    let init_f = focal as f64 * half_img_size;
    println!("init f {}", init_f);
    let init_alpha = lambda.abs() as f64;
    let initial_camera = init_ucm(
        frame_feature0,
        frame_feature1,
        &rtvec0,
        &rtvec1,
        init_f,
        init_alpha,
    );
    let mut final_model = cli.model;
    final_model.set_w_h(
        initial_camera.width().round() as u32,
        initial_camera.height().round() as u32,
    );
    println!("{:?}", final_model);
    convert_model(&initial_camera, &mut final_model);
    println!("{:?}", final_model);

    let (final_result, _rtvec_list) = calib_camera(&detected_feature_frames, &final_model);
    println!("{:?}", final_result);
    model_to_json(&cli.output_json, &final_result);
}
