use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic_calibration::board::Board;
use camera_intrinsic_calibration::board::{
    board_config_from_json, board_config_to_json, BoardConfig,
};
use camera_intrinsic_calibration::data_loader::{load_euroc, load_general};
use camera_intrinsic_calibration::optimization::*;
use camera_intrinsic_calibration::types::RvecTvec;
use camera_intrinsic_calibration::util::*;
use camera_intrinsic_calibration::visualization::*;
use camera_intrinsic_model::*;
use clap::{Parser, ValueEnum};
use log::trace;
use std::time::Instant;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DatasetFormat {
    Euroc,
    General,
}

#[derive(Parser)]
#[command(version, about, author)]
struct CCRSCli {
    /// path to image folder
    path: String,

    /// tag_family: ["t16h5", "t25h7", "t25h9", "t36h11", "t36h11b1"]
    #[arg(long, value_enum, default_value = "t36h11")]
    tag_family: TagFamily,

    /// model: ["ucm", "eucm", "kb4", "opencv5", "eucmt", "ftheta"]
    #[arg(short, long, value_enum, default_value = "eucm")]
    model: GenericModel<f64>,

    #[arg(long, default_value_t = 0)]
    start_idx: usize,

    #[arg(long, default_value_t = 1)]
    step: usize,

    #[arg(long, default_value_t = 600)]
    max_images: usize,

    #[arg(long, default_value_t = 1)]
    cam_num: usize,

    #[arg(long)]
    board_config: Option<String>,

    #[arg(short, long, default_value = "output.json")]
    output_json: String,

    #[arg(long, value_enum, default_value = "euroc")]
    dataset_format: DatasetFormat,

    #[arg(long, action)]
    one_focal: bool,

    #[arg(long, default_value_t = 0)]
    disabled_distortion_num: usize,
}

fn main() {
    env_logger::init();
    let cli = CCRSCli::parse();
    let detector = TagDetector::new(&cli.tag_family, None);
    let board = if let Some(board_config_path) = cli.board_config {
        Board::from_config(&board_config_from_json(&board_config_path))
    } else {
        let config = BoardConfig::default();
        board_config_to_json("default_board_config.json", &config);
        Board::from_config(&config)
    };
    let dataset_root = &cli.path;
    let now = Instant::now();
    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save("output.rrd")
        .unwrap();
    trace!("Start loading data");
    let mut cams_detected_feature_frames = match cli.dataset_format {
        DatasetFormat::Euroc => load_euroc(
            dataset_root,
            &detector,
            &board,
            cli.start_idx,
            cli.step,
            cli.cam_num,
            Some(&recording),
        ),
        DatasetFormat::General => load_general(
            dataset_root,
            &detector,
            &board,
            cli.start_idx,
            cli.step,
            cli.cam_num,
            Some(&recording),
        ),
    };
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!("total: {} images", cams_detected_feature_frames[0].len());
    cams_detected_feature_frames[0].truncate(cli.max_images);
    println!(
        "avg: {} sec",
        duration_sec / cams_detected_feature_frames[0].len() as f64
    );
    for (cam_idx, feature_frames) in cams_detected_feature_frames.iter().enumerate() {
        let topic = format!("/cam{}", cam_idx);
        log_feature_frames(&recording, &topic, feature_frames);
    }
    let (frame0, frame1) = find_best_two_frames(&cams_detected_feature_frames[0]);

    let frame_feature0 = &cams_detected_feature_frames[0][frame0].clone().unwrap();
    let frame_feature1 = &cams_detected_feature_frames[0][frame1].clone().unwrap();

    let key_frames = vec![Some(frame_feature0.clone()), Some(frame_feature1.clone())];
    log_feature_frames(&recording, "/cam0/key", &key_frames);

    // initialize focal length and undistorted p2d for init poses
    let (lambda, h_mat) = radial_distortion_homography(frame_feature0, frame_feature1);
    // focal
    let f_option = homography_to_focal(&h_mat);
    if f_option.is_none() {
        return;
    }
    let focal = f_option.unwrap();
    println!("focal {}", focal);

    // poses
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
    println!("Initialized {:?}", initial_camera);
    let mut final_model = cli.model;
    final_model.set_w_h(
        initial_camera.width().round() as u32,
        initial_camera.height().round() as u32,
    );
    convert_model(
        &initial_camera,
        &mut final_model,
        cli.disabled_distortion_num,
    );
    println!("Converted {:?}", final_model);

    let (final_result, _rtvec_list) = calib_camera(
        &cams_detected_feature_frames[0],
        &final_model,
        cli.one_focal,
        cli.disabled_distortion_num,
    );
    println!("Final {:?}", final_result);
    model_to_json(&cli.output_json, &final_result);
}
