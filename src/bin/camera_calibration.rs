use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic_calibration::board::Board;
use camera_intrinsic_calibration::board::{
    board_config_from_json, board_config_to_json, BoardConfig,
};
use camera_intrinsic_calibration::data_loader::{load_euroc, load_others};
use camera_intrinsic_calibration::util::*;
use camera_intrinsic_calibration::visualization::*;
use camera_intrinsic_model::*;
use clap::{Parser, ValueEnum};
use log::trace;
use std::time::Instant;
use time::OffsetDateTime;

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

    #[arg(short, long)]
    output_folder: Option<String>,

    #[arg(long, value_enum, default_value = "euroc")]
    dataset_format: DatasetFormat,

    #[arg(long, action)]
    one_focal: bool,

    #[arg(long, default_value_t = 0)]
    disabled_distortion_num: usize,

    #[arg(long)]
    fixed_focal: Option<f64>,
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
    let output_folder = if let Some(output_folder) = cli.output_folder {
        output_folder
    } else {
        let now = OffsetDateTime::now_local().unwrap();
        format!(
            "results/{}{:02}{:02}_{:02}_{:02}_{:02}",
            now.year(),
            now.month() as u8,
            now.day(),
            now.hour(),
            now.minute(),
            now.second(),
        )
    };
    std::fs::create_dir_all(&output_folder).expect("Valid path");

    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save(format!("{}/logging.rrd", output_folder))
        .unwrap();
    trace!("Start loading data");
    println!("Start loading images and detecting charts.");
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
        DatasetFormat::General => load_others(
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

    let key_frames = [Some(frame_feature0.clone()), Some(frame_feature1.clone())];
    key_frames.iter().enumerate().for_each(|(i, k)| {
        let topic = format!("/cam0/keyframe{}", i);
        recording.set_time_nanos("stable", k.clone().unwrap().time_ns);
        recording
            .log(topic, &rerun::TextLog::new("keyframe"))
            .unwrap();
    });

    let mut initial_camera = GenericModel::UCM(UCM::zeros());
    for i in 0..10 {
        trace!("Initialize ucm {}", i);
        if let Some(initialized_ucm) =
            try_init_camera(frame_feature0, frame_feature1, cli.fixed_focal)
        {
            initial_camera = initialized_ucm;
            break;
        }
    }
    if initial_camera.params()[0] == 0.0 {
        println!("calibration failed.");
        return;
    }
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
    let (one_focal, fixed_focal) = if let Some(focal) = cli.fixed_focal {
        // if fixed focal then set one focal true
        let mut p = final_model.params();
        p[0] = focal;
        p[1] = focal;
        final_model.set_params(&p);
        (true, true)
    } else {
        (cli.one_focal, false)
    };

    let (final_result, rtvec_list) = calib_camera(
        &cams_detected_feature_frames[0],
        &final_model,
        one_focal,
        cli.disabled_distortion_num,
        fixed_focal,
    );
    validation(
        &final_result,
        &rtvec_list,
        &cams_detected_feature_frames[0],
        Some(&recording),
    );
    println!(
        "Final params{}",
        serde_json::to_string_pretty(&final_result).unwrap()
    );
    model_to_json(&format!("{}/result.json", output_folder), &final_result);
}
