use aprilgrid::TagFamily;
use aprilgrid::detector::TagDetector;
use camera_intrinsic_calibration::board::Board;
use camera_intrinsic_calibration::board::BoardConfig;
use camera_intrinsic_calibration::data_loader::{load_euroc, load_others};
use camera_intrinsic_calibration::detected_points::FrameFeature;
use camera_intrinsic_calibration::io::{object_from_json, object_to_json, write_report};
use camera_intrinsic_calibration::types::{CalibParams, Extrinsics, RvecTvec, ToRvecTvec};
use camera_intrinsic_calibration::util::*;
use camera_intrinsic_calibration::visualization::*;
use camera_intrinsic_model::*;
use clap::{Parser, ValueEnum};
use log::trace;
use std::collections::BTreeMap;
use std::collections::HashMap;
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
    let board = setup_board(&cli);
    let output_folder = setup_output_folder(&cli);

    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save(format!("{}/logging.rrd", output_folder))
        .unwrap();
    recording
        .log_static("/", &rerun::ViewCoordinates::RDF())
        .unwrap();

    let cams_detected_feature_frames = load_feature_data(&cli, &detector, &board, &recording);

    let (calibrated_intrinsics, cam_rtvecs) =
        calibrate_all_cameras(&cli, &cams_detected_feature_frames, &recording);

    let t_cam_i_0_init = init_camera_extrinsic(&cam_rtvecs);

    save_and_validate_results(
        &cli,
        &output_folder,
        &cams_detected_feature_frames,
        &calibrated_intrinsics,
        &cam_rtvecs,
        &t_cam_i_0_init,
        &recording,
    );
}

/// Loads the board configuration specified in the CLI arguments or creates a default one.
///
/// If a config file path is provided via `--board-config`, it loads from that file.
/// Otherwise, it creates a default 6x6 AprilGrid configuration and saves it to `default_board_config.json`.
fn setup_board(cli: &CCRSCli) -> Board {
    if let Some(board_config_path) = &cli.board_config {
        Board::from_config(&object_from_json(board_config_path))
    } else {
        let config = BoardConfig::default();
        object_to_json("default_board_config.json", &config);
        Board::from_config(&config)
    }
}

/// Sets up the output directory for calibration results.
///
/// If `--output-folder` is specified, uses that path.
/// Otherwise, creates a directory named with the current timestamp under `results/`.
/// Ensures the directory exists.
fn setup_output_folder(cli: &CCRSCli) -> String {
    let output_folder = if let Some(output_folder) = &cli.output_folder {
        output_folder.clone()
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
    output_folder
}

/// Loads feature data from the dataset.
///
/// Supports Euroc and General dataset formats.
/// Uses the provided tag detector and board configuration to extract features.
/// Logs images to Rerun if enabled.
///
/// # Returns
/// A vector of vectors, where each inner vector contains `Option<FrameFeature>` for a camera.
fn load_feature_data(
    cli: &CCRSCli,
    detector: &TagDetector,
    board: &Board,
    recording: &rerun::RecordingStream,
) -> Vec<Vec<Option<FrameFeature>>> {
    trace!("Start loading data");
    println!("Start loading images and detecting charts.");
    let now = Instant::now();
    let mut cams_detected_feature_frames: Vec<Vec<Option<FrameFeature>>> = match cli.dataset_format
    {
        DatasetFormat::Euroc => load_euroc(
            &cli.path,
            detector,
            board,
            cli.start_idx,
            cli.step,
            cli.cam_num,
            Some(recording),
        ),
        DatasetFormat::General => load_others(
            &cli.path,
            detector,
            board,
            cli.start_idx,
            cli.step,
            cli.cam_num,
            Some(recording),
        ),
    };
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    if !cams_detected_feature_frames.is_empty() {
        println!("total: {} images", cams_detected_feature_frames[0].len());
        println!(
            "avg: {} sec",
            duration_sec / cams_detected_feature_frames[0].len() as f64
        );
    }

    cams_detected_feature_frames
        .iter_mut()
        .for_each(|f| f.truncate(cli.max_images));

    cams_detected_feature_frames
}

/// Calibrates all cameras individually.
///
/// Iterates through each camera, detecting features and running the optimization.
/// Retries calibration up to 3 times if it fails.
///
/// # Returns
/// A tuple containing:
/// - `Vec<GenericModel<f64>>`: The calibrated intrinsic models for each camera.
/// - `Vec<HashMap<usize, RvecTvec>>`: estimated camera poses for each frame.
fn calibrate_all_cameras(
    cli: &CCRSCli,
    cams_detected_feature_frames: &[Vec<Option<FrameFeature>>],
    recording: &rerun::RecordingStream,
) -> (Vec<GenericModel<f64>>, Vec<HashMap<usize, RvecTvec>>) {
    cams_detected_feature_frames
        .iter()
        .enumerate()
        .map(|(cam_idx, feature_frames)| {
            let topic = format!("/cam{}", cam_idx);
            log_feature_frames(recording, &topic, feature_frames);
            let mut calibrated_result: Option<(GenericModel<f64>, HashMap<usize, RvecTvec>)> = None;
            let max_trials = 3;
            let cam0_fixed_focal = if cam_idx == 0 { cli.fixed_focal } else { None };
            let calib_params = CalibParams {
                fixed_focal: cam0_fixed_focal,
                disabled_distortion_num: cli.disabled_distortion_num,
                one_focal: cli.one_focal,
            };
            for trial in 0..max_trials {
                calibrated_result = init_and_calibrate_one_camera(
                    cam_idx,
                    cams_detected_feature_frames,
                    &cli.model,
                    recording,
                    &calib_params,
                    trial > 0,
                );
                if calibrated_result.is_some() {
                    break;
                }
            }
            if calibrated_result.is_none() {
                panic!(
                    "Failed to calibrate cam{} after {} times",
                    cam_idx, max_trials
                );
            }
            calibrated_result.unwrap()
        })
        .unzip()
}

/// Saves calibration results and performs validation.
///
/// saves intrinsics, extrinsics, and pose data to JSON files.
/// Generates a validation report and logs visualization data to Rerun.
/// If multiple cameras are present, it also attempts to calibrate extrinsics between cameras.
#[allow(clippy::too_many_arguments)]
fn save_and_validate_results(
    cli: &CCRSCli,
    output_folder: &str,
    cams_detected_feature_frames: &[Vec<Option<FrameFeature>>],
    intrinsics: &[GenericModel<f64>],
    cam_rtvecs: &[HashMap<usize, RvecTvec>],
    t_cam_i_0_init: &[RvecTvec],
    recording: &rerun::RecordingStream,
) {
    for t in t_cam_i_0_init {
        println!("r {} t {}", t.na_rvec(), t.na_tvec());
    }

    if let Some((camera_intrinsics, t_i_0, board_rtvecs)) = calib_all_camera_with_extrinsics(
        intrinsics,
        t_cam_i_0_init,
        cam_rtvecs,
        cams_detected_feature_frames,
        cli.one_focal || cli.fixed_focal.is_some(),
        cli.disabled_distortion_num,
        cli.fixed_focal.is_some(),
    ) {
        let mut rep_rms = Vec::new();
        for (cam_idx, intrinsic) in camera_intrinsics.iter().enumerate() {
            model_to_json(&format!("{}/cam{}.json", output_folder, cam_idx), intrinsic);
            let new_rtvec_map: HashMap<usize, RvecTvec> = board_rtvecs
                .iter()
                .map(|(k, t_0_b)| {
                    (
                        *k,
                        (t_i_0[cam_idx].to_na_isometry3() * t_0_b.to_na_isometry3()).to_rvec_tvec(),
                    )
                })
                .collect();
            object_to_json(
                &format!("{}/cam{}_poses.json", output_folder, cam_idx),
                &new_rtvec_map
                    .iter()
                    .collect::<BTreeMap<&usize, &RvecTvec>>(),
            );
            let cam_transform =
                na_isometry3_to_rerun_transform3d(&t_i_0[cam_idx].to_na_isometry3().inverse())
                    .with_axis_length(0.1);
            recording
                .log_static(format!("/cam{}", cam_idx), &cam_transform)
                .unwrap();
            let rep = validation(
                cam_idx,
                intrinsic,
                &new_rtvec_map,
                &cams_detected_feature_frames[cam_idx],
                Some(recording),
            );
            rep_rms.push(rep);
            println!(
                "Cam {} final params with extrinsic{}",
                cam_idx,
                serde_json::to_string_pretty(intrinsic).unwrap()
            );
        }
        write_report(&format!("{}/report.txt", output_folder), true, &rep_rms);

        object_to_json(
            &format!("{}/extrinsics.json", output_folder),
            &Extrinsics::new(&t_i_0),
        );
    } else {
        let mut rep_rms = Vec::new();
        for (cam_idx, (intrinsic, rtvec_map)) in intrinsics.iter().zip(cam_rtvecs).enumerate() {
            let rep = validation(
                cam_idx,
                intrinsic,
                rtvec_map,
                &cams_detected_feature_frames[cam_idx],
                Some(recording),
            );
            rep_rms.push(rep);
            println!(
                "Cam {} final params{}",
                cam_idx,
                serde_json::to_string_pretty(intrinsic).unwrap()
            );
            model_to_json(&format!("{}/cam{}.json", output_folder, cam_idx), intrinsic);
            object_to_json(
                &format!("{}/cam{}_poses.json", output_folder, cam_idx),
                &rtvec_map.iter().collect::<BTreeMap<&usize, &RvecTvec>>(),
            );
        }
        write_report(&format!("{}/report.txt", output_folder), false, &rep_rms);
    }
}
