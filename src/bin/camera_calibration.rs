use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic_calibration::board::Board;
use camera_intrinsic_calibration::board::{
    board_config_from_json, board_config_to_json, BoardConfig,
};
use camera_intrinsic_calibration::data_loader::{load_euroc, load_others};
use camera_intrinsic_calibration::detected_points::FrameFeature;
use camera_intrinsic_calibration::io::{extrinsics_to_json, write_report};
use camera_intrinsic_calibration::types::{Extrinsics, RvecTvec, ToRvecTvec};
use camera_intrinsic_calibration::util::*;
use camera_intrinsic_calibration::visualization::*;
use camera_intrinsic_model::*;
use clap::{Parser, ValueEnum};
use log::trace;
use rerun::RecordingStream;
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
    recording
        .log_static("/", &rerun::ViewCoordinates::RDF)
        .unwrap();
    trace!("Start loading data");
    println!("Start loading images and detecting charts.");
    let mut cams_detected_feature_frames: Vec<Vec<Option<FrameFeature>>> = match cli.dataset_format
    {
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
    cams_detected_feature_frames
        .iter_mut()
        .for_each(|f| f.truncate(cli.max_images));
    println!(
        "avg: {} sec",
        duration_sec / cams_detected_feature_frames[0].len() as f64
    );
    let (calibrated_intrinsics, cam_rtvecs): (Vec<_>, Vec<_>) = cams_detected_feature_frames
        .iter()
        .enumerate()
        .map(|(cam_idx, feature_frames)| {
            let topic = format!("/cam{}", cam_idx);
            log_feature_frames(&recording, &topic, feature_frames);
            let mut calibrated_result: Option<(GenericModel<f64>, HashMap<usize, RvecTvec>)> = None;
            let max_trials = 3;
            let cam0_fixed_focal = if cam_idx == 0 { cli.fixed_focal } else { None };
            for _ in 0..max_trials {
                calibrated_result = init_and_calibrate_one_camera(
                    cam_idx,
                    &cams_detected_feature_frames,
                    &cli.model,
                    &recording,
                    cam0_fixed_focal,
                    cli.disabled_distortion_num,
                    cli.one_focal,
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
            let (final_result, rtvec_map) = calibrated_result.unwrap();
            (final_result, rtvec_map)
        })
        .unzip();
    let t_cam_i_0_init = init_camera_extrinsic(&cam_rtvecs);
    for t in &t_cam_i_0_init {
        println!("r {} t {}", t.na_rvec(), t.na_tvec());
    }
    if let Some((camera_intrinsics, t_i_0, board_rtvecs)) = calib_all_camera_with_extrinsics(
        &calibrated_intrinsics,
        &t_cam_i_0_init,
        &cam_rtvecs,
        &cams_detected_feature_frames,
        cli.one_focal || cli.fixed_focal.is_some(),
        cli.disabled_distortion_num,
        cli.fixed_focal.is_some(),
    ) {
        let mut rep_rms = Vec::new();
        for (cam_idx, intrinsic) in camera_intrinsics.iter().enumerate() {
            model_to_json(
                &format!("{}/cam{}.json", output_folder, cam_idx),
                &intrinsic,
            );
            let new_rtvec_map: HashMap<usize, RvecTvec> = board_rtvecs
                .iter()
                .map(|(k, t_0_b)| {
                    (
                        *k,
                        (t_i_0[cam_idx].to_na_isometry3() * t_0_b.to_na_isometry3()).to_rvec_tvec(),
                    )
                })
                .collect();
            recording
                .log_static(
                    format!("/cam{}", cam_idx),
                    &na_isometry3_to_rerun_transform3d(&t_i_0[cam_idx].to_na_isometry3().inverse()),
                )
                .unwrap();
            let rep = validation(
                cam_idx,
                intrinsic,
                &new_rtvec_map,
                &cams_detected_feature_frames[cam_idx],
                Some(&recording),
            );
            rep_rms.push(rep);
            println!(
                "Cam {} final params with extrinsic{}",
                cam_idx,
                serde_json::to_string_pretty(intrinsic).unwrap()
            );
        }
        write_report(&format!("{}/report.txt", output_folder), true, &rep_rms);

        extrinsics_to_json(
            &format!("{}/extrinsics.json", output_folder),
            &Extrinsics::new(&t_i_0),
        );
    } else {
        let mut rep_rms = Vec::new();
        for (cam_idx, (intrinsic, rtvec_map)) in
            calibrated_intrinsics.iter().zip(cam_rtvecs).enumerate()
        {
            let rep = validation(
                cam_idx,
                intrinsic,
                &rtvec_map,
                &cams_detected_feature_frames[cam_idx],
                Some(&recording),
            );
            rep_rms.push(rep);
            println!(
                "Cam {} final params{}",
                cam_idx,
                serde_json::to_string_pretty(intrinsic).unwrap()
            );
            model_to_json(
                &format!("{}/cam{}.json", output_folder, cam_idx),
                &intrinsic,
            );
        }
        write_report(&format!("{}/report.txt", output_folder), false, &rep_rms);
    }
}
