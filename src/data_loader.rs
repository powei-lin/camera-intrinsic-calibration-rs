use std::collections::HashMap;
use std::path::Path;

use crate::board::{self, Board};
use crate::detected_points::{FeaturePoint, FrameFeature};
use crate::visualization::log_image_as_compressed;
use aprilgrid::detector::TagDetector;
use glam::Vec2;
use glob::glob;
use image::{DynamicImage, ImageReader};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

const MIN_CORNERS: usize = 24;

fn path_to_timestamp(path: &Path) -> i64 {
    let time_ns: i64 = path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap_or(0);
    time_ns
}

fn image_to_option_feature_frame(
    tag_detector: &TagDetector,
    img: &DynamicImage,
    board: &Board,
    min_corners: usize,
    time_ns: i64,
) -> Option<FrameFeature> {
    let detected_tag = tag_detector.detect(img);
    let tags_expand_ids: HashMap<u32, FeaturePoint> = detected_tag
        .iter()
        .flat_map(|(k, v)| {
            v.iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    let id = k * 4 + i as u32;
                    if let Some(p3d) = board.id_to_3d.get(&id) {
                        let p2d = Vec2::new(p.0, p.1);
                        Some((id, FeaturePoint { p2d, p3d: *p3d }))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    if tags_expand_ids.len() < min_corners {
        None
    } else {
        Some(FrameFeature {
            time_ns,
            img_w_h: (img.width(), img.height()),
            features: tags_expand_ids,
        })
    }
}

fn img_filter(rp: glob::GlobResult) -> Option<std::path::PathBuf> {
    if let Ok(p) = rp {
        for ext in &[".png", ".jpg"] {
            if p.as_os_str().to_string_lossy().ends_with(ext) {
                return Some(p);
            }
        }
    }
    None
}

pub fn load_euroc(
    root_folder: &str,
    tag_detector: &TagDetector,
    board: &board::Board,
    start_idx: usize,
    step: usize,
    cam_num: usize,
    recording_option: Option<&rerun::RecordingStream>,
) -> Vec<Vec<Option<FrameFeature>>> {
    (0..cam_num)
        .map(|cam_idx| {
            log::trace!("loading cam{}", cam_idx);
            let img_paths = glob(format!("{}/mav0/cam{}/data/*", root_folder, cam_idx).as_str())
                .expect("failed");
            let mut sorted_path: Vec<std::path::PathBuf> =
                img_paths.into_iter().filter_map(img_filter).collect();

            sorted_path.sort();
            let new_paths: Vec<_> = sorted_path.iter().skip(start_idx).step_by(step).collect();
            let mut time_frame: Vec<_> = new_paths
                .iter()
                .par_bridge()
                .progress_count(new_paths.len() as u64)
                .map(|path| {
                    let time_ns = path_to_timestamp(path);
                    let img = ImageReader::open(path).unwrap().decode().unwrap();
                    if let Some(recording) = recording_option {
                        recording.set_time_nanos("stable", time_ns);
                        let topic = format!("/cam{}", cam_idx);
                        log_image_as_compressed(recording, &topic, &img, image::ImageFormat::Jpeg);
                    };
                    (
                        time_ns,
                        image_to_option_feature_frame(
                            tag_detector,
                            &img,
                            board,
                            MIN_CORNERS,
                            time_ns,
                        ),
                    )
                })
                .collect();
            time_frame.sort_by(|a, b| a.0.cmp(&b.0));
            time_frame.iter().map(|f| f.1.clone()).collect()
        })
        .collect()
}

pub fn load_others(
    root_folder: &str,
    tag_detector: &TagDetector,
    board: &board::Board,
    start_idx: usize,
    step: usize,
    cam_num: usize,
    recording_option: Option<&rerun::RecordingStream>,
) -> Vec<Vec<Option<FrameFeature>>> {
    (0..cam_num)
        .map(|cam_idx| {
            let img_paths =
                glob(format!("{}/**/cam{}/**/*", root_folder, cam_idx).as_str()).expect("failed");
            log::trace!("loading cam{}", cam_idx);
            let mut sorted_path: Vec<std::path::PathBuf> =
                img_paths.into_iter().filter_map(img_filter).collect();

            sorted_path.sort();
            let new_paths: Vec<_> = sorted_path
                .iter()
                .skip(start_idx)
                .step_by(step)
                .enumerate()
                .collect();
            let mut time_frame: Vec<_> = new_paths
                .iter()
                .par_bridge()
                .progress_count(new_paths.len() as u64)
                .map(|(idx, path)| {
                    let time_ns = *idx as i64 * 100000000;
                    let img = ImageReader::open(path).unwrap().decode().unwrap();
                    if let Some(recording) = recording_option {
                        recording.set_time_nanos("stable", time_ns);
                        let topic = format!("/cam{}", cam_idx);
                        log_image_as_compressed(recording, &topic, &img, image::ImageFormat::Jpeg);
                    };
                    (
                        time_ns,
                        image_to_option_feature_frame(
                            tag_detector,
                            &img,
                            board,
                            MIN_CORNERS,
                            time_ns,
                        ),
                    )
                })
                .collect();
            time_frame.sort_by(|a, b| a.0.cmp(&b.0));
            time_frame.iter().map(|f| f.1.clone()).collect()
        })
        .collect()
}
