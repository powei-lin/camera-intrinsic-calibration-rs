use std::collections::HashMap;
use std::path::PathBuf;

use crate::board;
use crate::detected_points::{FeaturePoint, FrameFeature};
use aprilgrid::detector::TagDetector;
use glam::Vec2;
use glob::glob;
use image::ImageReader;
use rayon::prelude::*;

fn path_to_timestamp(path: &PathBuf) -> i64 {
    let time_ns: i64 = path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap_or(0);
    time_ns
}

pub fn load_euroc(
    root_folder: &str,
    tag_detector: &TagDetector,
    board: &board::Board,
) -> Vec<FrameFeature> {
    let img_paths = glob(format!("{}/mav0/cam0/data/*.png", root_folder).as_str()).expect("failed");
    img_paths
        .par_bridge()
        .map(|path| {
            let path = path.unwrap();
            let time_ns = path_to_timestamp(&path);
            let img = ImageReader::open(&path).unwrap().decode().unwrap();
            let detected_tag = tag_detector.detect(&img);
            let tags_expand_ids: HashMap<u32, FeaturePoint> = detected_tag
                .iter()
                .flat_map(|(k, v)| {
                    v.iter()
                        .enumerate()
                        .filter_map(|(i, p)| {
                            let id = k * 4 + i as u32;
                            if let Some(p3d) = board.id_to_3d.get(k) {
                                let p2d = Vec2::new(p.0, p.1);
                                Some((id, FeaturePoint { p2d, p3d: *p3d }))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            FrameFeature {
                time_ns,
                features: tags_expand_ids,
            }
        })
        .collect()
}
