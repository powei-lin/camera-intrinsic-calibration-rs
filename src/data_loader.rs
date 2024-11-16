use std::collections::HashMap;

use crate::board;
use crate::detected_points::FeaturePoint;
use aprilgrid::detector::TagDetector;
use glam::Vec2;
use glob::glob;
use image::ImageReader;
use rayon::prelude::*;

pub fn load_euroc(
    root_folder: &str,
    tag_detector: &TagDetector,
    board: &board::Board,
) -> Vec<HashMap<u32, FeaturePoint>> {
    let img_paths = glob(format!("{}/mav0/cam0/data/*.png", root_folder).as_str()).expect("failed");
    img_paths
        .par_bridge()
        .map(|path| {
            let img = ImageReader::open(path.unwrap()).unwrap().decode().unwrap();
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
            tags_expand_ids
        })
        .collect()
}
