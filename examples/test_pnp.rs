use std::{collections::HashMap, f64::consts::PI};

use aprilgrid::detector::TagDetector;
use camera_intrinsic::{
    board::create_default_6x6_board,
    camera_model::{remap, GenericModel, UCM},
    detected_points::FeaturePoint,
};
use glam::Vec2;
use image::ImageReader;
use nalgebra as na;

fn main() {
    env_logger::init();
    let params = na::dvector![471.019, 470.243, 367.122, 246.741, 0.67485];
    let model = GenericModel::UCM(UCM::new(&params, 752, 480));

    let img = ImageReader::open("data/euroc.png")
        .unwrap()
        .decode()
        .unwrap();
    let board = create_default_6x6_board();

    let detector = TagDetector::new(&aprilgrid::TagFamily::T36H11, None);
    let detected_tag = detector.detect(&img);
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
    let (p3ds, p2ds): (Vec<_>, Vec<_>) = tags_expand_ids
        .iter()
        .map(|p| {
            (
                p.1.p3d,
                na::Vector2::new(p.1.p2d.x as f64, p.1.p2d.y as f64),
            )
        })
        .collect();
    let undistorted = model.unproject(&p2ds);

    let (p3ds, p2ds_z): (Vec<_>, Vec<_>) = undistorted
        .iter()
        .zip(p3ds)
        .filter_map(|(p2, p3)| {
            if let Some(p2) = p2 {
                Some((p3, glam::Vec2::new(p2.x as f32, p2.y as f32)))
            } else {
                None
            }
        })
        .unzip();
    let (r, t) = sqpnp_simple::sqpnp_solve_glam(&p3ds, &p2ds_z).unwrap();
    println!("r {:?}", r);
    println!("t {:?}", t);
    let rt = na::Isometry3::new(
        na::Vector3::new(t.0, t.1, t.2),
        na::Vector3::new(r.0, r.1, r.2),
    );

    // println!("{}", *yy)
    p3ds.iter().zip(p2ds_z).for_each(|(p3, p2)| {
        let p33 = rt * na::Point3::new(p3.x, p3.y, p3.z).cast();
        println!("{}", p33);
        println!("{}", p2);
        println!("{}", p33 / p33.z);
        println!("");
    });
    let new_w_h = 1024;
    let p = model.estimate_new_camera_matrix_for_undistort(1.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model.init_undistort_map(&p, (new_w_h, new_w_h));
    let remaped = remap(&img, &xmap, &ymap);
    remaped.save("remaped_euroc.png").unwrap()
}
