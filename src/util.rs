use std::cmp::Ordering;
use std::collections::HashMap;

use crate::detected_points::{FeaturePoint, FrameFeature};

use super::camera_model::generic::GenericModel;
use super::camera_model::UCM;
use super::optimization::factors::*;
use log::debug;
use nalgebra as na;
use tiny_solver::loss_functions::HuberLoss;
use tiny_solver::Optimizer;

pub fn rtvec_to_na_dvec(
    rtvec: ((f64, f64, f64), (f64, f64, f64)),
) -> (na::DVector<f64>, na::DVector<f64>) {
    (
        na::dvector![rtvec.0 .0, rtvec.0 .1, rtvec.0 .2],
        na::dvector![rtvec.1 .0, rtvec.1 .1, rtvec.1 .2],
    )
}

fn set_problem_parameter_bound(
    problem: &mut tiny_solver::Problem,
    generic_camera: &GenericModel<f64>,
) {
    problem.set_variable_bounds("params", 0, 0.0, 10000.0);
    problem.set_variable_bounds("params", 1, 0.0, 10000.0);
    problem.set_variable_bounds("params", 2, 0.0, generic_camera.width());
    problem.set_variable_bounds("params", 3, 0.0, generic_camera.height());
    for (distortion_idx, (lower, upper)) in generic_camera.distortion_params_bound() {
        problem.set_variable_bounds("params", distortion_idx, lower, upper);
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
fn features_covered_area(features: &HashMap<u32, FeaturePoint>) -> f32 {
    let (xmin, ymin, xmax, ymax) = features.iter().map(|(_, p)| p.p2d).fold(
        (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
        |acc, e| {
            let xmin = acc.0.min(e.x);
            let ymin = acc.1.min(e.y);
            let xmax = acc.0.max(e.x);
            let ymax = acc.1.max(e.y);
            (xmin, ymin, xmax, ymax)
        },
    );
    (xmax - xmin) * (ymax - ymin)
}

fn vec2_distance2(v0: &glam::Vec2, v1: &glam::Vec2) -> f32 {
    let v = v0 - v1;
    v.x * v.x + v.y * v.y
}

pub fn find_best_two_frames(detected_feature_frames: &[FrameFeature]) -> (usize, usize) {
    let mut max_detection = 0;
    let mut max_detection_idxs = Vec::new();
    for (i, f) in detected_feature_frames.iter().enumerate() {
        match f.features.len().cmp(&max_detection) {
            Ordering::Greater => {
                max_detection = f.features.len();
                max_detection_idxs = vec![i];
            }
            Ordering::Less => {}
            Ordering::Equal => {
                max_detection_idxs.push(i);
            }
        }
        // if f.features.len() > max_detection {
        // } else if f.features.len() == max_detection {
        // }
    }
    let mut v0: Vec<_> = max_detection_idxs
        .iter()
        .map(|i| {
            let p_avg = features_avg_center(&detected_feature_frames[*i].features);
            (i, p_avg)
        })
        .collect();

    let avg_all = v0.iter().map(|(_, p)| *p).reduce(|acc, e| acc + e).unwrap() / v0.len() as f32;
    // let avg_all = Vec2::ZERO;
    v0.sort_by(|a, b| {
        vec2_distance2(&a.1, &avg_all)
            .partial_cmp(&vec2_distance2(&b.1, &avg_all))
            .unwrap()
    });
    let mut v1: Vec<_> = max_detection_idxs
        .iter()
        .map(|&i| {
            let area = features_covered_area(&detected_feature_frames[i].features);
            (i, area)
        })
        .collect();
    v1.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // (*v0[0].0, *v0.last().unwrap().0)
    (v1.last().unwrap().0, *v0.last().unwrap().0)
}

pub fn convert_model(source_model: &GenericModel<f64>, target_model: &mut GenericModel<f64>) {
    let mut problem = tiny_solver::Problem::new();
    let edge_pixels = source_model.width().max(source_model.height()) as u32 / 100;
    let cost = ModelConvertFactor::new(source_model, target_model, edge_pixels, 3);
    problem.add_residual_block(
        cost.residaul_num(),
        vec![("x".to_string(), target_model.params().len())],
        Box::new(cost),
        Some(Box::new(HuberLoss::new(1.0))),
    );

    let camera_params = source_model.camera_params();
    let mut target_params_init = target_model.params();
    target_params_init.rows_mut(0, 4).copy_from(&camera_params);

    let initial_values =
        HashMap::<String, na::DVector<f64>>::from([("x".to_string(), target_params_init)]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    // distortion parameter bound
    set_problem_parameter_bound(&mut problem, target_model);

    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None);

    // save result
    let result_params = result.get("x").unwrap();
    target_model.set_params(result_params);
}

pub fn init_ucm(
    frame_feature0: &FrameFeature,
    frame_feature1: &FrameFeature,
    rvec0: &na::DVector<f64>,
    tvec0: &na::DVector<f64>,
    rvec1: &na::DVector<f64>,
    tvec1: &na::DVector<f64>,
    init_f: f64,
    init_alpha: f64,
) -> GenericModel<f64> {
    let half_w = frame_feature0.img_w_h.0 as f64 / 2.0;
    let half_h = frame_feature0.img_w_h.1 as f64 / 2.0;
    let init_params = na::dvector![init_f, init_f, half_w, half_h, init_alpha];
    let ucm_init_model = GenericModel::UCM(UCM::new(
        &init_params,
        frame_feature0.img_w_h.0,
        frame_feature0.img_w_h.1,
    ));

    let mut init_focal_alpha_problem = tiny_solver::Problem::new();
    let init_f_alpha = na::dvector![init_f, init_alpha];

    for fp in frame_feature0.features.values() {
        let cost = UCMInitFocalAlphaFactor::new(&ucm_init_model, &fp.p3d, &fp.p2d);
        init_focal_alpha_problem.add_residual_block(
            2,
            vec![
                ("params".to_string(), 2),
                ("rvec0".to_string(), 3),
                ("tvec0".to_string(), 3),
            ],
            Box::new(cost),
            Some(Box::new(HuberLoss::new(1.0))),
        );
    }

    for fp in frame_feature1.features.values() {
        let cost = UCMInitFocalAlphaFactor::new(&ucm_init_model, &fp.p3d, &fp.p2d);
        init_focal_alpha_problem.add_residual_block(
            2,
            vec![
                ("params".to_string(), 2),
                ("rvec1".to_string(), 3),
                ("tvec1".to_string(), 3),
            ],
            Box::new(cost),
            Some(Box::new(HuberLoss::new(1.0))),
        );
    }

    let initial_values = HashMap::<String, na::DVector<f64>>::from([
        ("params".to_string(), init_f_alpha),
        ("rvec0".to_string(), rvec0.clone()),
        ("tvec0".to_string(), tvec0.clone()),
        ("rvec1".to_string(), rvec1.clone()),
        ("tvec1".to_string(), tvec1.clone()),
    ]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    println!("init ucm init f {}", initial_values.get("params").unwrap());
    println!("init rvec0{}", initial_values.get("rvec0").unwrap());
    println!("init tvec0{}", initial_values.get("tvec0").unwrap());
    println!("init rvec1{}", initial_values.get("rvec1").unwrap());
    println!("init tvec1{}", initial_values.get("tvec1").unwrap());

    // optimize
    init_focal_alpha_problem.set_variable_bounds("params", 0, init_f / 3.0, init_f * 3.0);
    init_focal_alpha_problem.set_variable_bounds("params", 1, 1e-6, 1.0);
    let mut second_round_values =
        optimizer.optimize(&init_focal_alpha_problem, &initial_values, None);

    println!(
        "params after {:?}\n",
        second_round_values.get("params").unwrap()
    );
    println!("after rvec0{}", second_round_values.get("rvec0").unwrap());
    println!("after tvec0{}", second_round_values.get("tvec0").unwrap());
    println!("after rvec1{}", second_round_values.get("rvec1").unwrap());
    println!("after tvec1{}", second_round_values.get("tvec1").unwrap());
    // panic!("stop");

    let focal = second_round_values["params"][0];
    let alpha = second_round_values["params"][1];
    let ucm_all_params = na::dvector![focal, focal, half_w, half_h, alpha];
    let ucm_camera = GenericModel::UCM(UCM::new(
        &ucm_all_params,
        frame_feature0.img_w_h.0,
        frame_feature0.img_w_h.1,
    ));
    second_round_values.remove("params");
    calib_camera(
        &[frame_feature0.clone(), frame_feature1.clone()],
        &ucm_camera,
    )
    .0
}

pub type RTvecList = Vec<(na::DVector<f64>, na::DVector<f64>)>;

pub fn calib_camera(
    frame_feature_list: &[FrameFeature],
    generic_camera: &GenericModel<f64>,
) -> (GenericModel<f64>, RTvecList) {
    let params = generic_camera.params();
    let params_len = params.len();
    let mut problem = tiny_solver::Problem::new();
    let mut initial_values =
        HashMap::<String, na::DVector<f64>>::from([("params".to_string(), params)]);
    debug!("init {:?}", initial_values);
    let mut valid_indexes = Vec::new();
    for (i, frame_feature) in frame_feature_list.iter().enumerate() {
        let mut p3ds = Vec::new();
        let mut p2ds = Vec::new();
        let rvec_name = format!("rvec{}", i);
        let tvec_name = format!("tvec{}", i);
        for fp in frame_feature.features.values() {
            let cost = ReprojectionFactor::new(generic_camera, &fp.p3d, &fp.p2d);
            problem.add_residual_block(
                2,
                vec![
                    ("params".to_string(), params_len),
                    (rvec_name.clone(), 3),
                    (tvec_name.clone(), 3),
                ],
                Box::new(cost),
                Some(Box::new(HuberLoss::new(1.0))),
            );
            p3ds.push(fp.p3d);
            p2ds.push(na::Vector2::new(fp.p2d.x as f64, fp.p2d.y as f64));
        }
        let undistorted = generic_camera.unproject(&p2ds);
        let (p3ds, p2ds_z): (Vec<_>, Vec<_>) = undistorted
            .iter()
            .zip(p3ds)
            .filter_map(|(p2, p3)| {
                p2.as_ref()
                    .map(|p2| (p3, glam::Vec2::new(p2.x as f32, p2.y as f32)))
            })
            .unzip();
        // if p3ds.len() < 6 {
        //     println!("skip frame {}", i);
        //     continue;
        // }
        valid_indexes.push(i);
        let (rvec, tvec) =
            rtvec_to_na_dvec(sqpnp_simple::sqpnp_solve_glam(&p3ds, &p2ds_z).unwrap());

        initial_values.entry(rvec_name).or_insert(rvec);
        initial_values.entry(tvec_name).or_insert(tvec);
    }

    let optimizer = tiny_solver::GaussNewtonOptimizer {};
    let initial_values = optimizer.optimize(&problem, &initial_values, None);

    set_problem_parameter_bound(&mut problem, generic_camera);
    let mut result = optimizer.optimize(&problem, &initial_values, None);

    let new_params = result.get("params").unwrap();
    println!("params {}", new_params);
    let mut calibrated_camera = *generic_camera;
    calibrated_camera.set_params(new_params);
    let rtvec_vec: Vec<_> = valid_indexes
        .iter()
        .map(|&i| {
            let rvec_name = format!("rvec{}", i);
            let tvec_name = format!("tvec{}", i);
            (
                result.remove(&rvec_name).unwrap(),
                result.remove(&tvec_name).unwrap(),
            )
        })
        .collect();
    (calibrated_camera, rtvec_vec)
}
