use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::detected_points::{FeaturePoint, FrameFeature};
use crate::optimization::{homography_to_focal, init_pose, radial_distortion_homography};
use crate::types::RvecTvec;
use crate::visualization::rerun_shift;

use super::optimization::factors::*;
use super::types::{DVecVec3, Vec3DVec};
use camera_intrinsic_model::*;
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
    xy_same_focal: bool,
) {
    let shift = if xy_same_focal { 1 } else { 0 };
    problem.set_variable_bounds("params", 0, 0.0, 10000.0);
    problem.set_variable_bounds("params", 1 - shift, 0.0, 10000.0);
    problem.set_variable_bounds("params", 2 - shift, 0.0, generic_camera.width());
    problem.set_variable_bounds("params", 3 - shift, 0.0, generic_camera.height());
    for (distortion_idx, (lower, upper)) in generic_camera.distortion_params_bound() {
        log::trace!(
            "set params bound {} {} {}",
            distortion_idx - shift,
            lower,
            upper
        );
        problem.set_variable_bounds("params", distortion_idx - shift, lower, upper);
    }
}
fn set_problem_parameter_disabled(
    problem: &mut tiny_solver::Problem,
    init_values: &mut HashMap<String, na::DVector<f64>>,
    generic_camera: &GenericModel<f64>,
    xy_same_focal: bool,
    disabled_distortions: usize,
) {
    let shift = if xy_same_focal { 1 } else { 0 };
    for i in 0..disabled_distortions {
        let distortion_idx = generic_camera.params().len() - 1 - shift - i;
        problem.fix_variable("params", distortion_idx);
        let params = init_values.get_mut("params").unwrap();
        log::trace!(
            "shift {} distortion {} {:?}",
            shift,
            distortion_idx,
            params.shape()
        );
        params[distortion_idx] = 0.0;
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
pub fn try_init_camera(
    frame_feature0: &FrameFeature,
    frame_feature1: &FrameFeature,
    fixed_focal: Option<f64>,
) -> Option<GenericModel<f64>> {
    // initialize focal length and undistorted p2d for init poses
    let (lambda, h_mat) = radial_distortion_homography(frame_feature0, frame_feature1);

    // focal
    let f_option = homography_to_focal(&h_mat);
    if f_option.is_none() {
        println!("Initialization failed, try again.");
        return None;
    }
    let unit_plane_focal = f_option.unwrap() as f64;
    println!("focal {}", unit_plane_focal);

    // poses
    let (rvec0, tvec0) = rtvec_to_na_dvec(init_pose(frame_feature0, lambda));
    let (rvec1, tvec1) = rtvec_to_na_dvec(init_pose(frame_feature1, lambda));
    let rtvec0 = RvecTvec::new(rvec0, tvec0);
    let rtvec1 = RvecTvec::new(rvec1, tvec1);

    let half_w = frame_feature0.img_w_h.0 as f64 / 2.0;
    let half_h = frame_feature0.img_w_h.1 as f64 / 2.0;
    let half_img_size = half_h.max(half_w);
    let init_f = if let Some(focal) = fixed_focal {
        focal
    } else {
        unit_plane_focal * half_img_size
    };
    println!("init f {}", init_f);
    let init_alpha = lambda.abs() as f64;
    if let Some(initial_camera) = init_ucm(
        frame_feature0,
        frame_feature1,
        &rtvec0,
        &rtvec1,
        init_f,
        init_alpha,
        fixed_focal.is_some(),
    ) {
        println!("Initialized {:?}", initial_camera);
        if initial_camera.params()[0] == 0.0 {
            println!("Failed to initialize UCM. Try again.");
            None
        } else {
            Some(initial_camera)
        }
    } else {
        None
    }
}

pub fn find_best_two_frames(detected_feature_frames: &[Option<FrameFeature>]) -> (usize, usize) {
    let mut max_detection = 0;
    let mut max_detection_idxs = Vec::new();
    for (i, f) in detected_feature_frames.iter().enumerate() {
        if let Some(f) = f {
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
        }
    }
    let mut v0: Vec<_> = max_detection_idxs
        .iter()
        .map(|&i| {
            let p_avg = features_avg_center(&detected_feature_frames[i].clone().unwrap().features);
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
            let area = features_covered_area(&detected_feature_frames[i].clone().unwrap().features);
            (i, area)
        })
        .collect();
    v1.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // (*v0[0].0, *v0.last().unwrap().0)
    (v1.last().unwrap().0, v0.last().unwrap().0)
}

pub fn convert_model(
    source_model: &GenericModel<f64>,
    target_model: &mut GenericModel<f64>,
    disabled_distortions: usize,
) {
    if let GenericModel::UCM(m0) = source_model {
        if let GenericModel::EUCM(_) = target_model {
            let params = m0.params();
            let params = params.insert_row(5, 1.0);
            target_model.set_params(&params);
            return;
        } else if let GenericModel::EUCMT(_) = target_model {
            let params = m0.params();
            let params = params.insert_row(5, 1.0);
            let params = params.insert_row(6, 0.0);
            let params = params.insert_row(7, 0.0);
            target_model.set_params(&params);
            return;
        }
    }
    let mut problem = tiny_solver::Problem::new();
    let edge_pixels = source_model.width().max(source_model.height()) as u32 / 100;
    let steps = source_model.width().max(source_model.height()) / 30.0;
    let cost = ModelConvertFactor::new(source_model, target_model, edge_pixels, steps as usize);
    problem.add_residual_block(
        cost.residaul_num(),
        vec![("params".to_string(), target_model.params().len())],
        Box::new(cost),
        Some(Box::new(HuberLoss::new(1.0))),
    );

    let camera_params = source_model.camera_params();
    let mut target_params_init = target_model.params();
    target_params_init.rows_mut(0, 4).copy_from(&camera_params);

    let mut initial_values =
        HashMap::<String, na::DVector<f64>>::from([("params".to_string(), target_params_init)]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    // distortion parameter bound
    set_problem_parameter_bound(&mut problem, target_model, false);
    set_problem_parameter_disabled(
        &mut problem,
        &mut initial_values,
        target_model,
        false,
        disabled_distortions,
    );
    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None).unwrap();

    // save result
    let result_params = result.get("params").unwrap();
    target_model.set_params(result_params);
}

pub fn init_ucm(
    frame_feature0: &FrameFeature,
    frame_feature1: &FrameFeature,
    rtvec0: &RvecTvec,
    rtvec1: &RvecTvec,
    init_f: f64,
    init_alpha: f64,
    fixed_focal: bool,
) -> Option<GenericModel<f64>> {
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
        ("rvec0".to_string(), rtvec0.rvec.clone()),
        ("tvec0".to_string(), rtvec0.tvec.clone()),
        ("rvec1".to_string(), rtvec1.rvec.clone()),
        ("tvec1".to_string(), rtvec1.tvec.clone()),
    ]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};
    if fixed_focal {
        init_focal_alpha_problem.fix_variable("params", 0);
    }

    println!("init ucm init f {}", initial_values.get("params").unwrap());

    // optimize
    init_focal_alpha_problem.set_variable_bounds("params", 0, init_f / 3.0, init_f * 3.0);
    init_focal_alpha_problem.set_variable_bounds("params", 1, 1e-6, 1.0);
    if let Some(mut second_round_values) =
        optimizer.optimize(&init_focal_alpha_problem, &initial_values, None)
    {
        println!(
            "params after {:?}\n",
            second_round_values.get("params").unwrap()
        );

        let focal = second_round_values["params"][0];
        let alpha = second_round_values["params"][1];
        let ucm_all_params = na::dvector![focal, focal, half_w, half_h, alpha];
        let ucm_camera = GenericModel::UCM(UCM::new(
            &ucm_all_params,
            frame_feature0.img_w_h.0,
            frame_feature0.img_w_h.1,
        ));
        second_round_values.remove("params");
        Some(
            calib_camera(
                &[Some(frame_feature0.clone()), Some(frame_feature1.clone())],
                &ucm_camera,
                true,
                0,
                fixed_focal,
            )
            .unwrap()
            .0,
        )
    } else {
        None
    }
}

pub fn calib_camera(
    frame_feature_list: &[Option<FrameFeature>],
    generic_camera: &GenericModel<f64>,
    xy_same_focal: bool,
    disabled_distortions: usize,
    fixed_focal: bool,
) -> Option<(GenericModel<f64>, HashMap<usize, RvecTvec>)> {
    let mut params = generic_camera.params();
    if xy_same_focal {
        // remove fy
        params = params.remove_row(1);
    };
    let params_len = params.len();
    let mut initial_values =
        HashMap::<String, na::DVector<f64>>::from([("params".to_string(), params)]);
    debug!("init {:?}", initial_values);
    let mut problem = tiny_solver::Problem::new();
    let mut valid_indexes = Vec::new();
    for (i, frame_feature) in frame_feature_list.iter().enumerate() {
        if let Some(frame_feature) = frame_feature {
            let mut p3ds = Vec::new();
            let mut p2ds = Vec::new();
            let rvec_name = format!("rvec{}", i);
            let tvec_name = format!("tvec{}", i);
            for fp in frame_feature.features.values() {
                let cost = ReprojectionFactor::new(generic_camera, &fp.p3d, &fp.p2d, xy_same_focal);
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
            valid_indexes.push(i);
            let (rvec, tvec) =
                rtvec_to_na_dvec(sqpnp_simple::sqpnp_solve_glam(&p3ds, &p2ds_z).unwrap());

            initial_values.entry(rvec_name).or_insert(rvec);
            initial_values.entry(tvec_name).or_insert(tvec);
        }
    }

    let optimizer = tiny_solver::GaussNewtonOptimizer {};
    // let initial_values = optimizer.optimize(&problem, &initial_values, None);

    set_problem_parameter_bound(&mut problem, generic_camera, xy_same_focal);
    set_problem_parameter_disabled(
        &mut problem,
        &mut initial_values,
        generic_camera,
        xy_same_focal,
        disabled_distortions,
    );
    let result_option = optimizer.optimize(&problem, &initial_values, None);
    // check is some
    result_option.as_ref()?;
    let mut result = result_option.unwrap();
    if fixed_focal {
        println!("set focal and opt again.");
        problem.fix_variable("params", 0);
        result.get_mut("params").unwrap()[0] = generic_camera.params()[0];
        result = optimizer.optimize(&problem, &result, None).unwrap();
    }

    let mut new_params = result.get("params").unwrap().clone();
    if xy_same_focal {
        // remove fy
        new_params = new_params.clone().insert_row(1, new_params[0]);
    };
    println!("params {}", new_params);
    let mut calibrated_camera = *generic_camera;
    calibrated_camera.set_params(&new_params);
    let rtvec_vec: HashMap<usize, RvecTvec> = valid_indexes
        .iter()
        .map(|&i| {
            let rvec_name = format!("rvec{}", i);
            let tvec_name = format!("tvec{}", i);

            (
                i,
                RvecTvec {
                    rvec: result.remove(&rvec_name).unwrap(),
                    tvec: result.remove(&tvec_name).unwrap(),
                },
            )
        })
        .collect();
    Some((calibrated_camera, rtvec_vec))
}

pub fn init_camera_extrinsic(cam_rtvecs: &[HashMap<usize, RvecTvec>]) -> Vec<RvecTvec> {
    (1..cam_rtvecs.len())
        .map(|cam_i| {
            let cam_0_keys: HashSet<_> = cam_rtvecs[0].keys().cloned().collect();
            let cam_i_keys: HashSet<_> = cam_rtvecs[cam_i].keys().cloned().collect();
            let key_intersection = cam_0_keys.intersection(&cam_i_keys);
            let t_0_b_and_t_i_b: Vec<_> = key_intersection
                .into_iter()
                .map(|k| {
                    let t_0_b = cam_rtvecs[0].get(k).unwrap().to_na_isometry3();
                    let t_i_b = cam_rtvecs[cam_i].get(k).unwrap().to_na_isometry3();
                    (t_0_b, t_i_b)
                })
                .collect();
            let mut problem = tiny_solver::Problem::new();
            let t_i_0_init = t_0_b_and_t_i_b[0].1 * t_0_b_and_t_i_b[0].0.inverse();
            for (t_0_b, t_i_b) in &t_0_b_and_t_i_b {
                let td = t_i_b.inverse() * t_i_0_init * t_0_b;
                print!("td {}", td);
                let cost = SE3Factor::new(t_0_b, t_i_b);
                problem.add_residual_block(
                    6,
                    vec![("rvec".to_string(), 3), ("tvec".to_string(), 3)],
                    Box::new(cost),
                    Some(Box::new(HuberLoss::new(0.1))),
                );
            }
            let rvec = t_i_0_init.rotation.scaled_axis().to_dvec();
            let tvec = na::dvector![
                t_i_0_init.translation.x,
                t_i_0_init.translation.y,
                t_i_0_init.translation.z,
            ];
            let initial_values = HashMap::<String, na::DVector<f64>>::from([
                ("rvec".to_string(), rvec),
                ("tvec".to_string(), tvec),
            ]);

            let optimizer = tiny_solver::GaussNewtonOptimizer {};
            let result = optimizer.optimize(&problem, &initial_values, None).unwrap();
            println!("extrinsic cam{} cam0", cam_i);
            println!("rvec: {}", result["rvec"]);
            println!("tvec: {}", result["tvec"]);
            RvecTvec {
                rvec: result.get("rvec").unwrap().clone(),
                tvec: result.get("tvec").unwrap().clone(),
            }
        })
        .collect()
}

pub fn validation(
    cam_idx: usize,
    final_result: &GenericModel<f64>,
    rtvec_list: &HashMap<usize, RvecTvec>,
    detected_feature_frames: &[Option<FrameFeature>],
    recording_option: Option<&rerun::RecordingStream>,
) {
    let time_reprojection_errors_p2ds: Vec<_> = rtvec_list
        .iter()
        .map(|(&i, rtvec)| {
            let f = detected_feature_frames[i].clone().unwrap();
            let transform = na::Isometry3::new(rtvec.tvec.to_vec3(), rtvec.rvec.to_vec3());
            let (reprojection, p2ds): (Vec<_>, Vec<_>) = f
                .features
                .values()
                .map(|feature| {
                    let p3 = na::Point3::new(feature.p3d.x, feature.p3d.y, feature.p3d.z);
                    let p3p = transform * p3.cast();
                    let p3p = na::Vector3::new(p3p.x, p3p.y, p3p.z);
                    let p2p = final_result.project_one(&p3p);
                    let dx = p2p.x - feature.p2d.x as f64;
                    let dy = p2p.y - feature.p2d.y as f64;
                    ((dx * dx + dy * dy).sqrt(), (feature.p2d.x, feature.p2d.y))
                })
                .unzip();
            if let Some(recording) = recording_option {
                let p3p_rerun: Vec<_> = f
                    .features
                    .values()
                    .map(|feature| {
                        let p3 = na::Point3::new(feature.p3d.x, feature.p3d.y, feature.p3d.z);
                        let p3p = transform.cast() * p3;
                        (p3p.x, p3p.y, p3p.z)
                    })
                    .collect();
                recording.set_time_nanos("stable", f.time_ns);
                recording
                    .log("/board", &rerun::Points3D::new(p3p_rerun))
                    .unwrap();
                recording
                    .log("/camera_coordinate", &rerun::Transform3D::IDENTITY)
                    .unwrap();
                let avg_err = reprojection.iter().sum::<f64>() / reprojection.len() as f64;
                recording
                    .log(
                        "/board/reprojection_err",
                        &rerun::TextLog::new(format!("{} px", avg_err)),
                    )
                    .unwrap();
            };
            (f.time_ns, reprojection, p2ds)
        })
        .collect();
    // detected_feature_frames
    //     .iter()
    //     .filter_map(|f| f.clone())
    //     .enumerate()
    //     .map(|(i, f)| {
    //         let tvec = na::Vector3::new(
    //             rtvec_list[i].tvec[0],
    //             rtvec_list[i].tvec[1],
    //             rtvec_list[i].tvec[2],
    //         );
    //         let rvec = na::Vector3::new(
    //             rtvec_list[i].rvec[0],
    //             rtvec_list[i].rvec[1],
    //             rtvec_list[i].rvec[2],
    //         );
    //         let transform = na::Isometry3::new(tvec, rvec);
    //         let (reprojection, p2ds): (Vec<_>, Vec<_>) = f
    //             .features
    //             .values()
    //             .map(|feature| {
    //                 let p3 = na::Point3::new(feature.p3d.x, feature.p3d.y, feature.p3d.z);
    //                 let p3p = transform * p3.cast();
    //                 let p3p = na::Vector3::new(p3p.x, p3p.y, p3p.z);
    //                 let p2p = final_result.project_one(&p3p);
    //                 let dx = p2p.x - feature.p2d.x as f64;
    //                 let dy = p2p.y - feature.p2d.y as f64;
    //                 ((dx * dx + dy * dy).sqrt(), (feature.p2d.x, feature.p2d.y))
    //             })
    //             .unzip();
    //         if let Some(recording) = recording_option {
    //             let p3p_rerun: Vec<_> = f
    //                 .features
    //                 .values()
    //                 .map(|feature| {
    //                     let p3 = na::Point3::new(feature.p3d.x, feature.p3d.y, feature.p3d.z);
    //                     let p3p = transform.cast() * p3;
    //                     (p3p.x, p3p.y, p3p.z)
    //                 })
    //                 .collect();
    //             recording.set_time_nanos("stable", f.time_ns);
    //             recording
    //                 .log("/board", &rerun::Points3D::new(p3p_rerun))
    //                 .unwrap();
    //             recording
    //                 .log("/camera_coordinate", &rerun::Transform3D::IDENTITY)
    //                 .unwrap();
    //             let avg_err = reprojection.iter().sum::<f64>() / reprojection.len() as f64;
    //             recording
    //                 .log(
    //                     "/board/reprojection_err",
    //                     &rerun::TextLog::new(format!("{} px", avg_err)),
    //                 )
    //                 .unwrap();
    //         };
    //         (f.time_ns, reprojection, p2ds)
    //     })
    //     .collect();
    let mut reprojection_errors: Vec<_> = time_reprojection_errors_p2ds
        .iter()
        .flat_map(|f| f.1.clone())
        .collect();
    println!("total pts: {}", reprojection_errors.len());
    reprojection_errors.sort_by(|&a, b| a.partial_cmp(b).unwrap());
    println!(
        "Median reprojection error: {} px",
        reprojection_errors[reprojection_errors.len() / 2]
    );
    let len_99_percent = reprojection_errors.len() * 99 / 100;
    let avg_99_percent = reprojection_errors
        .iter()
        .take(len_99_percent)
        .map(|p| *p / len_99_percent as f64)
        .sum::<f64>();
    println!("Avg reprojection error of 99%: {} px", avg_99_percent);
    if let Some(recording) = recording_option {
        let topic = format!("/cam{}/detected", cam_idx);
        let color_gradient = colorous::ORANGE_RED;
        let min_v = 0.2;
        for (time_ns, reps, p2ds) in &time_reprojection_errors_p2ds {
            let (colors, text): (Vec<_>, Vec<_>) = reps
                .iter()
                .zip(p2ds)
                .map(|(&r, _)| {
                    let v = (r - min_v).clamp(0.0, 1.0);
                    let c = color_gradient.eval_continuous(v);
                    ((c.r, c.g, c.b, 255), format!("{}", r))
                })
                .unzip();
            recording.set_time_nanos("stable", *time_ns);
            recording
                .log(
                    topic.to_string(),
                    &rerun::Points2D::new(rerun_shift(p2ds))
                        .with_colors(colors)
                        .with_radii([rerun::Radius::new_ui_points(1.0)])
                        .with_labels(text),
                )
                .unwrap();
        }
    }
}
