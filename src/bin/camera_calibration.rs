use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;
use camera_intrinsic::board::create_default_6x6_board;
use camera_intrinsic::data_loader::load_euroc;
use camera_intrinsic::detected_points::{FeaturePoint, FrameFeature};
use camera_intrinsic::util::{init_ucm, rtvec_to_na_dvec};
use camera_intrinsic::visualization::*;
use clap::Parser;
use core::f32;
use faer::solvers::SpSolverLstsq;
use glam::Vec2;
use log::trace;
use nalgebra as na;
use rand::seq::SliceRandom;
use rerun::RecordingStream;
use sqpnp_simple::sqpnp_solve_glam;
use std::collections::HashMap;
use std::time::Instant;
use tiny_solver::loss_functions::HuberLoss;
use tiny_solver::Optimizer;

#[derive(Parser)]
#[command(version, about, author)]
struct CCRSCli {
    /// path to image folder
    path: String,

    /// tag_family: ["t16h5", "t25h7", "t25h9", "t36h11", "t36h11b1"]
    #[arg(value_enum, default_value = "t36h11")]
    tag_family: TagFamily,
}

fn log_frames(recording: &RecordingStream, detected_feature_frames: &[FrameFeature]) {
    for f in detected_feature_frames {
        let (pts, colors_labels): (Vec<_>, Vec<_>) = f
            .features
            .iter()
            .map(|(id, p)| {
                let color = id_to_color(*id as usize);
                (
                    (p.p2d.x, p.p2d.y),
                    (color, format!("{:?}", p.p3d).to_string()),
                )
            })
            .unzip();
        let (colors, labels): (Vec<_>, Vec<_>) = colors_labels.iter().cloned().unzip();
        let pts = rerun_shift(&pts);

        let topic = "/cam0";
        recording.set_time_nanos("stable", f.time_ns);
        recording
            .log(
                format!("{}/pts", topic),
                &rerun::Points2D::new(pts)
                    .with_colors(colors)
                    .with_labels(labels)
                    .with_radii([rerun::Radius::new_ui_points(5.0)]),
            )
            .unwrap();
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

fn vec2_distance2(v0: &glam::Vec2, v1: &glam::Vec2) -> f32 {
    let v = v0 - v1;
    v.x * v.x + v.y * v.y
}

fn find_best_two_frames(detected_feature_frames: &[FrameFeature]) -> (usize, usize) {
    let mut max_detection = 0;
    let mut max_detection_idxs = Vec::new();
    for (i, f) in detected_feature_frames.iter().enumerate() {
        if f.features.len() > max_detection {
            max_detection = f.features.len();
            max_detection_idxs = vec![i];
        } else if f.features.len() == max_detection {
            max_detection_idxs.push(i);
        }
    }
    let mut v: Vec<_> = max_detection_idxs
        .iter()
        .map(|i| {
            let p_avg = features_avg_center(&detected_feature_frames[*i].features);
            (i, p_avg)
        })
        .collect();

    let avg_all = v.iter().map(|(_, p)| *p).reduce(|acc, e| acc + e).unwrap() / v.len() as f32;
    // let avg_all = Vec2::ZERO;
    v.sort_by(|a, b| {
        vec2_distance2(&a.1, &avg_all)
            .partial_cmp(&vec2_distance2(&b.1, &avg_all))
            .unwrap()
    });
    (*v[0].0, *v.last().unwrap().0)
}

fn h6_l1l2_solver(six_pt_pairs: &[(glam::Vec2, glam::Vec2)]) -> Option<(f32, na::Matrix3<f32>)> {
    let mut m1: faer::Mat<f32> = faer::Mat::zeros(6, 8);
    for r in 0..6 {
        let (pt0, pt1) = six_pt_pairs[r];
        let x = pt0.x;
        let y = pt0.y;
        let x_p = pt1.x;
        let y_p = pt1.y;
        unsafe {
            m1.write_unchecked(r, 0, -x * y_p);
            m1.write_unchecked(r, 1, -y * y_p);
            m1.write_unchecked(r, 2, -y_p);
            m1.write_unchecked(r, 3, x * x_p);
            m1.write_unchecked(r, 4, x_p * y);
            m1.write_unchecked(r, 5, x_p);
            m1.write_unchecked(r, 6, -x * x * y_p - y * y * y_p);
            m1.write_unchecked(r, 7, x * x * x_p + x_p * y * y);
        }
    }
    // let q_mat = mm1.transpose().qr().q();
    let q_mat = m1.transpose().qr().compute_q();
    let q_mat_t = q_mat.transpose();
    let n = q_mat_t.subrows(6, 2);
    let n02 = *n.get(0, 2);
    let n12 = *n.get(1, 2);
    let n05 = *n.get(0, 5);
    let n15 = *n.get(1, 5);
    let n06 = *n.get(0, 6);
    let n16 = *n.get(1, 6);
    let n07 = *n.get(0, 7);
    let n17 = *n.get(1, 7);

    let b_minus = -n02 * n17 + n05 * n16 + n06 * n15 - n07 * n12;
    let bb_4ac = n02 * n02 * n17 * n17
        - 2.0 * n02 * n05 * n16 * n17
        - 2.0 * n02 * n06 * n15 * n17
        - 2.0 * n02 * n07 * n12 * n17
        + 4.0 * n02 * n07 * n15 * n16
        + n05 * n05 * n16 * n16
        + 4.0 * n05 * n06 * n12 * n17
        - 2.0 * n05 * n06 * n15 * n16
        - 2.0 * n05 * n07 * n12 * n16
        + n06 * n06 * n15 * n15
        - 2.0 * n06 * n07 * n12 * n15
        + n07 * n07 * n12 * n12;
    if (bb_4ac < 0.0) {
        println!("bad");
        return None;
    }
    let g_result = vec![
        (b_minus - bb_4ac.sqrt()) / (2.0 * (n02 * n07 - n05 * n06)),
        (b_minus + bb_4ac.sqrt()) / (2.0 * (n02 * n07 - n05 * n06)),
    ];
    let mut temp_h = vec![na::Matrix3::zeros(), na::Matrix3::zeros()];
    let mut l_l_p = na::Matrix2::zeros();
    for which_gamma in 0..2 {
        let gamma = g_result[which_gamma];
        let l = -(gamma * n06 + n16) / (-gamma * n02 - n12);
        let v1 = gamma * n.row(0) + n.row(1);
        temp_h[which_gamma][(0, 0)] = *v1.get(0);
        temp_h[which_gamma][(0, 1)] = *v1.get(1);
        temp_h[which_gamma][(0, 2)] = *v1.get(2);
        temp_h[which_gamma][(1, 0)] = *v1.get(3);
        temp_h[which_gamma][(1, 1)] = *v1.get(4);
        temp_h[which_gamma][(1, 2)] = *v1.get(5);

        let mut eq10A: faer::Mat<f32> = faer::Mat::zeros(6, 4);
        let mut eq10b: faer::Mat<f32> = faer::Mat::zeros(6, 1);

        for row in 0..6 {
            let (pt0, pt1) = six_pt_pairs[row];
            let x = pt0.x;
            let y = pt0.y;
            let x_p = pt1.x;
            let y_p = pt1.y;
            unsafe {
                eq10A.write_unchecked(row, 0, -x * x_p);
                eq10A.write_unchecked(row, 1, -x_p * y);
                eq10A.write_unchecked(row, 2, -l * x * x * x_p - l * x_p * y * y - x_p);
                eq10A.write_unchecked(
                    row,
                    3,
                    l * x * x * x_p * x_p * temp_h[which_gamma][(0, 2)]
                        + l * x * x * y_p * y_p * temp_h[which_gamma][(0, 2)]
                        + l * x_p * x_p * y * y * temp_h[which_gamma][(0, 2)]
                        + l * y * y * y_p * y_p * temp_h[which_gamma][(0, 2)]
                        + x * x_p * x_p * temp_h[which_gamma][(0, 0)]
                        + x * y_p * y_p * temp_h[which_gamma][(0, 0)]
                        + x_p * x_p * y * temp_h[which_gamma][(0, 1)]
                        + x_p * x_p * temp_h[which_gamma][(0, 2)]
                        + y * y_p * y_p * temp_h[which_gamma][(0, 1)]
                        + y_p * y_p * temp_h[which_gamma][(0, 2)],
                );

                eq10b.write_unchecked(
                    row,
                    0,
                    -l * x * x * temp_h[which_gamma][(0, 2)]
                        - l * y * y * temp_h[which_gamma][(0, 2)]
                        - x * temp_h[which_gamma][(0, 0)]
                        - y * temp_h[which_gamma][(0, 1)]
                        - temp_h[which_gamma][(0, 2)],
                );
            }
        }
        // std::cout << "svd\n";
        let eq10x = eq10A.qr().solve_lstsq(eq10b);
        // Eigen::JacobiSVD<Eigen::Matrix<float, 6, 4>> svd(
        //     eq10A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // Eigen::Matrix<float, 4, 1> eq10x = svd.solve(eq10b);
        temp_h[which_gamma][(2, 0)] = *eq10x.get(0, 0);
        temp_h[which_gamma][(2, 1)] = *eq10x.get(1, 0);
        temp_h[which_gamma][(2, 2)] = *eq10x.get(2, 0);
        let l_p = *eq10x.get(3, 0);
        l_l_p[(which_gamma, 0)] = l;
        l_l_p[(which_gamma, 1)] = l_p;
    }

    let which_l: usize;
    let l0_valid = l_l_p[(0, 0)] < 0.0 && l_l_p[(0, 1)] < 0.0;
    let l1_valid = l_l_p[(1, 0)] < 0.0 && l_l_p[(1, 1)] < 0.0;
    if (!l0_valid && !l1_valid) {
        println!("bbb");
        return None;
        // return 0.0;
    } else if (l0_valid && l1_valid) {
        //  all valid
        let s0 = ((l_l_p[(0, 0)] / l_l_p[(0, 1)]).log10()).abs();
        let s1 = ((l_l_p[(1, 0)] / l_l_p[(1, 1)]).log10()).abs();
        if (s1 != s1 || s0 < s1) {
            which_l = 0;
        } else {
            which_l = 1;
        }
    } else if (l0_valid) {
        which_l = 0;
    } else {
        which_l = 1;
    }
    let H = temp_h[which_l];

    let mut avg_lambda = (l_l_p[(which_l, 0)] * l_l_p[(which_l, 1)]).sqrt();
    if (avg_lambda > 0.0) {
        avg_lambda *= -1.0;
    }
    //   println!("avg lambda {}", avg_lambda);
    return Some((avg_lambda, H));
    // cout << "avg lambda: " << avg_lambda << endl;
    // l, l_p = avg_lambda, avg_lambda

    // return (l_l_p[which_l][0], l_l_p[which_l][1]), h
    //   return avg_lambda;
    //   l_l_p.x[(1, 1)];
}

fn evaluate_H_lambda(
    normalized_p2d_pair: &[(glam::Vec2, glam::Vec2)],
    h_mat: &na::Matrix3<f32>,
    lambda: f32,
) -> f32 {
    let mut which_a = 2;
    let mut avg_dist = 0.0;
    for &pt_pair in normalized_p2d_pair {
        let x = pt_pair.0.x;
        let y = pt_pair.0.y;
        let sc = 1.0 + lambda * (x * x + y * y);
        let x_p = pt_pair.1.x;
        let y_p = pt_pair.1.y;

        let pt0 = na::Vector3::new(x, y, sc);
        let r = h_mat * pt0;

        let mut in_sqrt = -4.0 * lambda * r[0] * r[0] - 4.0 * lambda * r[1] * r[1] + r[2] * r[2];
        in_sqrt = in_sqrt.max(0.0);

        let alpha = vec![
            r[2] / 2.0 - in_sqrt.sqrt() / 2.0,
            r[2] / 2.0 + in_sqrt.sqrt() / 2.0,
        ];
        if (which_a == 2) {
            if ((x_p - r[0] / alpha[0]).abs() < (x_p - r[0] / alpha[1]).abs()) {
                which_a = 0;
            } else {
                which_a = 1;
            }
        }
        let d = (x_p - r[0] / alpha[which_a]).powi(2) + (y_p - r[1] / alpha[which_a]).powi(2);
        avg_dist += d.sqrt();
    }

    return avg_dist / normalized_p2d_pair.len() as f32;
}

fn radial_distortion_homography(
    frame_feature0: &FrameFeature,
    frame_feature1: &FrameFeature,
) -> (f32, na::Matrix3<f32>) {
    let half_w = frame_feature0.img_w_h.0 as f32 / 2.0;
    let half_h = frame_feature0.img_w_h.1 as f32 / 2.0;
    let half_img_size = half_h.max(half_w);
    let cxcy = glam::Vec2::new(half_w, half_h);
    let normalized_p2d_pairs: Vec<_> = frame_feature0
        .features
        .iter()
        .filter_map(|(i, p0)| {
            if let Some(p1) = frame_feature1.features.get(i) {
                Some((
                    (p0.p2d - cxcy) / half_img_size,
                    (p1.p2d - cxcy) / half_img_size,
                ))
            } else {
                None
            }
        })
        .collect();
    let ransac_times = 1000;
    let mut rng = rand::thread_rng();

    let mut best_lambda = 0.0;
    let mut best_H = na::Matrix3::zeros();
    let mut best_distance = f32::MAX;
    let mut nums: Vec<usize> = (0..normalized_p2d_pairs.len()).collect();
    for _ in 0..ransac_times {
        nums.shuffle(&mut rng);
        // println!("{:?}", &nums[0..6]);
        let six_pt_pairs: Vec<_> = (0..6)
            .into_iter()
            .map(|i| normalized_p2d_pairs[nums[i]])
            .collect();
        if let Some((lambda, h_mat)) = h6_l1l2_solver(&six_pt_pairs) {
            let avg_distance = evaluate_H_lambda(&normalized_p2d_pairs, &h_mat, lambda);
            if (avg_distance < best_distance) {
                best_distance = avg_distance;
                best_lambda = lambda;
                best_H = h_mat;
            }
        }
    }
    println!("lambda {}, d {}", best_lambda, best_distance);
    println!("{}", best_H);
    (best_lambda, best_H)
}

fn homography_to_focal(h_mat: &na::Matrix3<f32>) -> Option<f32> {
    let h0 = h_mat[(0, 0)];
    let h1 = h_mat[(0, 1)];
    let h2 = h_mat[(0, 2)];
    let h3 = h_mat[(1, 0)];
    let h4 = h_mat[(1, 1)];
    let h5 = h_mat[(1, 2)];
    let h6 = h_mat[(2, 0)];
    let h7 = h_mat[(2, 1)];
    let h8 = h_mat[(2, 2)];

    let d1 = h6 * h7;
    let d2 = (h7 - h6) * (h7 + h6);
    let v1 = -(h0 * h1 + h3 * h4) / d1;
    let v2 = (h0 * h0 + h3 * h3 - h1 * h1 - h4 * h4) / d2;
    let (v1, v2) = if (v1 < v2) { (v2, v1) } else { (v1, v2) };

    let f1 = if (v1 > 0.0 && v2 > 0.0) {
        if d1.abs() > d2.abs() {
            Some(v1.sqrt())
        } else {
            Some(v2.sqrt())
        }
    } else if v1 > 0.0 {
        Some(v1.sqrt())
    } else {
        None
    };

    let d1 = h0 * h3 + h1 * h4;
    let d2 = h0 * h0 + h1 * h1 - h3 * h3 - h4 * h4;
    let v1 = -h2 * h5 / d1;
    let v2 = (h5 * h5 - h2 * h2) / d2;
    let (v1, v2) = if (v1 < v2) { (v2, v1) } else { (v1, v2) };
    let f0 = if (v1 > 0.0 && v2 > 0.0) {
        if d1.abs() > d2.abs() {
            Some(v1.sqrt())
        } else {
            Some(v2.sqrt())
        }
    } else if (v1 > 0.0) {
        Some(v1.sqrt())
    } else {
        None
    };
    if f0.is_some() && f1.is_some() {
        Some((f0.unwrap() * f1.unwrap()).sqrt())
    } else if f0.is_some() {
        f0
    } else if f1.is_some() {
        f1
    } else {
        None
    }
}

fn init_pose(frame_feature: &FrameFeature, lambda: f32) -> ((f64, f64, f64), (f64, f64, f64)) {
    let half_w = frame_feature.img_w_h.0 as f32 / 2.0;
    let half_h = frame_feature.img_w_h.1 as f32 / 2.0;
    let half_img_size = half_h.max(half_w);
    let cxcy = glam::Vec2::new(half_w, half_h);
    let (p2ds_z, p3ds): (Vec<_>, Vec<_>) = frame_feature
        .features
        .iter()
        .map(|f| {
            let xy = (f.1.p2d - cxcy) / half_img_size;
            let sc = 1.0 + lambda * (xy.x * xy.x + xy.y * xy.y);
            (xy / sc, f.1.p3d)
        })
        .unzip();

    sqpnp_solve_glam(&p3ds, &p2ds_z).unwrap()
}

fn main() {
    env_logger::init();
    let cli = CCRSCli::parse();
    let detector = TagDetector::new(&cli.tag_family, None);
    let board = create_default_6x6_board();
    let dataset_root = &cli.path;
    let now = Instant::now();
    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save("output.rrd")
        .unwrap();
    trace!("Start loading data");
    let detected_feature_frames = load_euroc(dataset_root, &detector, &board, None);
    let duration_sec = now.elapsed().as_secs_f64();
    println!("detecting feature took {:.6} sec", duration_sec);
    println!(
        "avg: {} sec",
        duration_sec / detected_feature_frames.len() as f64
    );
    log_frames(&recording, &detected_feature_frames);
    let (frame0, frame1) = find_best_two_frames(&detected_feature_frames);
    let key_frames = vec![
        detected_feature_frames[frame0].clone(),
        detected_feature_frames[frame1].clone(),
    ];
    log_frames(&recording, &key_frames);

    // initialize focal length and undistorted p2d for init poses
    let (lambda, h_mat) = radial_distortion_homography(
        &detected_feature_frames[frame0],
        &detected_feature_frames[frame1],
    );
    // focal
    let f_option = homography_to_focal(&h_mat);
    if f_option.is_none() {
        return;
    }
    let focal = f_option.unwrap();

    // poses
    let frame_feature0 = &detected_feature_frames[frame0];
    let frame_feature1 = &detected_feature_frames[frame1];
    let (rvec0, tvec0) = rtvec_to_na_dvec(init_pose(frame_feature0, lambda));
    let (rvec1, tvec1) = rtvec_to_na_dvec(init_pose(frame_feature1, lambda));

    println!("rvec {:?}", rvec0);
    println!("tvec {:?}", tvec0);

    let half_w = frame_feature0.img_w_h.0 as f64 / 2.0;
    let half_h = frame_feature0.img_w_h.1 as f64 / 2.0;
    let half_img_size = half_h.max(half_w);
    let init_f = focal as f64 * half_img_size;
    let init_alpha = lambda.abs() as f64;
    init_ucm(
        frame_feature0,
        frame_feature1,
        &rvec0,
        &tvec0,
        &rvec1,
        &tvec1,
        init_f,
        init_alpha,
    );

    return;
    // let normalized_p2d_pairs: Vec<_> = frame_feature0
    //     .features
    //     .iter()
    //     .filter_map(|(i, p0)| {
    //         if let Some(p1) = frame_feature1.features.get(i) {
    //             Some((
    //                 (p0.p2d - cxcy) / half_img_size,
    //                 (p1.p2d - cxcy) / half_img_size,
    //             ))
    //         } else {
    //             None
    //         }
    //     })
    //     .collect();
    // let mut pt0 = Vec::new();
    // let mut pt1 = Vec::new();
    // let mut colors = Vec::new();
    // for (p0, p1) in &normalized_p2d_pairs {
    //     let color = (
    //         rand::random::<u8>(),
    //         rand::random::<u8>(),
    //         rand::random::<u8>(),
    //         255u8,
    //     );
    //     colors.push(color);
    //     let x = p0.x;
    //     let y = p0.y;
    //     let sc = 1.0 + lambda * (x * x + y * y);

    //     let pp0 = (
    //         x / sc * half_img_size + half_w,
    //         y / sc * half_img_size + half_h,
    //     );

    //     let x_p = p1.x;
    //     let y_p = p1.y;
    //     let sc_p = 1.0 + lambda * (x_p * x_p + y_p * y_p);
    //     let pp1 = (
    //         x_p / sc_p * half_img_size + half_w,
    //         y_p / sc_p * half_img_size + half_h,
    //     );
    //     pt0.push(pp0);
    //     pt1.push(pp1);
    // }
    // recording
    //     .log(
    //         "/pta0",
    //         &rerun::Points2D::new(pt0).with_colors(colors.clone()),
    //     )
    //     .unwrap();
    // recording
    //     .log("/pta1", &rerun::Points2D::new(pt1).with_colors(colors))
    //     .unwrap();
}
