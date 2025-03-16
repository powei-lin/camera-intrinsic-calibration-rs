use faer::linalg::solvers::SolveLstsqCore;
use log::debug;
use nalgebra as na;
use rand::seq::SliceRandom;

use crate::detected_points::FrameFeature;

fn h6_l1l2_solver(six_pt_pairs: &[(glam::Vec2, glam::Vec2)]) -> Option<(f32, na::Matrix3<f32>)> {
    let mut m1: faer::Mat<f32> = faer::Mat::zeros(6, 8);
    for (r, (pt0, pt1)) in six_pt_pairs.iter().enumerate() {
        let x = pt0.x;
        let y = pt0.y;
        let x_p = pt1.x;
        let y_p = pt1.y;
        unsafe {
            *m1.get_mut_unchecked(r, 0) = -x * y_p;
            *m1.get_mut_unchecked(r, 1) = -y * y_p;
            *m1.get_mut_unchecked(r, 2) = -y_p;
            *m1.get_mut_unchecked(r, 3) = x * x_p;
            *m1.get_mut_unchecked(r, 4) = x_p * y;
            *m1.get_mut_unchecked(r, 5) = x_p;
            *m1.get_mut_unchecked(r, 6) = -x * x * y_p - y * y * y_p;
            *m1.get_mut_unchecked(r, 7) = x * x * x_p + x_p * y * y;
        }
    }
    // let q_mat = mm1.transpose().qr().q();
    let q_mat = m1.transpose().qr().compute_Q();
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
    if bb_4ac < 0.0 {
        debug!("b*b - 4ac < 0.0");
        return None;
    }
    let g_result = [
        (b_minus - bb_4ac.sqrt()) / (2.0 * (n02 * n07 - n05 * n06)),
        (b_minus + bb_4ac.sqrt()) / (2.0 * (n02 * n07 - n05 * n06)),
    ];
    let mut temp_h = [na::Matrix3::zeros(), na::Matrix3::zeros()];
    let mut l_l_p = na::Matrix2::zeros();
    for which_gamma in 0..2 {
        let gamma = g_result[which_gamma];
        let l = -(gamma * n06 + n16) / (-gamma * n02 - n12);
        let v1 = faer::Scale(gamma) * n.row(0) + n.row(1);
        temp_h[which_gamma][(0, 0)] = *v1.get(0);
        temp_h[which_gamma][(0, 1)] = *v1.get(1);
        temp_h[which_gamma][(0, 2)] = *v1.get(2);
        temp_h[which_gamma][(1, 0)] = *v1.get(3);
        temp_h[which_gamma][(1, 1)] = *v1.get(4);
        temp_h[which_gamma][(1, 2)] = *v1.get(5);

        let mut eq10a: faer::Mat<f32> = faer::Mat::zeros(6, 4);
        let mut eq10b: faer::Mat<f32> = faer::Mat::zeros(6, 1);

        for (row, (pt0, pt1)) in six_pt_pairs.iter().enumerate().take(6) {
            let x = pt0.x;
            let y = pt0.y;
            let x_p = pt1.x;
            let y_p = pt1.y;
            unsafe {
                *eq10a.get_mut_unchecked(row, 0) = -x * x_p;
                *eq10a.get_mut_unchecked(row, 1) = -x_p * y;
                *eq10a.get_mut_unchecked(row, 2) = -l * x * x * x_p - l * x_p * y * y - x_p;
                *eq10a.get_mut_unchecked(row, 3) =
                    l * x * x * x_p * x_p * temp_h[which_gamma][(0, 2)]
                        + l * x * x * y_p * y_p * temp_h[which_gamma][(0, 2)]
                        + l * x_p * x_p * y * y * temp_h[which_gamma][(0, 2)]
                        + l * y * y * y_p * y_p * temp_h[which_gamma][(0, 2)]
                        + x * x_p * x_p * temp_h[which_gamma][(0, 0)]
                        + x * y_p * y_p * temp_h[which_gamma][(0, 0)]
                        + x_p * x_p * y * temp_h[which_gamma][(0, 1)]
                        + x_p * x_p * temp_h[which_gamma][(0, 2)]
                        + y * y_p * y_p * temp_h[which_gamma][(0, 1)]
                        + y_p * y_p * temp_h[which_gamma][(0, 2)];

                *eq10b.get_mut_unchecked(row, 0) = -l * x * x * temp_h[which_gamma][(0, 2)]
                    - l * y * y * temp_h[which_gamma][(0, 2)]
                    - x * temp_h[which_gamma][(0, 0)]
                    - y * temp_h[which_gamma][(0, 1)]
                    - temp_h[which_gamma][(0, 2)];
            }
        }
        // std::cout << "svd\n";
        let mut eq10x = eq10b;
        eq10a
            .qr()
            .solve_lstsq_in_place_with_conj(faer::Conj::No, eq10x.as_mut());

        temp_h[which_gamma][(2, 0)] = *eq10x.get(0, 0);
        temp_h[which_gamma][(2, 1)] = *eq10x.get(1, 0);
        temp_h[which_gamma][(2, 2)] = *eq10x.get(2, 0);
        let l_p = *eq10x.get(3, 0);
        l_l_p[(which_gamma, 0)] = l;
        l_l_p[(which_gamma, 1)] = l_p;
    }

    let which_lambda_idx: usize;
    let l0_valid = l_l_p[(0, 0)] < 0.0 && l_l_p[(0, 1)] < 0.0;
    let l1_valid = l_l_p[(1, 0)] < 0.0 && l_l_p[(1, 1)] < 0.0;
    if !l0_valid && !l1_valid {
        debug!("both l0 and l1 are invalid.");
        return None;
        // return 0.0;
    } else if l0_valid && l1_valid {
        //  all valid
        let s0 = ((l_l_p[(0, 0)] / l_l_p[(0, 1)]).log10()).abs();
        let s1 = ((l_l_p[(1, 0)] / l_l_p[(1, 1)]).log10()).abs();
        if s0 < s1 {
            which_lambda_idx = 0;
        } else {
            which_lambda_idx = 1;
        }
    } else if l0_valid {
        which_lambda_idx = 0;
    } else {
        which_lambda_idx = 1;
    }
    let homography_mat = temp_h[which_lambda_idx];

    let mut avg_lambda = (l_l_p[(which_lambda_idx, 0)] * l_l_p[(which_lambda_idx, 1)]).sqrt();
    if avg_lambda > 0.0 {
        avg_lambda *= -1.0;
    }
    Some((avg_lambda, homography_mat))
}

fn evaluate_homography_lambda(
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

        let alpha = [
            r[2] / 2.0 - in_sqrt.sqrt() / 2.0,
            r[2] / 2.0 + in_sqrt.sqrt() / 2.0,
        ];
        if which_a == 2 {
            if (x_p - r[0] / alpha[0]).abs() < (x_p - r[0] / alpha[1]).abs() {
                which_a = 0;
            } else {
                which_a = 1;
            }
        }
        let d = (x_p - r[0] / alpha[which_a]).powi(2) + (y_p - r[1] / alpha[which_a]).powi(2);
        avg_dist += d.sqrt();
    }

    avg_dist / normalized_p2d_pair.len() as f32
}

pub fn radial_distortion_homography(
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
            frame_feature1.features.get(i).map(|p1| {
                (
                    (p0.p2d - cxcy) / half_img_size,
                    (p1.p2d - cxcy) / half_img_size,
                )
            })
        })
        .collect();
    let ransac_times = 1000;
    let mut rng = rand::rng();

    let mut best_lambda = 0.0;
    let mut best_homography_mat = na::Matrix3::zeros();
    let mut best_distance = f32::MAX;
    let mut nums: Vec<usize> = (0..normalized_p2d_pairs.len()).collect();
    for _ in 0..ransac_times {
        nums.shuffle(&mut rng);
        // println!("{:?}", &nums[0..6]);
        let six_pt_pairs: Vec<_> = (0..6).map(|i| normalized_p2d_pairs[nums[i]]).collect();
        if let Some((lambda, h_mat)) = h6_l1l2_solver(&six_pt_pairs) {
            let avg_distance = evaluate_homography_lambda(&normalized_p2d_pairs, &h_mat, lambda);
            if avg_distance < best_distance {
                best_distance = avg_distance;
                best_lambda = lambda;
                best_homography_mat = h_mat;
            }
        }
    }
    println!("lambda {}, d {}", best_lambda, best_distance);
    println!("{}", best_homography_mat);
    (best_lambda, best_homography_mat)
}

pub fn homography_to_focal(h_mat: &na::Matrix3<f32>) -> Option<f32> {
    let h0 = h_mat[(0, 0)];
    let h1 = h_mat[(0, 1)];
    let h2 = h_mat[(0, 2)];
    let h3 = h_mat[(1, 0)];
    let h4 = h_mat[(1, 1)];
    let h5 = h_mat[(1, 2)];
    let h6 = h_mat[(2, 0)];
    let h7 = h_mat[(2, 1)];
    // let h8 = h_mat[(2, 2)];

    let d1 = h6 * h7;
    let d2 = (h7 - h6) * (h7 + h6);
    let v1 = -(h0 * h1 + h3 * h4) / d1;
    let v2 = (h0 * h0 + h3 * h3 - h1 * h1 - h4 * h4) / d2;
    let (v1, v2) = if v1 < v2 { (v2, v1) } else { (v1, v2) };

    let f1 = if v1 > 0.0 && v2 > 0.0 {
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
    let (v1, v2) = if v1 < v2 { (v2, v1) } else { (v1, v2) };
    let f0 = if v1 > 0.0 && v2 > 0.0 {
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
    match (f0, f1) {
        (Some(f0), Some(f1)) => Some((f0 * f1).sqrt()),
        (Some(f0), None) => Some(f0),
        (None, Some(f1)) => Some(f1),
        _ => None,
    }
}
