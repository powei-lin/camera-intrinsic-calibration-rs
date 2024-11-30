use std::collections::HashMap;

use crate::camera_model::CameraModel;
use crate::detected_points::FrameFeature;

use super::camera_model::generic::GenericModel;
use super::camera_model::{KannalaBrandt4, OpenCVModel5, EUCM, UCM};
use nalgebra::{self as na, Const, Dyn};
use num_dual::DualDVec64;
use tiny_solver::factors::Factor;
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

#[derive(Clone)]
pub struct ModelConvertFactor {
    pub source: GenericModel<DualDVec64>,
    pub target: GenericModel<DualDVec64>,
    pub p3ds: Vec<na::Vector3<DualDVec64>>,
}

impl ModelConvertFactor {
    pub fn new(
        source: &GenericModel<f64>,
        target: &GenericModel<f64>,
        edge_pixels: u32,
        steps: usize,
    ) -> ModelConvertFactor {
        if source.width().round() as u32 != target.width().round() as u32 {
            panic!("source width and target width are not the same.")
        } else if source.height().round() as u32 != target.height().round() as u32 {
            panic!("source height and target height are not the same.")
        }
        let mut p2ds = Vec::new();
        for r in (edge_pixels..source.height() as u32 - edge_pixels).step_by(steps) {
            for c in (edge_pixels..source.width() as u32 - edge_pixels).step_by(steps) {
                p2ds.push(na::Vector2::new(c as f64, r as f64));
            }
        }
        let p3ds = source.unproject(&p2ds);
        let p3ds: Vec<_> = p3ds
            .iter()
            .filter_map(|p| p.as_ref().map(|pp| pp.cast()))
            .collect();
        ModelConvertFactor {
            source: source.cast(),
            target: target.cast(),
            p3ds,
        }
    }
    pub fn residaul_num(&self) -> usize {
        self.p3ds.len() * 2
    }
}

impl Factor for ModelConvertFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        let model: GenericModel<num_dual::DualDVec64> = match &self.target {
            GenericModel::EUCM(m) => GenericModel::EUCM(EUCM::new(&params[0], m.width, m.height)),
            GenericModel::UCM(m) => GenericModel::UCM(UCM::new(&params[0], m.width, m.height)),
            GenericModel::OpenCVModel5(m) => {
                GenericModel::OpenCVModel5(OpenCVModel5::new(&params[0], m.width, m.height))
            }
            GenericModel::KannalaBrandt4(m) => {
                GenericModel::KannalaBrandt4(KannalaBrandt4::new(&params[0], m.width, m.height))
            }
        };
        let p2ds0 = self.source.project(&self.p3ds);
        let p2ds1 = model.project(&self.p3ds);
        let diff: Vec<_> = p2ds0
            .iter()
            .zip(p2ds1)
            .flat_map(|(p0, p1)| {
                if let Some(p0) = p0 {
                    if let Some(p1) = p1 {
                        let pp = p0 - p1;
                        return vec![pp[0].clone(), pp[1].clone()];
                    }
                }
                vec![
                    num_dual::DualDVec64::from_re(0.0),
                    num_dual::DualDVec64::from_re(0.0),
                ]
            })
            .collect();
        na::DVector::from_vec(diff)
    }
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

    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None);

    // save result
    let result_params = result.get("x").unwrap();
    target_model.set_params(result_params);
}

pub struct CustomFactor {
    p2d0: na::Vector2<DualDVec64>,
    p2d1: na::Vector2<DualDVec64>,
    img_w: u32,
    img_h: u32,
}

impl CustomFactor {
    pub fn new(p2d0v: &glam::Vec2, p2d1v: &glam::Vec2, img_w: u32, img_h: u32) -> CustomFactor {
        let p2d0 = na::Vector2::new(
            DualDVec64::from_re(p2d0v.x as f64),
            DualDVec64::from_re(p2d0v.y as f64),
        );
        let p2d1 = na::Vector2::new(
            DualDVec64::from_re(p2d1v.x as f64),
            DualDVec64::from_re(p2d1v.y as f64),
        );
        CustomFactor {
            p2d0,
            p2d1,
            img_w,
            img_h,
        }
    }
}

impl Factor for CustomFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        // let f = params[0][0].clone();
        // let alpha = params[0][1].clone();
        let f = DualDVec64::from_re(189.0);
        let alpha = DualDVec64::from_re(0.6);
        let cx = DualDVec64::from_re(self.img_w as f64 / 2.0);
        let cy = DualDVec64::from_re(self.img_h as f64 / 2.0);
        let new_params = na::dvector![f.clone(), f, cx, cy, alpha];
        let ucm = UCM::new(&new_params, self.img_w, self.img_h);
        let h_flat = params[0].push(DualDVec64::from_re(1.0));
        let h = h_flat.reshape_generic(Const::<3>, Dyn(3));
        let p3d0 = ucm.unproject_one(&self.p2d0);
        let p3d1 = h * p3d0;
        let p2d1p = ucm.project_one(&p3d1);
        let diff = p2d1p - self.p2d1.clone();
        na::dvector![diff[0].clone(), diff[1].clone()]
    }
}

struct UCMInitFocalAlphaFactor {
    pub target: GenericModel<DualDVec64>,
    pub p3d: na::Point3<DualDVec64>,
    pub p2d: na::Vector2<DualDVec64>,
}

impl UCMInitFocalAlphaFactor {
    pub fn new(
        target: &GenericModel<f64>,
        p3d: &glam::Vec3,
        p2d: &glam::Vec2,
    ) -> UCMInitFocalAlphaFactor {
        let target = target.cast();
        let p3d = na::Point3::new(p3d.x, p3d.y, p3d.z).cast();
        let p2d = na::Vector2::new(p2d.x, p2d.y).cast();
        UCMInitFocalAlphaFactor { target, p3d, p2d }
    }
}
impl Factor for UCMInitFocalAlphaFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        // params[[f, alpha], rvec, tvec]
        let mut cam_params = self.target.params();
        cam_params[0] = params[0][0].clone();
        cam_params[1] = params[0][0].clone();
        cam_params[4] = params[0][1].clone();
        let model: GenericModel<num_dual::DualDVec64> = match &self.target {
            GenericModel::EUCM(m) => GenericModel::EUCM(EUCM::new(&cam_params, m.width, m.height)),
            GenericModel::UCM(m) => GenericModel::UCM(UCM::new(&cam_params, m.width, m.height)),
            GenericModel::OpenCVModel5(m) => {
                GenericModel::OpenCVModel5(OpenCVModel5::new(&cam_params, m.width, m.height))
            }
            GenericModel::KannalaBrandt4(m) => {
                GenericModel::KannalaBrandt4(KannalaBrandt4::new(&cam_params, m.width, m.height))
            }
        };
        let rvec = na::Vector3::new(
            params[1][0].clone(),
            params[1][1].clone(),
            params[1][2].clone(),
        );
        let tvec = na::Vector3::new(
            params[2][0].clone(),
            params[2][1].clone(),
            params[2][2].clone(),
        );
        let transform = na::Isometry3::new(tvec, rvec);
        let p3d_t = transform * self.p3d.clone();
        let p3d_t = na::Vector3::new(p3d_t.x.clone(), p3d_t.y.clone(), p3d_t.z.clone());
        let p2d_p = model.project_one(&p3d_t);

        na::dvector![
            p2d_p[0].clone() - self.p2d[0].clone(),
            p2d_p[1].clone() - self.p2d[1].clone()
        ]
    }
}

struct ReprojectionFactor {
    pub target: GenericModel<DualDVec64>,
    pub p3d: na::Point3<DualDVec64>,
    pub p2d: na::Vector2<DualDVec64>,
}

impl ReprojectionFactor {
    pub fn new(
        target: &GenericModel<f64>,
        p3d: &glam::Vec3,
        p2d: &glam::Vec2,
    ) -> ReprojectionFactor {
        let target = target.cast();
        let p3d = na::Point3::new(p3d.x, p3d.y, p3d.z).cast();
        let p2d = na::Vector2::new(p2d.x, p2d.y).cast();
        ReprojectionFactor { target, p3d, p2d }
    }
}
impl Factor for ReprojectionFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        // params[params, rvec, tvec]
        let model: GenericModel<num_dual::DualDVec64> = match &self.target {
            GenericModel::EUCM(m) => GenericModel::EUCM(EUCM::new(&params[0], m.width, m.height)),
            GenericModel::UCM(m) => GenericModel::UCM(UCM::new(&params[0], m.width, m.height)),
            GenericModel::OpenCVModel5(m) => {
                GenericModel::OpenCVModel5(OpenCVModel5::new(&params[0], m.width, m.height))
            }
            GenericModel::KannalaBrandt4(m) => {
                GenericModel::KannalaBrandt4(KannalaBrandt4::new(&params[0], m.width, m.height))
            }
        };
        let rvec = na::Vector3::new(
            params[1][0].clone(),
            params[1][1].clone(),
            params[1][2].clone(),
        );
        let tvec = na::Vector3::new(
            params[2][0].clone(),
            params[2][1].clone(),
            params[2][2].clone(),
        );
        let transform = na::Isometry3::new(tvec, rvec);
        let p3d_t = transform * self.p3d.clone();
        let p3d_t = na::Vector3::new(p3d_t.x.clone(), p3d_t.y.clone(), p3d_t.z.clone());
        let p2d_p = model.project_one(&p3d_t);

        na::dvector![
            p2d_p[0].clone() - self.p2d[0].clone(),
            p2d_p[1].clone() - self.p2d[1].clone()
        ]
    }
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
    // let init_f = 471.0;
    // let init_alpha = 0.67;
    //     471.019
    // 470.243
    // 367.122
    // 246.741
    // 0.67485
    let init_params = na::dvector![init_f, init_f, half_w, half_h, init_alpha];
    let ucm_init_model = GenericModel::UCM(UCM::new(
        &init_params,
        frame_feature0.img_w_h.0,
        frame_feature0.img_w_h.1,
    ));

    let mut init_focal_alpha_problem = tiny_solver::Problem::new();
    let init_f_alpha = na::dvector![init_f, init_alpha];

    for (_, fp) in &frame_feature0.features {
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

    for (_, fp) in &frame_feature1.features {
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
        Some(second_round_values),
    )
    .0
}

pub fn calib_camera(
    frame_feature_list: &[FrameFeature],
    generic_camera: &GenericModel<f64>,
    initial_values_option: Option<HashMap<String, na::DVector<f64>>>,
) -> (GenericModel<f64>, Vec<(na::DVector<f64>, na::DVector<f64>)>) {
    let params = generic_camera.params();
    let params_len = params.len();
    let mut problem = tiny_solver::Problem::new();
    let mut initial_values = if let Some(mut init_values) = initial_values_option {
        init_values.insert("params".to_string(), params);
        init_values
    } else {
        HashMap::<String, na::DVector<f64>>::from([("params".to_string(), params)])
    };
    println!("init {:?}", initial_values);
    let mut valid_indexes = Vec::new();
    for (i, frame_feature) in frame_feature_list.iter().enumerate() {
        // println!("f{}", i);
        let mut p3ds = Vec::new();
        let mut p2ds = Vec::new();
        let rvec_name = format!("rvec{}", i);
        let tvec_name = format!("tvec{}", i);
        for (_, fp) in &frame_feature.features {
            let cost = ReprojectionFactor::new(&generic_camera, &fp.p3d, &fp.p2d);
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
                if let Some(p2) = p2 {
                    Some((p3, glam::Vec2::new(p2.x as f32, p2.y as f32)))
                } else {
                    None
                }
            })
            .unzip();
        if p3ds.len() < 6 {
            println!("skip frame {}", i);
            continue;
        }
        valid_indexes.push(i);
        let (rvec, tvec) =
            rtvec_to_na_dvec(sqpnp_simple::sqpnp_solve_glam(&p3ds, &p2ds_z).unwrap());
        // let tt = na::Vector3::new(tvec[0], tvec[1], tvec[2]);
        // let rr = na::Vector3::new(rvec[0], rvec[1], rvec[2]);
        // let rt = na::Isometry3::new(tt, rr);
        // if p3ds.iter().map(|p|{
        //     let pp = na::Point3::new(p.x as f64, p.y as f64, p.z as f64);
        //     let z = (rt * pp).z;
        //     z
        // }).any(|z| z < 0.0){
        //     continue;
        // }
        // println!("rvec pnp {}", rvec);
        if !initial_values.contains_key(&rvec_name) {
            initial_values.insert(rvec_name, rvec);
        }
        if !initial_values.contains_key(&tvec_name) {
            initial_values.insert(tvec_name, tvec);
        }
    }

    let optimizer = tiny_solver::GaussNewtonOptimizer {};
    let initial_values = optimizer.optimize(&problem, &initial_values, None);
    problem.set_variable_bounds("params", 0, 0.0, 10000.0);
    problem.set_variable_bounds("params", 1, 0.0, 10000.0);
    problem.set_variable_bounds("params", 2, 0.0, generic_camera.width());
    problem.set_variable_bounds("params", 3, 0.0, generic_camera.height());
    let mut result = optimizer.optimize(&problem, &initial_values, None);

    let new_params = result.get("params").unwrap();
    println!("params {}", new_params);
    let mut calibrated_camera = generic_camera.clone();
    calibrated_camera.set_params(&new_params);
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
