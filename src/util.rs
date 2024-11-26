use std::collections::HashMap;

use crate::camera_model::CameraModel;

use super::camera_model::generic::GenericModel;
use super::camera_model::{KannalaBrandt4, OpenCVModel5, EUCM, UCM};
use nalgebra::{self as na, Const, Dyn};
use num_dual::DualDVec64;
use tiny_solver::factors::Factor;
use tiny_solver::loss_functions::HuberLoss;
use tiny_solver::Optimizer;

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
            .filter_map(|p| {
                p.as_ref().map(|pp| {
                    na::Vector3::new(
                        num_dual::DualDVec64::from_re(pp.x),
                        num_dual::DualDVec64::from_re(pp.y),
                        num_dual::DualDVec64::from_re(pp.z),
                    )
                })
            })
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

pub fn init_ucm(p2d_pairs: &[(glam::Vec2, glam::Vec2)], img_w: u32, img_h: u32) {
    let mut problem = tiny_solver::Problem::new();
    for (p2d0, p2d1) in p2d_pairs {
        let cost = CustomFactor::new(p2d0, p2d1, img_w, img_h);
        problem.add_residual_block(
            2,
            vec![("h".to_string(), 8)],
            Box::new(cost),
            Some(Box::new(HuberLoss::new(1.0))),
        );
    }
    // vec![("fx_alpha".to_string(), 2), ("h".to_string(), 8)],

    let f_init = img_w.max(img_h) as f64 / 2.0;

    let fx_alpha_init = na::dvector![f_init, 0.6];
    let h_flat_init = na::dvector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    let initial_values =
        HashMap::<String, na::DVector<f64>>::from([("h".to_string(), h_flat_init)]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None);
    println!("{:?}", result);
    // save result
    // let result_params = result.get("x").unwrap();
}

pub struct RandomPnpFactor {
    p2d: na::Vector2<DualDVec64>,
    p3d: na::Point3<DualDVec64>,
}
impl RandomPnpFactor {
    pub fn new(p2d: &glam::Vec2, p3d: &glam::Vec3) -> RandomPnpFactor {
        let p2d = na::Vector2::new(
            DualDVec64::from_re(p2d.x as f64),
            DualDVec64::from_re(p2d.y as f64),
        );
        let p3d = na::Point3::new(
            DualDVec64::from_re(p3d.x as f64),
            DualDVec64::from_re(p3d.y as f64),
            DualDVec64::from_re(p3d.z as f64),
        );
        RandomPnpFactor { p2d, p3d }
    }
}
impl Factor for RandomPnpFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        let rvec = na::Vector3::new(
            params[0][0].clone(),
            params[0][1].clone(),
            params[0][2].clone(),
        );
        let tvec = na::Vector3::new(
            params[1][0].clone(),
            params[1][1].clone(),
            params[1][2].clone(),
        );
        let transform = na::Isometry3::new(tvec, rvec);
        let p3dp = transform * self.p3d.clone();
        let x = p3dp.x.clone() / p3dp.z.clone();
        let y = p3dp.y.clone() / p3dp.z.clone();
        na::dvector![x - self.p2d.x.clone(), y - self.p2d.y.clone()]
    }
}
