use std::collections::HashMap;

use super::camera_model::generic::GenericModel;
use super::camera_model::{KannalaBrandt4, OpenCVModel5, EUCM, UCM};
use nalgebra as na;
use num_dual::DualDVec64;
use tiny_solver::factors::Factor;
use tiny_solver::Optimizer;

#[derive(Clone)]
pub struct CustomFactor {
    pub source: GenericModel<DualDVec64>,
    pub target: GenericModel<DualDVec64>,
    pub p3ds: Vec<na::Vector3<DualDVec64>>,
}

impl CustomFactor {
    pub fn new(
        source: &GenericModel<f64>,
        target: &GenericModel<f64>,
        edge_pixels: u32,
        steps: usize,
    ) -> CustomFactor {
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
                p.as_ref().map(|pp| na::Vector3::new(
                        num_dual::DualDVec64::from_re(pp.x),
                        num_dual::DualDVec64::from_re(pp.y),
                        num_dual::DualDVec64::from_re(pp.z),
                    ))
            })
            .collect();
        CustomFactor {
            source: source.cast(),
            target: target.cast(),
            p3ds,
        }
    }
    pub fn residaul_num(&self) -> usize {
        self.p3ds.len() * 2
    }
}

impl Factor for CustomFactor {
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
    let cost = CustomFactor::new(source_model, target_model, edge_pixels, 3);
    problem.add_residual_block(
        cost.residaul_num(),
        vec![("x".to_string(), target_model.params().len())],
        Box::new(cost),
        None,
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
