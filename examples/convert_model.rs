use std::collections::HashMap;

use camera_intrinsic::camera_model::generic::{
    estimate_new_camera_matrix_for_undistort, init_undistort_map, remap, CameraModel, GenericModel,
};
use camera_intrinsic::camera_model::{
    model_from_json, model_to_json, KannalaBrandt4, OpenCVModel5, EUCM, UCM,
};
use image::ImageReader;
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
    pub fn new(source: &GenericModel<f64>, target: &GenericModel<f64>) -> CustomFactor {
        if source.width().round() as u32 != target.width().round() as u32 {
            panic!("source width and target width are not the same.")
        } else if source.height().round() as u32 != target.height().round() as u32 {
            panic!("source height and target height are not the same.")
        }
        let mut p2ds = Vec::new();
        for r in (10..source.height() as u32 - 10).step_by(3) {
            for c in (10..source.width() as u32 - 10).step_by(3) {
                p2ds.push(na::Vector2::new(c as f64, r as f64));
            }
        }
        let p3ds = source.unproject(&p2ds);
        let p3ds: Vec<_> = p3ds
            .iter()
            .filter_map(|p| {
                if let Some(pp) = p {
                    Some(na::Vector3::new(
                        num_dual::DualDVec64::from_re(pp.x),
                        num_dual::DualDVec64::from_re(pp.y),
                        num_dual::DualDVec64::from_re(pp.z),
                    ))
                } else {
                    None
                }
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

fn convert_model(source_model: &GenericModel<f64>, target_model: &mut GenericModel<f64>) {
    let mut problem = tiny_solver::Problem::new();
    let cost = CustomFactor::new(&source_model, &target_model);
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

fn main() {
    env_logger::init();
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = image::DynamicImage::ImageLuma8(img.to_luma8());
    let source_model = model_from_json("data/eucm.json");
    // let mut target_model = GenericModel::KannalaBrandt4(KannalaBrandt4::new(
    //     &na::dvector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     source_model.width().round() as u32,
    //     source_model.height().round() as u32,
    // ));
    let mut target_model = GenericModel::UCM(UCM::new(
        &na::dvector![0.0, 0.0, 0.0, 0.0, 0.0],
        source_model.width().round() as u32,
        source_model.height().round() as u32,
    ));
    convert_model(&source_model, &mut target_model);
    model_to_json("ucm.json", &target_model);
    let new_w_h = 1024;
    let p = target_model.estimate_new_camera_matrix_for_undistort(1.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = target_model.init_undistort_map(&p, (new_w_h, new_w_h));
    let remaped = remap(&img, &xmap, &ymap);
    remaped.save("remaped_ucm.png").unwrap()
}
