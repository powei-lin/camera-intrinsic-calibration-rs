use std::collections::HashMap;

use camera_intrinsic::camera_model::generic::{
    estimate_new_camera_matrix_for_undistort, init_undistort_map, remap, CameraModel, GenericModel,
};
use camera_intrinsic::camera_model::{model_from_json, KannalaBrandt4, OpenCVModel5, EUCM, UCM};
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

fn main() {
    env_logger::init();
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = image::DynamicImage::ImageLuma8(img.to_luma8());
    let source_model = model_from_json("data/eucm.json");
    let target_model = GenericModel::KannalaBrandt4(KannalaBrandt4::new(
        &na::dvector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        source_model.width().round() as u32,
        source_model.height().round() as u32,
    ));
    let mut problem = tiny_solver::Problem::new();
    let cost = CustomFactor::new(&source_model, &target_model);
    problem.add_residual_block(
        cost.residaul_num(),
        vec![("x".to_string(), 8)],
        Box::new(cost),
        None,
    );

    let params = source_model.params();
    // the initial values for x is 0.7 and yz is [-30.2, 123.4]
    let initial_values = HashMap::<String, na::DVector<f64>>::from([(
        "x".to_string(),
        na::dvector![params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0],
    )]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None);
    println!("{:?}", result);
    let fisheye = KannalaBrandt4::new(
        result.get("x").unwrap(),
        source_model.width() as u32,
        source_model.height() as u32,
    );
    let new_w_h = 1024;
    let p = estimate_new_camera_matrix_for_undistort(&fisheye, 1.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = init_undistort_map(&fisheye, &p, (new_w_h, new_w_h));
    let remaped = remap(&img, &xmap, &ymap);
    remaped.save("remaped_fisheye.png").unwrap()
}
