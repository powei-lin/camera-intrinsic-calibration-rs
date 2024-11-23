use std::collections::HashMap;

use camera_intrinsic::camera_model::generic::{
    estimate_new_camera_matrix_for_undistort, init_undistort_map, remap, CameraModel, GenericModel,
};
use camera_intrinsic::camera_model::{model_from_json, KannalaBrandt4, EUCM};
use image::ImageReader;
use nalgebra as na;
use tiny_solver::factors::Factor;
use tiny_solver::Optimizer;

pub struct CustomFactor {
    pub model0: GenericModel<f64>,
    pub model1: GenericModel<f64>,
    pub p3ds: Vec<na::Vector3<f64>>,
}

impl Factor for CustomFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        let kb4_model: KannalaBrandt4<num_dual::DualVec<f64, f64, nalgebra::Dyn>> =
            KannalaBrandt4::new(
                &params[0],
                self.model0.width().round() as u32,
                self.model0.height().round() as u32,
            );
        let p3ds: Vec<_> = self
            .p3ds
            .iter()
            .map(|p| {
                na::Vector3::new(
                    num_dual::DualDVec64::from_re(p.x),
                    num_dual::DualDVec64::from_re(p.y),
                    num_dual::DualDVec64::from_re(p.z),
                )
            })
            .collect();
        let p2ds0 = self.model0.cast().project(&p3ds);
        let p2ds1 = kb4_model.project(&p3ds);
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
    let model = model_from_json("data/eucm.json");
    let mut p2ds = Vec::new();
    for r in (10..model.height() as u32 - 10).step_by(3) {
        for c in (10..model.width() as u32 - 10).step_by(3) {
            p2ds.push(na::Vector2::new(c as f64, r as f64));
        }
    }
    let p3ds = model.unproject(&p2ds);
    let p3ds: Vec<_> = p3ds.iter().filter_map(|p| p.clone()).collect();
    // init problem (factor graph)
    let mut problem = tiny_solver::Problem::new();

    // add custom residual for x and yz
    problem.add_residual_block(
        p3ds.len() * 2,
        vec![("x".to_string(), 8)],
        Box::new(CustomFactor {
            model0: model,
            model1: model,
            p3ds,
        }),
        None,
    );

    let params = model.params();
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
        model.width() as u32,
        model.height() as u32,
    );
    let new_w_h = 1024;
    let p = estimate_new_camera_matrix_for_undistort(&fisheye, 1.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = init_undistort_map(&fisheye, &p, (new_w_h, new_w_h));
    let remaped = remap(&img, &xmap, &ymap);
    remaped.save("remaped_fisheye.png").unwrap()
}
