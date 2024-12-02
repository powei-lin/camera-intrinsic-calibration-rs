use camera_intrinsic_model::*;
use nalgebra as na;
use num_dual::DualDVec64;
use tiny_solver::factors::Factor;

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
        let model = self.target.new_from_params(&params[0]);
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

pub struct UCMInitFocalAlphaFactor {
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
        let model = self.target.new_from_params(&cam_params);
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

pub struct ReprojectionFactor {
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
        let model = self.target.new_from_params(&params[0]);
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
