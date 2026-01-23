use crate::types::DVecVec3;

use camera_intrinsic_model::*;
use nalgebra as na;
use tiny_solver::factors::Factor;

/// Factor for converting one camera model to another by minimizing reprojection error of a dense grid.
///
/// This factor is used to find parameters of `target` model that best approximate the `source` model.
#[derive(Clone)]
pub struct ModelConvertFactor {
    pub source: GenericModel<f64>,
    pub target: GenericModel<f64>,
    pub p3ds: Vec<na::Vector3<f64>>,
}

impl ModelConvertFactor {
    /// Creates a new conversion factor.
    ///
    /// Generates a grid of 2D points in the source image, unprojects them to 3D,
    /// and stores them for optimizing the target model.
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
        let p3ds: Vec<_> = p3ds.iter().filter_map(|p| p.as_ref().map(|pp| pp.cast())).collect();
        ModelConvertFactor { source: source.cast(), target: target.cast(), p3ds }
    }
    pub fn residaul_num(&self) -> usize {
        self.p3ds.len() * 2
    }
}

impl<T: na::RealField> Factor<T> for ModelConvertFactor {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        let model = self.target.cast::<T>().new_from_params(&params[0]);
        let p3d_template: Vec<_> = self.p3ds.iter().map(|&p| p.cast::<T>()).collect();
        let p2ds0 = self.source.cast::<T>().project(&p3d_template);
        let p2ds1 = model.project(&p3d_template);
        let diff: Vec<_> = p2ds0
            .iter()
            .zip(p2ds1)
            .flat_map(|(p0, p1)| {
                if let Some(p0) = p0
                    && let Some(p1) = p1
                {
                    let pp = p0 - p1;
                    return vec![pp[0].clone(), pp[1].clone()];
                }
                vec![T::from_f64(10000.0).unwrap(), T::from_f64(10000.0).unwrap()]
            })
            .collect();
        na::DVector::from_vec(diff)
    }
}

/// Factor for initializing focal length and alpha for UCM model.
///
/// Optimizes focal length (fx=fy) and alpha while keeping other params fixed?
/// Actually it seems to optimize f and alpha, plus extrinsic rvec/tvec?
pub struct UCMInitFocalAlphaFactor {
    pub target: GenericModel<f64>,
    pub p3d: na::Point3<f64>,
    pub p2d: na::Vector2<f64>,
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
impl<T: na::RealField> Factor<T> for UCMInitFocalAlphaFactor {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        // params[[f, alpha], rvec, tvec]
        let mut cam_params = self.target.cast::<T>().params();
        cam_params[0] = params[0][0].clone();
        cam_params[1] = params[0][0].clone();
        cam_params[4] = params[0][1].clone();
        let model = self.target.cast().new_from_params(&cam_params);
        let rvec = params[1].to_vec3();
        let tvec = params[2].to_vec3();
        let transform = na::Isometry3::new(tvec, rvec);
        let p3d_t = transform * self.p3d.cast();
        let p3d_t = na::Vector3::new(p3d_t.x.clone(), p3d_t.y.clone(), p3d_t.z.clone());
        let p2d_p = model.project_one(&p3d_t);
        let p2d_tp = self.p2d.cast::<T>();
        na::dvector![p2d_p[0].clone() - p2d_tp[0].clone(), p2d_p[1].clone() - p2d_tp[1].clone()]
    }
}

/// Standard reprojection error factor.
///
/// Minimizes the distance between projected 3D point and observed 2D point.
/// Optimizes camera intrinsics and extrinsics (rvec, tvec).
pub struct ReprojectionFactor {
    pub target: GenericModel<f64>,
    pub p3d: na::Point3<f64>,
    pub p2d: na::Vector2<f64>,
    /// If true, constraints fx = fy (focal length x equals focal length y).
    pub xy_same_focal: bool,
}

impl ReprojectionFactor {
    pub fn new(
        target: &GenericModel<f64>,
        p3d: &glam::Vec3,
        p2d: &glam::Vec2,
        xy_same_focal: bool,
    ) -> ReprojectionFactor {
        let target = target.cast();
        let p3d = na::Point3::new(p3d.x, p3d.y, p3d.z).cast();
        let p2d = na::Vector2::new(p2d.x, p2d.y).cast();
        ReprojectionFactor { target, p3d, p2d, xy_same_focal }
    }
}
impl<T: na::RealField> Factor<T> for ReprojectionFactor {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        // params[params, rvec, tvec]
        let mut params0 = params[0].clone();
        if self.xy_same_focal {
            params0 = params0.clone().insert_row(1, params0[0].clone());
        }
        let model = self.target.cast().new_from_params(&params0);
        let rvec = params[1].to_vec3();
        let tvec = params[2].to_vec3();
        let transform = na::Isometry3::new(tvec, rvec);
        let p3d_t = transform * self.p3d.cast();
        let p3d_t = na::Vector3::new(p3d_t.x.clone(), p3d_t.y.clone(), p3d_t.z.clone());
        let p2d_p = model.project_one(&p3d_t);

        let p2d_tp = self.p2d.cast::<T>();
        na::dvector![p2d_p[0].clone() - p2d_tp[0].clone(), p2d_p[1].clone() - p2d_tp[1].clone()]
    }
}

/// Reprojection factor involving another camera's extrinsic.
///
/// Used for multi-camera calibration or when chaining transforms.
/// T_cam = T_i_0 * T_0_b
pub struct OtherCamReprojectionFactor {
    pub target: GenericModel<f64>,
    pub p3d: na::Point3<f64>,
    pub p2d: na::Vector2<f64>,
    pub xy_same_focal: bool,
}

impl OtherCamReprojectionFactor {
    pub fn new(
        target: &GenericModel<f64>,
        p3d: &glam::Vec3,
        p2d: &glam::Vec2,
        xy_same_focal: bool,
    ) -> OtherCamReprojectionFactor {
        let target = target.cast();
        let p3d = na::Point3::new(p3d.x, p3d.y, p3d.z).cast();
        let p2d = na::Vector2::new(p2d.x, p2d.y).cast();
        OtherCamReprojectionFactor { target, p3d, p2d, xy_same_focal }
    }
}
impl<T: na::RealField> Factor<T> for OtherCamReprojectionFactor {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        // params[params, rvec, tvec]
        let mut params0 = params[0].clone();
        if self.xy_same_focal {
            params0 = params0.clone().insert_row(1, params0[0].clone());
        }
        let model = self.target.cast().new_from_params(&params0);
        let rvec0 = params[1].to_vec3();
        let tvec0 = params[2].to_vec3();
        let t_0_b = na::Isometry3::new(tvec0, rvec0);
        let rvec1 = params[3].to_vec3();
        let tvec1 = params[4].to_vec3();
        let t_i_0 = na::Isometry3::new(tvec1, rvec1);
        let p3d_t = t_i_0 * t_0_b * self.p3d.cast();
        let p3d_t = na::Vector3::new(p3d_t.x.clone(), p3d_t.y.clone(), p3d_t.z.clone());
        let p2d_p = model.project_one(&p3d_t);

        let p2d_tp = self.p2d.cast::<T>();
        na::dvector![p2d_p[0].clone() - p2d_tp[0].clone(), p2d_p[1].clone() - p2d_tp[1].clone()]
    }
}

/// SE3 (Pose) error factor.
///
/// Computes error between estimated relative pose and measurement.
/// Used for extrinsic calibration or loop closure factors.
pub struct SE3Factor {
    pub t_0_b: na::Isometry3<f64>,
    pub t_i_b: na::Isometry3<f64>,
}

impl SE3Factor {
    pub fn new(t_0_b: &na::Isometry3<f64>, t_i_b: &na::Isometry3<f64>) -> SE3Factor {
        SE3Factor { t_0_b: t_0_b.cast(), t_i_b: t_i_b.cast() }
    }
}

impl<T: na::RealField> Factor<T> for SE3Factor {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        let rvec =
            na::Vector3::new(params[0][0].clone(), params[0][1].clone(), params[0][2].clone());
        let tvec =
            na::Vector3::new(params[1][0].clone(), params[1][1].clone(), params[1][2].clone());
        let t_i_0 = na::Isometry3::new(tvec, rvec);
        let t_diff = self.t_i_b.cast().inverse() * t_i_0 * self.t_0_b.cast();
        let r_diff = t_diff.rotation.scaled_axis();
        na::dvector![
            r_diff[0].clone(),
            r_diff[1].clone(),
            r_diff[2].clone(),
            t_diff.translation.x.clone(),
            t_diff.translation.y.clone(),
            t_diff.translation.z.clone(),
        ]
    }
}
