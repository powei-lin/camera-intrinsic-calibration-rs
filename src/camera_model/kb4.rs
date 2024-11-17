use super::generic::CameraModel;
use nalgebra as na;
use rayon::prelude::*;
use std::ops::Add;

pub struct OpenCVFisheye {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
    pub width: u32,
    pub height: u32,
}
impl OpenCVFisheye {
    pub fn new(
        camera_matrix: &na::Matrix3<f64>,
        distortion: &na::DMatrix<f64>,
        width: u32,
        height: u32,
    ) -> OpenCVFisheye {
        OpenCVFisheye {
            fx: camera_matrix[(0, 0)],
            fy: camera_matrix[(1, 1)],
            cx: camera_matrix[(0, 2)],
            cy: camera_matrix[(1, 2)],
            k1: distortion[0],
            k2: distortion[1],
            k3: distortion[2],
            k4: distortion[3],
            width,
            height,
        }
    }
    fn f(&self, theta: f64) -> f64 {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta2 * theta4;
        let theta8 = theta2 * theta6;
        theta * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)
    }
    fn df_dtheta(&self, theta: f64) -> f64 {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta2 * theta4;
        let theta8 = theta2 * theta6;
        1.0 + 3.0 * self.k1 * theta2
            + 5.0 * self.k2 * theta4
            + 7.0 * self.k3 * theta6
            + 9.0 * self.k4 * theta8
    }
    fn project_one<T: na::RealField + Clone>(
        params: &na::DVector<T>,
        pt: &na::Vector3<T>,
    ) -> Option<na::Vector2<T>> {
        // let a = params[0] + params[1].clone();
        // let b = a.clone() * a.clone();
        let xn = pt[0].clone() / pt[2].clone();
        let yn = pt[1].clone() / pt[2].clone();
        xn.sin();
        // let r2 = xn * xn + yn * yn;
        // let r = r2.sqrt();
        // let theta = r.atan();
        // let theta_d = self.f(theta);
        // let d = theta_d / r;
        // let px = self.fx * (xn * d) + self.cx;
        // let py = self.fy * (yn * d) + self.cy;
        // if px < 0.0 || px > self.width as f64 || py < 0.0 || py > self.height as f64 {
        //     None
        // } else {
        //     Some((px as f32, py as f32))
        // }
        // na::dvector![na::Vector2::new(a.clone(), a)]
        None
    }
}
// num_dual::DualDVec64

impl CameraModel for OpenCVFisheye {
    fn project(&self, p3d: &[nalgebra::Point3<f64>]) -> Vec<Option<(f32, f32)>> {
        let param = na::dvector![3.0];
        let p3d0 = na::Vector3::new(1.0, 2.0, 3.0);
        // let param = na::dvector![num_dual::DualDVec64::from_re(0.0)];
        // let p3d0 = na::Vector3::new(
        //     num_dual::DualDVec64::from_re(0.0),
        //     num_dual::DualDVec64::from_re(0.0),
        //     num_dual::DualDVec64::from_re(0.0),
        // );

        Self::project_one(&param, &p3d0);
        p3d.par_iter()
            .map(|pt| {
                let xn = pt.x / pt.z;
                let yn: f64 = pt.y / pt.z;
                let r2 = xn * xn + yn * yn;
                let r = r2.sqrt();
                let theta = r.atan();
                let theta_d = self.f(theta);
                let d = theta_d / r;
                let px = self.fx * (xn * d) + self.cx;
                let py = self.fy * (yn * d) + self.cy;
                if px < 0.0 || px > self.width as f64 || py < 0.0 || py > self.height as f64 {
                    None
                } else {
                    Some((px as f32, py as f32))
                }
            })
            .collect()
    }

    fn unproject(&self, p2d: &[nalgebra::Point2<f64>]) -> Vec<Option<(f32, f32)>> {
        p2d.into_par_iter()
            .map(|p| {
                if p.x < 0.0 || p.y < 0.0 || p.x >= self.width as f64 || p.y >= self.height as f64 {
                    return None;
                }
                let xd = (p.x - self.cx) / self.fx;
                let yd = (p.y - self.cy) / self.fy;

                let theta_d_2 = xd * xd + yd * yd;
                let theta_d = theta_d_2.sqrt();
                let mut theta = theta_d;
                if theta > 1e-6 {
                    for _ in 0..5 {
                        let theta_next = theta - (self.f(theta) - theta_d) / self.df_dtheta(theta);
                        if (theta_next - theta).abs() < 1e-6 {
                            theta = theta_next;
                            break;
                        }
                        theta = theta_next;
                    }
                    let scaling = theta.tan() / theta_d;
                    Some(((xd * scaling) as f32, (yd * scaling) as f32))
                } else {
                    Some((0.0, 0.0))
                }
            })
            .collect()
    }
}
