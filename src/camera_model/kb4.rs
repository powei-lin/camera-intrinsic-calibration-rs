use super::generic::CameraModel;
use nalgebra as na;
use num_traits::FromPrimitive;
use rayon::prelude::*;

pub struct OpenCVFisheye<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub k1: T,
    pub k2: T,
    pub k3: T,
    pub k4: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> OpenCVFisheye<T> {
    pub fn new(
        camera_matrix: &na::Matrix3<T>,
        distortion: &na::DMatrix<T>,
        width: u32,
        height: u32,
    ) -> OpenCVFisheye<T> {
        OpenCVFisheye {
            fx: camera_matrix[(0, 0)].clone(),
            fy: camera_matrix[(1, 1)].clone(),
            cx: camera_matrix[(0, 2)].clone(),
            cy: camera_matrix[(1, 2)].clone(),
            k1: distortion[0].clone(),
            k2: distortion[1].clone(),
            k3: distortion[2].clone(),
            k4: distortion[3].clone(),
            width,
            height,
        }
    }
    fn f(k1: &T, k2: &T, k3: &T, k4: &T, theta: &T) -> T {
        let theta2 = theta.clone() * theta.clone();
        let theta4 = theta2.clone() * theta2.clone();
        let theta6 = theta2.clone() * theta4.clone();
        let theta8 = theta2.clone() * theta6.clone();

        theta.clone()
            * (T::from_f64(1.0).unwrap()
                + k1.clone() * theta2
                + k2.clone() * theta4
                + k3.clone() * theta6
                + k4.clone() * theta8)
    }
    fn df_dtheta(k1: &T, k2: &T, k3: &T, k4: &T, theta: T) -> T {
        let theta2 = theta.clone() * theta;
        let theta4 = theta2.clone() * theta2.clone();
        let theta6 = theta2.clone() * theta4.clone();
        let theta8 = theta2.clone() * theta6.clone();
        T::from_f64(1.0).unwrap()
            + T::from_f64(3.0).unwrap() * k1.clone() * theta2
            + T::from_f64(5.0).unwrap() * k2.clone() * theta4
            + T::from_f64(7.0).unwrap() * k3.clone() * theta6
            + T::from_f64(9.0).unwrap() * k4.clone() * theta8
    }
    fn project_one_impl(params: &na::DVector<T>, pt: &na::Vector3<T>) -> na::Vector2<T> {
        let xn = pt[0].clone() / pt[2].clone();
        let yn = pt[1].clone() / pt[2].clone();
        let r2 = xn.clone() * xn.clone() + yn.clone() * yn.clone();
        let r = r2.sqrt();
        let theta = r.clone().atan();
        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let k1 = &params[4];
        let k2 = &params[5];
        let k3 = &params[6];
        let k4 = &params[7];
        let theta_d = Self::f(k1, k2, k3, k4, &theta);
        let d = theta_d / r.clone();
        let px = fx.clone() * (xn * d.clone()) + cx.clone();
        let py = fy.clone() * (yn * d) + cy.clone();
        na::Vector2::new(px, py)
    }
}
// // num_dual::DualDVec64

impl CameraModel<f64> for OpenCVFisheye<f64> {
    fn params(&self) -> nalgebra::DVector<f64> {
        na::dvector![self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4]
    }
    fn project_one(&self, pt: &nalgebra::Vector3<f64>) -> nalgebra::Vector2<f64> {
        Self::project_one_impl(&self.params(), pt)
    }

    fn width(&self) -> f64 {
        self.width as f64
    }

    fn height(&self) -> f64 {
        self.height as f64
    }

    // fn unproject(&self, p2d: &[nalgebra::Point2<f64>]) -> Vec<Option<(f32, f32)>> {
    //     p2d.into_par_iter()
    //         .map(|p| {
    //             if p.x < 0.0 || p.y < 0.0 || p.x >= self.width as f64 || p.y >= self.height as f64 {
    //                 return None;
    //             }
    //             let xd = (p.x - self.cx) / self.fx;
    //             let yd = (p.y - self.cy) / self.fy;

    //             let theta_d_2 = xd * xd + yd * yd;
    //             let theta_d = theta_d_2.sqrt();
    //             let mut theta = theta_d;
    //             if theta > 1e-6 {
    //                 for _ in 0..5 {
    //                     let theta_next = theta - (self.f(theta) - theta_d) / self.df_dtheta(theta);
    //                     if (theta_next - theta).abs() < 1e-6 {
    //                         theta = theta_next;
    //                         break;
    //                     }
    //                     theta = theta_next;
    //                 }
    //                 let scaling = theta.tan() / theta_d;
    //                 Some(((xd * scaling) as f32, (yd * scaling) as f32))
    //             } else {
    //                 Some((0.0, 0.0))
    //             }
    //         })
    //         .collect()
    // }
}

impl CameraModel<num_dual::DualDVec64> for OpenCVFisheye<num_dual::DualDVec64> {
    fn params(&self) -> nalgebra::DVector<num_dual::DualDVec64> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.k1.clone(),
            self.k2.clone(),
            self.k3.clone(),
            self.k4.clone()
        ]
    }
    fn project_one(
        &self,
        pt: &nalgebra::Vector3<num_dual::DualDVec64>,
    ) -> nalgebra::Vector2<num_dual::DualDVec64> {
        Self::project_one_impl(&self.params(), pt)
    }

    fn width(&self) -> num_dual::DualDVec64 {
        num_dual::DualDVec64::from_u32(self.width).unwrap()
    }

    fn height(&self) -> num_dual::DualDVec64 {
        num_dual::DualDVec64::from_u32(self.height).unwrap()
    }
}
