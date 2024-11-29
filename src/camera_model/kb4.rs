use super::generic::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
pub struct KannalaBrandt4<T: na::RealField + Clone> {
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
impl<T: na::RealField + Clone> ModelCast<T> for KannalaBrandt4<T> {}
impl<T: na::RealField + Clone> KannalaBrandt4<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> KannalaBrandt4<T> {
        KannalaBrandt4 {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            k1: params[4].clone(),
            k2: params[5].clone(),
            k3: params[6].clone(),
            k4: params[7].clone(),
            width,
            height,
        }
    }
    pub fn zeros() -> KannalaBrandt4<T> {
        KannalaBrandt4 {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            k1: T::zero(),
            k2: T::zero(),
            k3: T::zero(),
            k4: T::zero(),
            width: 0,
            height: 0,
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
    fn df_dtheta(k1: &T, k2: &T, k3: &T, k4: &T, theta: &T) -> T {
        let theta2 = theta.clone() * theta.clone();
        let theta4 = theta2.clone() * theta2.clone();
        let theta6 = theta2.clone() * theta4.clone();
        let theta8 = theta2.clone() * theta6.clone();
        T::from_f64(1.0).unwrap()
            + T::from_f64(3.0).unwrap() * k1.clone() * theta2
            + T::from_f64(5.0).unwrap() * k2.clone() * theta4
            + T::from_f64(7.0).unwrap() * k3.clone() * theta6
            + T::from_f64(9.0).unwrap() * k4.clone() * theta8
    }
    pub fn from<U: na::RealField + Clone>(m: &KannalaBrandt4<U>) -> KannalaBrandt4<T> {
        KannalaBrandt4::new(&m.cast(), m.width, m.height)
    }
}

impl<T: na::RealField + Clone> CameraModel<T> for KannalaBrandt4<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        self.k1 = params[4].clone();
        self.k2 = params[5].clone();
        self.k3 = params[6].clone();
        self.k4 = params[7].clone();
    }
    fn params(&self) -> nalgebra::DVector<T> {
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

    fn width(&self) -> T {
        T::from_u32(self.width).unwrap()
    }

    fn height(&self) -> T {
        T::from_u32(self.height).unwrap()
    }

    fn project_one(&self, pt: &nalgebra::Vector3<T>) -> nalgebra::Vector2<T> {
        let params = self.params();
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

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let xd = (pt[0].clone() - self.cx.clone()) / self.fx.clone();
        let yd = (pt[1].clone() - self.cy.clone()) / self.fy.clone();

        let theta_d_2 = xd.clone() * xd.clone() + yd.clone() * yd.clone();
        let theta_d = theta_d_2.sqrt();
        let mut theta = theta_d.clone();
        let theta_threshold = T::from_f64(1e-6).unwrap();
        let one = T::from_f64(1.0).unwrap();
        let zero = T::from_f64(0.0).unwrap();
        if theta > theta_threshold {
            for _ in 0..5 {
                let theta_next = theta.clone()
                    - (Self::f(&self.k1, &self.k2, &self.k3, &self.k4, &theta.clone())
                        - theta_d.clone())
                        / Self::df_dtheta(&self.k1, &self.k2, &self.k3, &self.k4, &theta);
                if (theta_next.clone() - theta).abs() < theta_threshold {
                    theta = theta_next.clone();
                    break;
                }
                theta = theta_next;
            }
            let scaling = theta.tan() / theta_d;
            na::Vector3::new(xd * scaling.clone(), yd * scaling, one)
        } else {
            na::Vector3::new(zero.clone(), zero, one)
        }
    }
    fn camera_params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone()
        ]
    }

    fn distortion_params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.k1.clone(),
            self.k2.clone(),
            self.k3.clone(),
            self.k4.clone()
        ]
    }
    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }
}
