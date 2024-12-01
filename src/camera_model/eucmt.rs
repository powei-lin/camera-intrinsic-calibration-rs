use super::generic::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct EUCMT<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub alpha: T,
    pub beta: T,
    pub t1: T,
    pub t2: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> EUCMT<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> EUCMT<T> {
        if params.shape() != (8, 1) {
            panic!("the length of the vector should be 8");
        }
        EUCMT {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            alpha: params[4].clone(),
            beta: params[5].clone(),
            t1: params[6].clone(),
            t2: params[7].clone(),
            width,
            height,
        }
    }
    pub fn from<U: na::RealField + Clone>(m: &EUCMT<U>) -> EUCMT<T> {
        EUCMT::new(&m.cast(), m.width, m.height)
    }
    pub fn zeros() -> EUCMT<T> {
        EUCMT {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            alpha: T::from_f64(0.4).unwrap(),
            beta: T::from_f64(1.0).unwrap(),
            t1: T::zero(),
            t2: T::zero(),
            width: 0,
            height: 0,
        }
    }
}

impl<T: na::RealField + Clone> ModelCast<T> for EUCMT<T> {}

impl<T: na::RealField + Clone> CameraModel<T> for EUCMT<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        self.alpha = params[4].clone();
        self.beta = params[5].clone();
        self.t1 = params[6].clone();
        self.t2 = params[7].clone();
    }
    #[inline]
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.alpha.clone(),
            self.beta.clone(),
            self.t1.clone(),
            self.t2.clone(),
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
        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let alpha = &params[4];
        let beta = &params[5];
        let t1 = &params[6];
        let t2 = &params[7];

        let x = pt[0].clone();
        let y = pt[1].clone();
        let z = pt[2].clone();

        let one = T::from_f64(1.0).unwrap();
        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let rho2 = beta.clone() * r2.clone() + z.clone() * z.clone();
        let rho = rho2.sqrt();

        let norm = alpha.clone() * rho + (T::from_f64(1.0).unwrap() - alpha.clone()) * z;

        let distort_x = x / norm.clone();
        let distort_y = y / norm;

        let denom = distort_x.clone() * t2.clone() - distort_y.clone() * t1.clone() + one;

        let mx = (distort_x - distort_y.clone() * t2.clone() * t1.clone()) / denom.clone();
        let my = distort_y / denom;

        na::Vector2::new(fx.clone() * mx + cx.clone(), fy.clone() * my + cy.clone())
    }

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let params = self.params();
        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let alpha = &params[4];
        let beta = &params[5];
        let t1 = &params[6];
        let t2 = &params[7];

        let one = T::from_f64(1.0).unwrap();

        let tilt_y = (pt[1].clone() - cy.clone()) / fy.clone();
        let tilt_x = (pt[0].clone() - cx.clone()) / fx.clone();

        let my = tilt_y.clone()
            / (one.clone() + tilt_y.clone() * t1.clone()
                - (tilt_x.clone() + tilt_y.clone() * t2.clone() * t1.clone()) * t2.clone());

        let mx = tilt_x * my.clone() / tilt_y + my.clone() * t2.clone() * t1.clone();

        // let mx = (pt[0].clone() - cx.clone()) / fx.clone();
        // let my = (pt[1].clone() - cy.clone()) / fy.clone();

        let r2 = mx.clone() * mx.clone() + my.clone() * my.clone();
        let gamma = one.clone() - alpha.clone();

        let tmp1 = one.clone() - alpha.clone() * alpha.clone() * beta.clone() * r2.clone();
        let tmp_sqrt = (one.clone() - (alpha.clone() - gamma.clone()) * beta.clone() * r2).sqrt();
        let tmp2 = alpha.clone() * tmp_sqrt + gamma;

        let k = tmp1 / tmp2;

        na::Vector3::new(mx / k.clone(), my / k, one)
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
            self.alpha.clone(),
            self.beta.clone(),
            self.t1.clone(),
            self.t2.clone()
        ]
    }

    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }

    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))> {
        // alpha [0, 1], beta > 0
        vec![
            (4, (0.0, 1.0)),
            (5, (0.0, f64::MAX)),
            (6, (-1e2, 1e2)),
            (7, (-1e2, 1e2)),
        ]
    }
}
