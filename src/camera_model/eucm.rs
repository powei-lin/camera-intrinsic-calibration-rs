use super::generic::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct EUCM<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub alpha: T,
    pub beta: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> EUCM<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> EUCM<T> {
        if params.shape() != (6, 1) {
            panic!("the length of the vector should be 6");
        }
        EUCM {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            alpha: params[4].clone(),
            beta: params[5].clone(),
            width,
            height,
        }
    }
    pub fn from<U: na::RealField + Clone>(m: &EUCM<U>) -> EUCM<T> {
        EUCM::new(&m.cast(), m.width, m.height)
    }
    pub fn zeros() -> EUCM<T> {
        EUCM {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            alpha: T::from_f64(0.4).unwrap(),
            beta: T::from_f64(1.0).unwrap(),
            width: 0,
            height: 0,
        }
    }
}

impl<T: na::RealField + Clone> ModelCast<T> for EUCM<T> {}

impl<T: na::RealField + Clone> CameraModel<T> for EUCM<T> {
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
    }
    #[inline]
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.alpha.clone(),
            self.beta.clone()
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

        let x = pt[0].clone();
        let y = pt[1].clone();
        let z = pt[2].clone();

        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let rho2 = beta.clone() * r2.clone() + z.clone() * z.clone();
        let rho = rho2.sqrt();

        let norm = alpha.clone() * rho + (T::from_f64(1.0).unwrap() - alpha.clone()) * z;

        let mx = x / norm.clone();
        let my = y / norm.clone();

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
        let one = T::from_f64(1.0).unwrap();

        let mx = (pt[0].clone() - cx.clone()) / fx.clone();
        let my = (pt[1].clone() - cy.clone()) / fy.clone();

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
        na::dvector![self.alpha.clone(), self.beta.clone()]
    }

    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }

    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))> {
        // alpha [0, 1], beta > 0
        vec![(4, (0.0, 1.0)), (5, (0.0, f64::MAX))]
    }
}
