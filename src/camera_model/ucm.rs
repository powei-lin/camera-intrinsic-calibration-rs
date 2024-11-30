use super::generic::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct UCM<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub alpha: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> UCM<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> UCM<T> {
        if params.shape() != (5, 1) {
            panic!("the length of the vector should be 5");
        }
        UCM {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            alpha: params[4].clone(),
            width,
            height,
        }
    }
    pub fn from<U: na::RealField + Clone>(m: &UCM<U>) -> UCM<T> {
        UCM::new(&m.cast(), m.width, m.height)
    }
    pub fn zeros() -> UCM<T> {
        UCM {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            alpha: T::from_f64(0.2).unwrap(),
            width: 0,
            height: 0,
        }
    }
}

impl<T: na::RealField + Clone> ModelCast<T> for UCM<T> {}
impl<T: na::RealField + Clone> CameraModel<T> for UCM<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        self.alpha = params[4].clone();
    }
    #[inline]
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.alpha.clone(),
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

        let x = pt[0].clone();
        let y = pt[1].clone();
        let z = pt[2].clone();

        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let rho2 = r2.clone() + z.clone() * z.clone();
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
        let one = T::from_f64(1.0).unwrap();
        let xi = alpha.clone() / (one.clone() - alpha.clone());

        let mxx = (pt[0].clone() - cx.clone()) / fx.clone();
        let myy = (pt[1].clone() - cy.clone()) / fy.clone();

        let mx = (one.clone() - alpha.clone()) * mxx;
        let my = (one.clone() - alpha.clone()) * myy;

        let r2 = mx.clone() * mx.clone() + my.clone() * my.clone();

        let xi2 = xi.clone() * xi.clone();
        let n = (one.clone() + (one.clone() - xi2) * r2.clone()).sqrt();
        let m = one.clone() + r2;

        let k = (xi.clone() + n) / m;
        let z = k.clone() - xi;

        na::Vector3::new(k.clone() * mx / z.clone(), k * my / z, one)
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
        na::dvector![self.alpha.clone()]
    }
    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }
    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))> {
        // alpha [0, 1]
        vec![(4, (0.0, 1.0))]
    }
}
