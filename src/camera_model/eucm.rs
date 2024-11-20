use super::generic::CameraModel;
use nalgebra as na;

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
    fn project_one_impl(params: &na::DVector<T>, pt: &na::Vector3<T>) -> na::Vector2<T> {
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
    fn unproject_one_impl(params: &na::DVector<T>, pt: &na::Vector2<T>) -> na::Vector3<T> {
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
}

impl CameraModel<f64> for EUCM<f64> {
    fn params(&self) -> nalgebra::DVector<f64> {
        na::dvector![self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta]
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

    fn unproject_one(&self, pt: &nalgebra::Vector2<f64>) -> nalgebra::Vector3<f64> {
        Self::unproject_one_impl(&self.params(), pt)
    }
}
