use super::generic::CameraModel;
use nalgebra as na;

pub struct OpenCVModel5<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub k1: T,
    pub k2: T,
    pub p1: T,
    pub p2: T,
    pub k3: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> OpenCVModel5<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> OpenCVModel5<T> {
        OpenCVModel5 {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            k1: params[4].clone(),
            k2: params[5].clone(),
            p1: params[6].clone(),
            p2: params[7].clone(),
            k3: params[8].clone(),
            width,
            height,
        }
    }
    fn rd(&self, r: &T) -> T {
        let r2 = r.clone() * r.clone();
        let one = T::from_f64(1.0).unwrap();
        r.clone()
            * (one
                + self.k1.clone() * r2.clone()
                + self.k2.clone() * r2.clone() * r2.clone()
                + self.k3.clone() * r2.clone() * r2.clone() * r2)
    }
    fn rd_dr(&self, r: &T) -> T {
        let one = T::from_f64(1.0).unwrap();
        let three = T::from_f64(3.0).unwrap();
        let five = T::from_f64(5.0).unwrap();
        let seven = T::from_f64(7.0).unwrap();
        let r2 = r.clone() * r.clone();
        one + three * self.k1.clone() * r2.clone()
            + five * self.k2.clone() * r2.clone() * r2.clone()
            + seven * self.k3.clone() * r2.clone() * r2.clone() * r2
    }
    fn tangential_distort(&self, xn: &T, yn: &T) -> (T, T) {
        let r2 = xn.clone() * xn.clone() + yn.clone() * yn.clone();
        let r = r2.clone().sqrt();
        let d = self.rd(&r) / r;
        let two = T::from_f64(2.0).unwrap();
        let xd = xn.clone() * d.clone()
            + two.clone() * self.p1.clone() * xn.clone() * yn.clone()
            + self.p2.clone() * (r2.clone() + two.clone() * xn.clone() * xn.clone());
        let yd = yn.clone() * d
            + two.clone() * self.p1.clone() * (r2 + two * yn.clone() * yn.clone())
            + self.p2.clone() * xn.clone() * yn.clone();
        (xd, yd)
    }
}

impl<T: na::RealField + Clone> CameraModel<T> for OpenCVModel5<T> {
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.k1.clone(),
            self.k2.clone(),
            self.p1.clone(),
            self.p2.clone(),
            self.k3.clone(),
        ]
    }

    fn width(&self) -> T {
        T::from_u32(self.width).unwrap()
    }

    fn height(&self) -> T {
        T::from_u32(self.height).unwrap()
    }

    fn project_one(&self, pt: &nalgebra::Vector3<T>) -> nalgebra::Vector2<T> {
        let xn = pt[0].clone() / pt[2].clone();
        let yn = pt[1].clone() / pt[2].clone();
        let one = T::from_f64(1.0).unwrap();
        let two = T::from_f64(2.0).unwrap();
        let r2 = xn.clone() * xn.clone() + yn.clone() * yn.clone();
        let r4 = r2.clone() * r2.clone();
        let r6 = r4.clone() * r2.clone();
        let d = one + self.k1.clone() * r2.clone() + self.k2.clone() * r4 + self.k3.clone() * r6;
        let px = self.fx.clone()
            * (xn.clone() * d.clone()
                + two.clone() * self.p1.clone() * xn.clone() * yn.clone()
                + self.p2.clone() * (r2.clone() + two.clone() * xn.clone() * xn.clone()))
            + self.cx.clone();
        let py = self.fy.clone()
            * (yn.clone() * d
                + self.p1.clone() * (r2.clone() + two.clone() * yn.clone() * yn.clone())
                + two * self.p2.clone() * xn * yn)
            + self.cy.clone();

        na::Vector2::new(px, py)
    }

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let xd = (pt[0].clone() - self.cx.clone()) / self.fx.clone();
        let yd = (pt[1].clone() - self.cy.clone()) / self.fy.clone();
        let threshold0 = T::from_f64(1e-6).unwrap();
        let threshold1 = T::from_f64(1e-12).unwrap();
        let zero = T::from_f64(0.0).unwrap();
        let one = T::from_f64(1.0).unwrap();
        let rd_2 = xd.clone() * xd.clone() + yd.clone() * yd.clone();
        let rd = rd_2.sqrt();
        let mut r = rd.clone();
        if rd.clone() > threshold0.clone() {
            for _ in 0..5 {
                let r_next = r.clone() - (self.rd(&r) - rd.clone()) / self.rd_dr(&r);
                if (r_next.clone() - r).abs() < threshold0.clone() {
                    r = r_next.clone();
                    break;
                }
                r = r_next;
            }
            let d = self.rd(&r) / r;
            let mut xn = xd.clone() / d.clone();
            let mut yn = yd.clone() / d;
            let max_iter = 10;
            for _ in 0..max_iter {
                let (temp_dx, temp_dy) = self.tangential_distort(&xn, &yn);
                let (step_x, step_y) = (temp_dx - xd.clone(), temp_dy - yd.clone());
                (xn, yn) = (xn - step_x.clone(), yn - step_y.clone());
                if (step_x.clone() * step_x + step_y.clone() * step_y) < threshold1 {
                    break;
                }
            }
            na::Vector3::new(xn, yn, one)
        } else {
            na::Vector3::new(zero.clone(), zero, one)
        }
    }
}
