use nalgebra as na;
pub struct RvecTvec {
    pub rvec: na::DVector<f64>,
    pub tvec: na::DVector<f64>,
}

impl RvecTvec {
    pub fn new(rvec: na::DVector<f64>, tvec: na::DVector<f64>) -> RvecTvec {
        RvecTvec { rvec, tvec }
    }
}
