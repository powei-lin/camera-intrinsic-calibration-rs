use nalgebra as na;
pub struct RvecTvec {
    pub rvec: na::DVector<f64>,
    pub tvec: na::DVector<f64>,
}

impl RvecTvec {
    pub fn new(rvec: na::DVector<f64>, tvec: na::DVector<f64>) -> RvecTvec {
        RvecTvec { rvec, tvec }
    }
    pub fn to_na_isometry3(&self) -> na::Isometry3<f64> {
        let tvec = na::Vector3::new(self.tvec[0], self.tvec[1], self.tvec[2]);
        let rvec = na::Vector3::new(self.rvec[0], self.rvec[1], self.rvec[2]);
        na::Isometry3::new(tvec, rvec)
    }
}
