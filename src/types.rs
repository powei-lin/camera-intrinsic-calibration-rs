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
        na::Isometry3::new(self.tvec.to_vec3(), self.rvec.to_vec3())
    }
}

pub trait ToRvecTvec {
    fn to_rvec_tvec(&self) -> RvecTvec;
}
impl ToRvecTvec for na::Isometry3<f64> {
    fn to_rvec_tvec(&self) -> RvecTvec {
        let rvec = self.rotation.scaled_axis().to_dvec();
        let tvec = na::dvector![self.translation.x, self.translation.y, self.translation.z,];
        RvecTvec { rvec, tvec }
    }
}

pub trait Vec3DVec<T: Clone> {
    fn to_dvec(&self) -> na::DVector<T>;
}
pub trait DVecVec3<T: Clone> {
    fn to_vec3(&self) -> nalgebra::Vector3<T>;
}
impl<T: Clone> DVecVec3<T> for na::DVector<T> {
    fn to_vec3(&self) -> nalgebra::Vector3<T> {
        na::Vector3::new(self[0].clone(), self[1].clone(), self[2].clone())
    }
}
impl<T: Clone> Vec3DVec<T> for na::Vector3<T> {
    fn to_dvec(&self) -> na::DVector<T> {
        na::dvector![self[0].clone(), self[1].clone(), self[2].clone()]
    }
}
