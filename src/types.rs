use std::io::Write;

use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvecTvec {
    rvec: (f64, f64, f64),
    tvec: (f64, f64, f64),
}

impl RvecTvec {
    pub fn new(rvec: &na::DVector<f64>, tvec: &na::DVector<f64>) -> RvecTvec {
        RvecTvec {
            rvec: (rvec[0], rvec[1], rvec[2]),
            tvec: (tvec[0], tvec[1], tvec[2]),
        }
    }
    pub fn to_na_isometry3(&self) -> na::Isometry3<f64> {
        na::Isometry3::new(self.na_tvec().to_vec3(), self.na_rvec().to_vec3())
    }
    pub fn na_rvec(&self) -> na::DVector<f64> {
        na::dvector![self.rvec.0, self.rvec.1, self.rvec.2]
    }
    pub fn na_tvec(&self) -> na::DVector<f64> {
        na::dvector![self.tvec.0, self.tvec.1, self.tvec.2]
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Extrinsics {
    rtvecs: Vec<RvecTvec>,
}

impl Extrinsics {
    pub fn new(rtvecs: &[RvecTvec]) -> Extrinsics {
        Extrinsics {
            rtvecs: rtvecs.to_vec(),
        }
    }
}

pub trait ToRvecTvec {
    fn to_rvec_tvec(&self) -> RvecTvec;
}
impl ToRvecTvec for na::Isometry3<f64> {
    fn to_rvec_tvec(&self) -> RvecTvec {
        let rvec = self.rotation.scaled_axis().to_dvec();
        let tvec = na::dvector![self.translation.x, self.translation.y, self.translation.z,];
        RvecTvec::new(&rvec, &tvec)
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
