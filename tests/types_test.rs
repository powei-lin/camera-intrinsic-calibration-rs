use camera_intrinsic_calibration::types::{RvecTvec, ToRvecTvec};
use nalgebra as na;

#[test]
fn test_rvec_tvec_conversion() {
    let rvec_in = na::dvector![0.1, 0.2, 0.3];
    let tvec_in = na::dvector![1.0, 2.0, 3.0];
    
    let rt = RvecTvec::new(&rvec_in, &tvec_in);
    
    let iso = rt.to_na_isometry3();
    
    let rt_back = iso.to_rvec_tvec();
    
    let r_out = rt_back.na_rvec();
    let t_out = rt_back.na_tvec();
    
    assert!((r_out - rvec_in).norm() < 1e-6);
    assert!((t_out - tvec_in).norm() < 1e-6);
}
