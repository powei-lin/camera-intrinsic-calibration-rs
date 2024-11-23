use super::{KannalaBrandt4, OpenCVModel5, EUCM, UCM};
use image::DynamicImage;
use nalgebra as na;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum GenericModel<T: na::RealField> {
    EUCM(EUCM<T>),
    UCM(UCM<T>),
    OpenCVModel5(OpenCVModel5<T>),
    KannalaBrandt4(KannalaBrandt4<T>),
}
impl GenericModel<f64> {
    pub fn init_undistort_map(
        &self,
        projection_mat: &na::Matrix3<f64>,
        new_w_h: (u32, u32),
    ) -> (na::DMatrix<f32>, na::DMatrix<f32>) {
        match self {
            GenericModel::EUCM(eucm) => init_undistort_map(eucm, projection_mat, new_w_h),
            GenericModel::UCM(ucm) => init_undistort_map(ucm, projection_mat, new_w_h),
            GenericModel::OpenCVModel5(open_cvmodel5) => {
                init_undistort_map(open_cvmodel5, projection_mat, new_w_h)
            }
            GenericModel::KannalaBrandt4(kannala_brandt4) => {
                init_undistort_map(kannala_brandt4, projection_mat, new_w_h)
            }
        }
    }
    pub fn estimate_new_camera_matrix_for_undistort(
        &self,
        balance: f64,
        new_image_w_h: Option<(u32, u32)>,
    ) -> na::Matrix3<f64> {
        match self {
            GenericModel::EUCM(eucm) => {
                estimate_new_camera_matrix_for_undistort(eucm, balance, new_image_w_h)
            }
            GenericModel::UCM(ucm) => {
                estimate_new_camera_matrix_for_undistort(ucm, balance, new_image_w_h)
            }
            GenericModel::OpenCVModel5(open_cvmodel5) => {
                estimate_new_camera_matrix_for_undistort(open_cvmodel5, balance, new_image_w_h)
            }
            GenericModel::KannalaBrandt4(kannala_brandt4) => {
                estimate_new_camera_matrix_for_undistort(kannala_brandt4, balance, new_image_w_h)
            }
        }
    }
}

macro_rules! remap_impl {
    ($src:expr, $map0:expr, $map1:expr, $($img_type:ident => ($inner_type:ident, $default_value:expr)),*) => {
        match $src {
            $(
                DynamicImage::$img_type(img) => {
                    let (r, c) = $map0.shape();
                    let out_img = image::ImageBuffer::from_par_fn(c as u32, r as u32, |x, y| {
                        let idx = y as usize * c + x as usize;
                        let (x_cor, y_cor) = unsafe { ($map0.get_unchecked(idx), $map1.get_unchecked(idx)) };
                        if x_cor.is_nan() || y_cor.is_nan() {
                            return image::$inner_type($default_value);
                        }
                        image::imageops::interpolate_bilinear(img, *x_cor, *y_cor)
                            .unwrap_or(image::$inner_type($default_value))
                    });
                    DynamicImage::$img_type(out_img)
                }
            )*
            _ => {
                panic!("Not support this image type.");
            }
        }
    };
}

pub fn remap(src: &DynamicImage, map0: &na::DMatrix<f32>, map1: &na::DMatrix<f32>) -> DynamicImage {
    remap_impl!(src, map0, map1,
        ImageLuma8 => (Luma, [0]),
        ImageLumaA8 => (LumaA, [0, 0]),
        ImageLuma16 => (Luma, [0]),
        ImageLumaA16 => (LumaA, [0, 0]),
        ImageRgb8 => (Rgb, [0, 0, 0]),
        ImageRgba8 => (Rgba, [0, 0, 0, 0]),
        ImageRgb16 => (Rgb, [0, 0, 0]),
        ImageRgba16 => (Rgba, [0, 0, 0, 0]),
        ImageRgb32F => (Rgb, [0.0, 0.0, 0.0]),
        ImageRgba32F => (Rgba, [0.0, 0.0, 0.0, 0.0])
    )
}

pub trait ModelCast<T: na::RealField + Clone>: CameraModel<T> {
    fn cast<U: na::RealField>(&self) -> na::DVector<U> {
        let v: Vec<_> = self
            .params()
            .iter()
            .map(|i| U::from_f64(i.to_subset().unwrap()).unwrap())
            .collect();
        na::DVector::from_vec(v)
    }
}

pub trait CameraModel<T: na::RealField + Clone>
where
    Self: Sync,
{
    fn params(&self) -> na::DVector<T>;
    fn width(&self) -> T;
    fn height(&self) -> T;
    fn project_one(&self, pt: &na::Vector3<T>) -> na::Vector2<T>;
    fn project(&self, p3d: &[na::Vector3<T>]) -> Vec<Option<na::Vector2<T>>> {
        p3d.par_iter()
            .map(|pt| {
                let p2d = self.project_one(pt);
                if p2d[0] < T::from_f64(0.0).unwrap()
                    || p2d[0] > self.width()
                    || p2d[1] < T::from_f64(0.0).unwrap()
                    || p2d[1] > self.height()
                {
                    None
                } else {
                    Some(p2d)
                }
            })
            .collect()
    }
    fn unproject_one(&self, pt: &na::Vector2<T>) -> na::Vector3<T>;
    fn unproject(&self, p2d: &[na::Vector2<T>]) -> Vec<Option<na::Vector3<T>>> {
        p2d.par_iter()
            .map(|pt| {
                if pt[0] < T::from_f64(0.0).unwrap()
                    || pt[0] > self.width() - T::from_f64(1.0).unwrap()
                    || pt[1] < T::from_f64(0.0).unwrap()
                    || pt[1] > self.height() - T::from_f64(1.0).unwrap()
                {
                    None
                } else {
                    let p3d = self.unproject_one(pt);
                    Some(p3d)
                }
            })
            .collect()
    }
}

pub fn init_undistort_map(
    camera_model: &dyn CameraModel<f64>,
    projection_mat: &na::Matrix3<f64>,
    new_w_h: (u32, u32),
) -> (na::DMatrix<f32>, na::DMatrix<f32>) {
    if projection_mat.shape() != (3, 3) {
        panic!("projection matrix has the wrong shape");
    }
    let fx = projection_mat[(0, 0)];
    let fy = projection_mat[(1, 1)];
    let cx = projection_mat[(0, 2)];
    let cy = projection_mat[(1, 2)];
    let p3ds: Vec<na::Vector3<f64>> = (0..new_w_h.1)
        .into_par_iter()
        .flat_map(|y| {
            (0..new_w_h.0)
                .into_par_iter()
                .map(|x| na::Vector3::new((x as f64 - cx) / fx, (y as f64 - cy) / fy, 1.0))
                .collect::<Vec<na::Vector3<f64>>>()
        })
        .collect();
    let p2ds = camera_model.project(&p3ds);
    let (xvec, yvec): (Vec<f32>, Vec<f32>) = p2ds
        .par_iter()
        .map(|xy| {
            if let Some(xy) = xy {
                (xy[0] as f32, xy[1] as f32)
            } else {
                (f32::NAN, f32::NAN)
            }
        })
        .unzip();
    let xmap = na::DMatrix::from_vec(new_w_h.1 as usize, new_w_h.0 as usize, xvec);
    let ymap = na::DMatrix::from_vec(new_w_h.1 as usize, new_w_h.0 as usize, yvec);
    (xmap, ymap)
}

pub fn estimate_new_camera_matrix_for_undistort(
    camera_model: &dyn CameraModel<f64>,
    balance: f64,
    new_image_w_h: Option<(u32, u32)>,
) -> na::Matrix3<f64> {
    if !(0.0..=1.0).contains(&balance) {
        panic!("balance should be [0.0-1.0], got {}", balance);
    }
    let params = camera_model.params();
    let cx = params[2];
    let cy = params[3];
    let w = camera_model.width();
    let h = camera_model.height();
    let p2ds = vec![
        na::Vector2::new(cx, 0.0),
        na::Vector2::new(w - 1.0, cy),
        na::Vector2::new(cx, h - 1.0),
        na::Vector2::new(0.0, cy),
    ];
    let undist_pts = camera_model.unproject(&p2ds);
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for p in undist_pts {
        let p = p.unwrap();
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    min_x = min_x.abs();
    min_y = min_y.abs();
    let (new_w, new_h) = if let Some((new_w, new_h)) = new_image_w_h {
        (new_w, new_h)
    } else {
        (camera_model.width() as u32, camera_model.height() as u32)
    };
    let new_cx = new_w as f64 * min_x / (min_x + max_x);
    let new_cy = new_h as f64 * min_y / (min_y + max_y);
    let fx = new_w as f64 / (min_x + max_x);
    let fy = new_h as f64 / (min_y + max_y);
    let fmin = fx.min(fy);
    let fmax = fx.max(fy);
    let new_f = balance * fmin + (1.0 - balance) * fmax;

    let mut out = na::Matrix3::identity();
    out[(0, 0)] = new_f;
    out[(1, 1)] = new_f;
    out[(0, 2)] = new_cx;
    out[(1, 2)] = new_cy;
    out
}
