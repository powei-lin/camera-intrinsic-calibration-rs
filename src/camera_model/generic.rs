use image::{DynamicImage, GenericImage};
use nalgebra as na;
use rayon::prelude::*;

pub fn remap(src: &DynamicImage, map0: &na::DMatrix<f32>, map1: &na::DMatrix<f32>) -> DynamicImage {
    let (r, c) = map0.shape();
    let out = match src {
        DynamicImage::ImageLuma8(img) => {
            let out_img = image::GrayImage::from_par_fn(c as u32, r as u32, |x, y| {
                let idx = y as usize * c + x as usize;
                let (x_cor, y_cor) = unsafe { (map0.get_unchecked(idx), map1.get_unchecked(idx)) };
                if x_cor.is_nan() || y_cor.is_nan() {
                    return image::Luma([0]);
                }
                let x_cor = x_cor.round() as u32;
                let y_cor = y_cor.round() as u32;
                if x_cor >= img.width() || y_cor >= img.height() {
                    image::Luma([0])
                } else {
                    img.get_pixel(x_cor, y_cor).to_owned()
                }
            });
            DynamicImage::ImageLuma8(out_img)
        }
        DynamicImage::ImageRgb8(img) => {
            let out_img = image::RgbImage::from_par_fn(c as u32, r as u32, |x, y| {
                let idx = y as usize * c + x as usize;
                let (x_cor, y_cor) = unsafe { (map0.get_unchecked(idx), map1.get_unchecked(idx)) };
                if x_cor.is_nan() || y_cor.is_nan() {
                    return image::Rgb([0, 0, 0]);
                }
                let x_cor = x_cor.round() as u32;
                let y_cor = y_cor.round() as u32;
                if x_cor >= img.width() || y_cor >= img.height() {
                    image::Rgb([0, 0, 0])
                } else {
                    img.get_pixel(x_cor, y_cor).to_owned()
                }
            });
            DynamicImage::ImageRgb8(out_img)
        }
        _ => {
            panic!("remap only supports gray8 and rgb8");
        }
    };
    out
}

pub trait CameraModel {
    fn project(&self, p3d: &[na::Point3<f64>]) -> Vec<Option<(f32, f32)>>;
    fn unproject(&self, p2d: &[na::Point2<f64>]) -> Vec<Option<(f32, f32)>>;
    fn init_undistort_map(
        &self,
        projection_mat: &na::Matrix3<f64>,
        new_h_w: (u32, u32),
    ) -> (na::DMatrix<f32>, na::DMatrix<f32>) {
        if projection_mat.shape() != (3, 3) {
            panic!("projection matrix has the wrong shape");
        }
        let fx = projection_mat[(0, 0)];
        let fy = projection_mat[(1, 1)];
        let cx = projection_mat[(0, 2)];
        let cy = projection_mat[(1, 2)];
        let p3ds: Vec<na::Point3<f64>> = (0..new_h_w.0)
            .into_par_iter()
            .flat_map(|y| {
                (0..new_h_w.1)
                    .into_par_iter()
                    .map(|x| na::Point3::new((x as f64 - cx) / fx, (y as f64 - cy) / fy, 1.0))
                    .collect::<Vec<na::Point3<f64>>>()
            })
            .collect();
        let p2ds = self.project(&p3ds);
        let (xvec, yvec): (Vec<f32>, Vec<f32>) = p2ds
            .par_iter()
            .map(|xy| {
                if let Some((x, y)) = xy {
                    (*x, *y)
                } else {
                    (f32::NAN, f32::NAN)
                }
            })
            .unzip();
        let xmap = na::DMatrix::from_vec(new_h_w.0 as usize, new_h_w.1 as usize, xvec);
        let ymap = na::DMatrix::from_vec(new_h_w.0 as usize, new_h_w.1 as usize, yvec);
        (xmap, ymap)
    }
}
