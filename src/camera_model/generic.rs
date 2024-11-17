use image::{DynamicImage, GenericImage, GenericImageView};
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
                let p2d = self.project_one(&pt);
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
}

pub fn init_undistort_map(
    camera_model: Box<&dyn CameraModel<f64>>,
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
    let p3ds: Vec<na::Vector3<f64>> = (0..new_h_w.0)
        .into_par_iter()
        .flat_map(|y| {
            (0..new_h_w.1)
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
    let xmap = na::DMatrix::from_vec(new_h_w.0 as usize, new_h_w.1 as usize, xvec);
    let ymap = na::DMatrix::from_vec(new_h_w.0 as usize, new_h_w.1 as usize, yvec);
    (xmap, ymap)
}
