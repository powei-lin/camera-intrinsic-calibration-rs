use camera_intrinsic::camera_model::eucm;
use camera_intrinsic::camera_model::generic::{
    estimate_new_camera_matrix_for_undistort, init_undistort_map, remap,
};
use image::ImageReader;
use nalgebra as na;

fn main() {
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    // let img = image::DynamicImage::ImageLuma8(img.to_luma8());
    let params = na::dvector![
        190.89618687183938,
        190.87022285882367,
        254.9375370481962,
        256.86414483060787,
        0.6283550447635853,
        1.0458678747533083
    ];
    let model = eucm::EUCM::new(&params, 512, 512);
    let new_w_h = 1024;
    let p = estimate_new_camera_matrix_for_undistort(&model, 1.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = init_undistort_map(&model, &p, (new_w_h, new_w_h));
    let remaped = remap(&img, &xmap, &ymap);
    remaped.save("remaped.png").unwrap()
}
