use camera_intrinsic::camera_model::{
    model_from_json, model_to_json, remap, GenericModel, KannalaBrandt4, OpenCVModel5, EUCM, UCM,
};
use camera_intrinsic::util::convert_model;
use image::ImageReader;
use nalgebra as na;

fn main() {
    env_logger::init();
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = image::DynamicImage::ImageLuma8(img.to_luma8());
    let source_model = model_from_json("data/eucm.json");
    // let mut target_model = GenericModel::KannalaBrandt4(KannalaBrandt4::new(
    //     &na::dvector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     source_model.width().round() as u32,
    //     source_model.height().round() as u32,
    // ));
    let mut target_model = GenericModel::UCM(UCM::new(
        &na::dvector![0.0, 0.0, 0.0, 0.0, 0.0],
        source_model.width().round() as u32,
        source_model.height().round() as u32,
    ));
    convert_model(&source_model, &mut target_model);
    model_to_json("ucm.json", &target_model);
    let new_w_h = 1024;
    let p = target_model.estimate_new_camera_matrix_for_undistort(1.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = target_model.init_undistort_map(&p, (new_w_h, new_w_h));
    let remaped = remap(&img, &xmap, &ymap);
    remaped.save("remaped_ucm.png").unwrap()
}
