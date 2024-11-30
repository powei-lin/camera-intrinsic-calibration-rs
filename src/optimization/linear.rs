use sqpnp_simple::sqpnp_solve_glam;

use crate::detected_points::FrameFeature;

pub fn init_pose(frame_feature: &FrameFeature, lambda: f32) -> ((f64, f64, f64), (f64, f64, f64)) {
    let half_w = frame_feature.img_w_h.0 as f32 / 2.0;
    let half_h = frame_feature.img_w_h.1 as f32 / 2.0;
    let half_img_size = half_h.max(half_w);
    let cxcy = glam::Vec2::new(half_w, half_h);
    let (p2ds_z, p3ds): (Vec<_>, Vec<_>) = frame_feature
        .features
        .iter()
        .map(|f| {
            let xy = (f.1.p2d - cxcy) / half_img_size;
            let sc = 1.0 + lambda * (xy.x * xy.x + xy.y * xy.y);
            (xy / sc, f.1.p3d)
        })
        .unzip();

    sqpnp_solve_glam(&p3ds, &p2ds_z).unwrap()
}
