use image::DynamicImage;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::io::Cursor;

use crate::detected_points::FrameFeature;

pub fn log_image_as_compressed(
    recording: &RecordingStream,
    topic: &str,
    img: &DynamicImage,
    format: image::ImageFormat,
) {
    let mut bytes: Vec<u8> = Vec::new();

    img.to_luma8()
        .write_to(&mut Cursor::new(&mut bytes), format)
        .unwrap();

    recording
        .log(
            format!("{}/image", topic),
            &rerun::Image::from_file_contents(bytes, None).unwrap(),
        )
        .unwrap();
}

pub fn id_to_color(id: usize) -> (u8, u8, u8, u8) {
    let mut rng = ChaCha8Rng::seed_from_u64(id as u64);
    let color_num = rng.gen_range(0..2u32.pow(24));
    (
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
        255,
    )
}

/// rerun use top left corner as (0, 0)
pub fn rerun_shift(p2ds: &[(f32, f32)]) -> Vec<(f32, f32)> {
    p2ds.iter().map(|(x, y)| (*x + 0.5, *y + 0.5)).collect()
}

pub fn log_frames(recording: &RecordingStream, detected_feature_frames: &[FrameFeature]) {
    for f in detected_feature_frames {
        let (pts, colors_labels): (Vec<_>, Vec<_>) = f
            .features
            .iter()
            .map(|(id, p)| {
                let color = id_to_color(*id as usize);
                (
                    (p.p2d.x, p.p2d.y),
                    (color, format!("{:?}", p.p3d).to_string()),
                )
            })
            .unzip();
        let (colors, labels): (Vec<_>, Vec<_>) = colors_labels.iter().cloned().unzip();
        let pts = rerun_shift(&pts);

        let topic = "/cam0";
        recording.set_time_nanos("stable", f.time_ns);
        recording
            .log(
                format!("{}/pts", topic),
                &rerun::Points2D::new(pts)
                    .with_colors(colors)
                    .with_labels(labels)
                    .with_radii([rerun::Radius::new_ui_points(5.0)]),
            )
            .unwrap();
    }
}
