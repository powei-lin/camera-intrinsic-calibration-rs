use std::io::Cursor;

use image::DynamicImage;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;

use crate::detected_points::FrameFeature;

/// Logs an image to Rerun using PNG compression.
pub fn log_image(recording: &RecordingStream, topic: &str, img: &DynamicImage) {
    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("not able to write png");
    let rr_image = rerun::EncodedImage::from_file_contents(bytes);
    recording.log(topic.to_string(), &rr_image).unwrap();
}

/// Generates a consistent random color for a given ID.
pub fn id_to_color(id: usize) -> (u8, u8, u8, u8) {
    let mut rng = ChaCha8Rng::seed_from_u64(id as u64);
    let color_num = rng.random_range(0..2u32.pow(24));
    (
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
        255,
    )
}

/// Shifts 2D points by 0.5 for Rerun visualization.
///
/// Rerun considers pixel center coordinates differently, so a shift is often needed for alignment.
pub fn rerun_shift(p2ds: &[(f32, f32)]) -> Vec<(f32, f32)> {
    p2ds.iter().map(|(x, y)| (*x + 0.5, *y + 0.5)).collect()
}

/// Logs detected feature points to Rerun.
///
/// Visualizes points with colors based on their IDs and 3D coordinate labels.
pub fn log_feature_frames(recording: &RecordingStream, topic: &str, detected_feature_frames: &[Option<FrameFeature>]) {
    for f in detected_feature_frames {
        let ((pts, colors_labels), time_ns): ((Vec<_>, Vec<_>), i64) = if let Some(f) = f {
            (
                f.features
                    .iter()
                    .map(|(id, p)| {
                        let color = id_to_color(*id as usize);
                        ((p.p2d.x, p.p2d.y), (color, format!("{:?}", p.p3d).to_string()))
                    })
                    .unzip(),
                f.time_ns,
            )
        } else {
            continue;
        };
        let (colors, labels): (Vec<_>, Vec<_>) = colors_labels.iter().cloned().unzip();
        let pts = rerun_shift(&pts);

        recording.set_time("stable", rerun::TimeCell::from_timestamp_nanos_since_epoch(time_ns));
        recording
            .log(
                format!("{}/pts", topic),
                &rerun::Points2D::new(pts)
                    .with_colors(colors)
                    .with_labels(labels)
                    .with_radii([rerun::Radius::new_ui_points(2.0)]),
            )
            .unwrap();
    }
}
