use camera_intrinsic_calibration::detected_points::{FeaturePoint, FrameFeature};
use camera_intrinsic_calibration::util::{convert_model, find_best_two_frames_idx};
use camera_intrinsic_model::{EUCM, GenericModel, UCM};
use glam::Vec2;
use nalgebra as na;
use std::collections::HashMap;

#[test]
fn test_find_best_two_frames() {
    // Mock features
    // Frame 0: 10 points
    // Frame 1: 100 points
    // Frame 2: 50 points

    let make_frame = |n: usize| -> FrameFeature {
        let mut features = HashMap::new();
        for i in 0..n {
            features.insert(i as u32, FeaturePoint { p2d: Vec2::ZERO, p3d: glam::Vec3::ZERO });
        }
        FrameFeature { time_ns: 0, img_w_h: (100, 100), features }
    };

    let frames = vec![Some(make_frame(10)), Some(make_frame(100)), Some(make_frame(50)), None];

    let (idx1, idx2) = find_best_two_frames_idx(&frames, false);

    // Logic selects max detection.
    // Code:
    // 1. Find indices with max detections. Frame 1 has 100.
    // If we only have single max, logic might be tricky.
    // Let's see code.
    // It sorts logic based on distance to center / area.
    // But first it filters by "max_detection".
    // Wait, the code says:
    // if len > max: max = len; idxs = [i]
    // if len == max: idxs.push(i)

    // So if Frame 1 has 100, and no other has 100, then idxs = [1].
    // Then it returns (idxs[0], idxs[0]) ??
    // (v1.last, v0.last).

    // Ideally we should have multiple frames with same max if we want distinct frames?
    // Or maybe the loop logic is designed to keep top N?
    // Actually the code:
    // iter enumerate: if f.len > max ...
    // It only keeps the *absolute best* count.

    // So if Frame 1 is strictly best, it will return (1, 1).
    // Let's verify this behavior.
    assert_eq!(idx1, 1);
    assert_eq!(idx2, 1);

    // If we have two frames with 100
    let frames2 = vec![Some(make_frame(100)), Some(make_frame(100))];
    let (i1, i2) = find_best_two_frames_idx(&frames2, false);
    // Should be some combination of 0 and 1.
    assert!(i1 == 0 || i1 == 1);
    assert!(i2 == 0 || i2 == 1);
}

#[test]
fn test_convert_model() {
    // Source UCM
    let params = na::dvector![500.0, 500.0, 320.0, 240.0, 0.5];
    let ucm = GenericModel::UCM(UCM::new(&params, 640, 480));

    // Target EUCM
    let mut eucm = GenericModel::EUCM(EUCM::new(
        &na::dvector![400.0, 400.0, 320.0, 240.0, 0.0, 1.0],
        640,
        480,
    ));

    // Convert
    convert_model(&ucm, &mut eucm, 0);

    // Check if parameters changed/optimized
    // UCM to EUCM: alpha maps to alpha, beta fixed to 1.0 if not optimized?
    // Logic for UCM -> EUCM:
    // if target is EUCM, it just copies params and sets beta=1.0.
    /*
    if let GenericModel::UCM(m0) = source_model {
        if let GenericModel::EUCM(_) = target_model {
             ... target_model.set_params ...
             return;
        }
    }
    */
    // So it should be exact copy plus beta=1.

    let p = eucm.params();
    assert!((p[0] - 500.0).abs() < 1e-6); // fx
    assert!((p[4] - 0.5).abs() < 1e-6); // alpha
    assert!((p[5] - 1.0).abs() < 1e-6); // beta
}
