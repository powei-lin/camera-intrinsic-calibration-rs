use glam;
use std::collections::HashMap;

/// Represents a detected feature point with 2D image coordinates and 3D world coordinates.
#[derive(Debug, Clone, Copy)]
pub struct FeaturePoint {
    pub p2d: glam::Vec2,
    pub p3d: glam::Vec3,
}

/// Represents a set of features detected in a single frame.
#[derive(Debug, Clone)]
pub struct FrameFeature {
    pub time_ns: i64,
    pub img_w_h: (u32, u32),
    pub features: HashMap<u32, FeaturePoint>,
}
