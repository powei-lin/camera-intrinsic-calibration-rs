use camera_intrinsic_calibration::{calib_config, parallel_utils, vis_ext};

#[test]
fn test_calib_config_macro() {
    // Test the macro compiles and expands correctly
    let config = calib_config!(42.0, "board");
    assert_eq!(config, (42.0, "board"));

    let config_with_solver = calib_config!(42.0, "board", solver: "gauss-newton");
    assert_eq!(config_with_solver, (42.0, "board", "gauss-newton"));
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_utils() {
    let data = vec![Some(1), None, Some(3)];
    let results: Vec<Option<i32>> = parallel_utils::process_frames_parallel(&data, |x| x * 2);
    assert_eq!(results, vec![Some(2), None, Some(6)]);
}

#[test]
#[cfg(feature = "visualization")]
fn test_vis_ext() {
    // Test that the function exists and can be called
    // In real test, would need a mock recording stream
    let _result = vis_ext::create_recording_stream("test");
}

#[test]
fn test_feature_flags() {
    // Test that features compile correctly
    // This is more of a compile-time test
    #[cfg(feature = "visualization")]
    println!("Visualization feature enabled");

    #[cfg(feature = "parallel")]
    println!("Parallel feature enabled");

    #[cfg(feature = "image-support")]
    println!("Image support feature enabled");
}
