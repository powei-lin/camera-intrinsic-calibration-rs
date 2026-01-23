pub mod board;
pub mod data_loader;
pub mod detected_points;
pub mod io;
pub mod optimization;
pub mod types;
pub mod util;
pub mod visualization;

/// Macros for easier configuration
#[macro_export]
macro_rules! calib_config {
    ($model:expr, $board:expr) => {
        ($model, $board)
    };
    ($model:expr, $board:expr, solver: $solver:expr) => {
        ($model, $board, $solver)
    };
}

/// Feature-gated modules
#[cfg(feature = "parallel")]
pub mod parallel_utils {
    use rayon::prelude::*;
    use std::collections::HashMap;

    /// Parallel processing of frame features
    pub fn process_frames_parallel<T, F>(
        frames: &[Option<T>],
        processor: F,
    ) -> Vec<Option<T::Output>>
    where
        T: Send + Sync,
        F: Fn(&T) -> T::Output + Send + Sync,
        T::Output: Send,
    {
        frames
            .par_iter()
            .map(|frame| frame.as_ref().map(&processor))
            .collect()
    }
}

/// Extended visualization utilities
#[cfg(feature = "visualization")]
pub mod vis_ext {
    use rerun::{RecordingStream, RecordingStreamBuilder};

    /// Create a recording stream with custom settings
    pub fn create_recording_stream(name: &str) -> Result<RecordingStream, Box<dyn std::error::Error>> {
        Ok(RecordingStreamBuilder::new(name).connect()?)
    }
}
