use glam;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct BoardConfig {
    tag_size_meter: f32,
    tag_spacing: f32,
    tag_rows: usize,
    tag_cols: usize,
    first_id: u32,
}

impl Default for BoardConfig {
    fn default() -> Self {
        Self {
            tag_size_meter: 0.088,
            tag_spacing: 0.3,
            tag_rows: 6,
            tag_cols: 6,
            first_id: 0,
        }
    }
}

pub struct Board {
    pub id_to_3d: HashMap<u32, glam::Vec3>,
}

impl Board {
    pub fn from_config(board_config: &BoardConfig) -> Board {
        Self::init_aprilgrid(
            board_config.tag_size_meter,
            board_config.tag_spacing,
            board_config.tag_rows,
            board_config.tag_cols,
            board_config.first_id,
        )
    }
    pub fn init_aprilgrid(
        tag_size_meter: f32,
        tag_spacing: f32,
        tag_rows: usize,
        tag_cols: usize,
        first_id: u32,
    ) -> Board {
        let mut id_to_3d = HashMap::new();
        let mut count_id = first_id * 4;
        for r in 0..tag_rows {
            for c in 0..tag_cols {
                let start_x = (c as f32) * tag_size_meter * (1.0 + tag_spacing);
                let start_y = -1.0 * (r as f32) * tag_size_meter * (1.0 + tag_spacing);
                id_to_3d.insert(
                    count_id,
                    glam::Vec3 {
                        x: start_x,
                        y: start_y,
                        z: 0.0,
                    },
                );
                id_to_3d.insert(
                    count_id + 1,
                    glam::Vec3 {
                        x: start_x + tag_size_meter,
                        y: start_y,
                        z: 0.0,
                    },
                );
                id_to_3d.insert(
                    count_id + 2,
                    glam::Vec3 {
                        x: start_x + tag_size_meter,
                        y: start_y - tag_size_meter,
                        z: 0.0,
                    },
                );
                id_to_3d.insert(
                    count_id + 3,
                    glam::Vec3 {
                        x: start_x,
                        y: start_y - tag_size_meter,
                        z: 0.0,
                    },
                );
                count_id += 4;
            }
        }
        Board { id_to_3d }
    }
}

pub fn create_default_6x6_board() -> Board {
    Board::init_aprilgrid(0.088, 0.3, 6, 6, 0)
}
