use glam;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Write};

#[derive(Debug, Serialize, Deserialize)]
pub struct BoardConfig {
    tag_size_meter: f32,
    tag_spacing: f32,
    tag_rows: usize,
    tag_cols: usize,
}

pub fn board_config_to_json(output_path: &str, board_config: &BoardConfig) {
    let j = serde_json::to_string_pretty(board_config).unwrap();
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(j.as_bytes()).unwrap();
}

pub fn board_config_from_json(file_path: &str) -> BoardConfig {
    let contents =
        std::fs::read_to_string(file_path).expect("Should have been able to read the file");
    serde_json::from_str(&contents).unwrap()
}

impl Default for BoardConfig {
    fn default() -> Self {
        Self {
            tag_size_meter: 0.088,
            tag_spacing: 0.3,
            tag_rows: 6,
            tag_cols: 6,
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
        )
    }
    pub fn init_aprilgrid(
        tag_size_meter: f32,
        tag_spacing: f32,
        tag_rows: usize,
        tag_cols: usize,
    ) -> Board {
        let mut id_to_3d = HashMap::new();
        let mut count_id = 0;
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
    Board::init_aprilgrid(0.088, 0.3, 6, 6)
}
