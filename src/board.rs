use glam;
use std::collections::HashMap;

pub struct Board {
    pub id_to_3d: HashMap<u32, glam::Vec3>,
}

impl Board {
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
