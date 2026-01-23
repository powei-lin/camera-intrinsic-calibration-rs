use camera_intrinsic_calibration::board::{Board, BoardConfig};
use camera_intrinsic_model::{GenericModel, Pinhole, model_from_json};
use clap::{Parser, Subcommand};
use glam::Vec3;
use nalgebra as na;
use std::path::Path;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate synthetic calibration dataset
    Generate {
        /// Output directory
        #[arg(short, long)]
        output: String,

        /// Board configuration JSON
        #[arg(short, long)]
        board_config: String,

        /// Camera model JSON
        #[arg(short, long)]
        camera_model: String,

        /// Number of frames to generate
        #[arg(short, long, default_value = "20")]
        num_frames: usize,

        /// Image width
        #[arg(long, default_value = "640")]
        width: u32,

        /// Image height
        #[arg(long, default_value = "480")]
        height: u32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Commands::Generate {
            output,
            board_config,
            camera_model,
            num_frames,
            width,
            height,
        } => {
            generate_dataset(&output, &board_config, &camera_model, num_frames, width, height)?;
        }
    }

    Ok(())
}

fn generate_dataset(
    output_dir: &str,
    board_config_path: &str,
    camera_model_path: &str,
    num_frames: usize,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    // Load board config
    let board_config: BoardConfig = serde_json::from_str(&fs::read_to_string(board_config_path)?)?;
    let board = Board::from_config(&board_config);

    // Load camera model
    let camera_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(camera_model_path)?)?;
    let mut model: GenericModel<f64> = model_from_json(&camera_json)?;

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Generate random poses
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for frame_idx in 0..num_frames {
        // Random pose: translation 0.5-2m away, small rotation
        let distance = rng.gen_range(0.5..2.0);
        let angle_x = rng.gen_range(-0.1..0.1);
        let angle_y = rng.gen_range(-0.1..0.1);
        let angle_z = rng.gen_range(-0.3..0.3);

        let rvec = na::dvector![angle_x, angle_y, angle_z];
        let rot = na::Rotation3::from_scaled_axis(rvec);
        let trans = rot * na::Vector3::new(0.0, 0.0, -distance);

        // Project board points
        let mut features = Vec::new();
        for (id, pos) in &board.id_to_3d {
            // Transform to camera frame
            let p_world = na::Vector3::new(pos.x as f64, pos.y as f64, pos.z as f64);
            let p_cam = rot * p_world + trans;

            if p_cam.z > 0.0 {
                // Project
                let p2d = model.project_one(&p_cam);
                let u = p2d[0] as f32;
                let v = p2d[1] as f32;

                // Check if in image
                if u >= 0.0 && u < width as f32 && v >= 0.0 && v < height as f32 {
                    features.push((*id, u, v, pos));
                }
            }
        }

        // Save as text file (similar to dataset format)
        let mut content = format!("# Frame {}\n", frame_idx);
        content += &format!("{} {}\n", width, height);
        for (id, u, v, _pos) in features {
            content += &format!("{} {} {}\n", id, u, v);
        }

        let filename = format!("{:06}.txt", frame_idx);
        fs::write(Path::new(output_dir).join(filename), content)?;
    }

    // Save camera model
    fs::write(
        Path::new(output_dir).join("camera.json"),
        serde_json::to_string_pretty(&camera_json)?,
    )?;

    // Save board config
    fs::write(
        Path::new(output_dir).join("board.json"),
        serde_json::to_string_pretty(&board_config)?,
    )?;

    println!("Generated {} frames in {}", num_frames, output_dir);
    Ok(())
}
