use camera_intrinsic_calibration::board::{Board, BoardConfig};

#[test]
fn test_board_init() {
    let board = Board::from_config(&BoardConfig::default());
    // Default 6x6 board, 4 points per tag = 36 * 4 = 144 points
    assert_eq!(board.id_to_3d.len(), 6 * 6 * 4);
    
    // Check first tag (id 0)
    // Points are (0,0), (size, 0), (size, -size), (0, -size) relative to start
    // Row 0, Col 0 start at (0,0)
    
    // Tag size 0.088
    let s = 0.088;
    
    let p0 = board.id_to_3d.get(&0).unwrap();
    let p1 = board.id_to_3d.get(&1).unwrap();
    let p2 = board.id_to_3d.get(&2).unwrap();
    let p3 = board.id_to_3d.get(&3).unwrap();
    
    assert!((p0.x - 0.0).abs() < 1e-6);
    assert!((p0.y - 0.0).abs() < 1e-6);
    
    assert!((p1.x - s).abs() < 1e-6);
    assert!((p1.y - 0.0).abs() < 1e-6);

    assert!((p2.x - s).abs() < 1e-6);
    assert!((p2.y + s).abs() < 1e-6); // Note: y is negative in init_aprilgrid: start_y - tag_size => 0 - s = -s. Wait.
    // Code says: y: start_y - tag_size_meter. 
    // start_y is -(r) * ...
    // So for r=0, start_y=0.
    // p3 y is 0 - s = -s. 
    // Wait, let's re-read code for P2.
    // p2: y: start_y - tag_size_meter. = -s.
    
    assert!((p2.y - (-s)).abs() < 1e-6);

    assert!((p3.x - 0.0).abs() < 1e-6);
    assert!((p3.y - (-s)).abs() < 1e-6);
}
