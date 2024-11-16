use glob::glob;
pub fn load_euroc(root_folder: &str) {
    let img_paths = glob(format!("{}/mav0/cam0/data/*.png", root_folder).as_str()).expect("failed");
}
