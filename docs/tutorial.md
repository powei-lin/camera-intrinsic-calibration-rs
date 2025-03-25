# Before calibration
1. What kind of camera are you calibrating?
    * Global shutter camera -> you can record a video and convert the video to images, or take photos.
    * Rolling shutter camera -> make sure you only take photo while camera is static.
2. Print a custom board and get the board config json. See [generate chart pdf section.](https://github.com/powei-lin/aprilgrid-rs?tab=readme-ov-file#generate-chart-pdf). If you don't care camera to camera extrinsic, you can print the board in any size. **MAKE SURE THE BOARD IS FLAT!**
3. Make sure the folder structure follows [supported dataset format](https://github.com/powei-lin/camera-intrinsic-calibration-rs?tab=readme-ov-file#dataset-format).

# Run calibration
1. What camera model should I use?
    * Wide FoV camera -> `kb4` or `eucm`
    * Other -> `opencv5`
    * If you know of other models, you probably don't need my recommendation.
    * My favorite -> EUCMT
2. Use rerun to analyze the result. You can click `rep_err` and change to visable time range from begining to see the accumulated reprojection errors.

    For example using `kb4` model with only two distortion parameters. This shows the edge has higher errors.
    <img src="../data/kb2.jpg" width="800" alt="kb2">
    Use `kb4` model with four distortion parameters.
    <img src="../data/kb4.jpg" width="800" alt="kb4">

# Calibration tricks
* If two distortion parameter is enough, don't use more. Using more distortion can cause overfitting.
* `--one-focal` is highly recommended.
* Make sure your dataset covers multiple angles and multiple distances like the example dataset.