# camera-intrinsic-calibration
[![crate](https://img.shields.io/crates/v/camera-intrinsic-calibration.svg)](https://crates.io/crates/camera-intrinsic-calibration)

A pure rust camera intrinsic calibration library.

## CLI Usage
```sh
# install cli
cargo install camera-intrinsic-calibration

ccrs -h

# run intrinsic calibration on TUM vi dataset
# Download and untar
wget https://vision.in.tum.de/tumvi/exported/euroc/1024_16/dataset-calib-cam1_1024_16.tar
tar xvzf dataset-calib-cam1_1024_16.tar

# [Optional] export RUST_LOG=trace
ccrs dataset-calib-cam1_1024_16 --model eucm
```
## Examples
```sh
cargo run -r --example convert_model
```

## Acknowledgements
Links:
* https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
* https://github.com/itt-ustutt/num-dual
* https://github.com/sarah-quinones/faer-rs

Papers:

* Kukelova, Zuzana, et al. "Radial distortion homography." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

### TODO
* [ ] Multi-camera extrinsic
* [ ] More calibration info