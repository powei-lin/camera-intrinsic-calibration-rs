# camera-intrinsic
[![crate](https://img.shields.io/crates/v/camera-intrinsic.svg)](https://crates.io/crates/camera-intrinsic)

A pure rust camera intrinsic library. Including
* calibration
* project / unproject points
* undistort image

## CLI Usage
```sh
# install cli
cargo install camera-intrinsic

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
# undistort and remap
cargo run -r --example remap
```

## Acknowledgements
Links:
* https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
* https://gitlab.com/VladyslavUsenko/basalt-headers
* https://github.com/itt-ustutt/num-dual
* https://github.com/sarah-quinones/faer-rs

Papers:

* Kukelova, Zuzana, et al. "Radial distortion homography." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
* Usenko, Vladyslav, Nikolaus Demmel, and Daniel Cremers. "The double sphere camera model." 2018 International Conference on 3D Vision (3DV). IEEE, 2018.

### TODO
* [ ] Multi-camera extrinsic
* [ ] Stereo Rectify
* [ ] More calibration info