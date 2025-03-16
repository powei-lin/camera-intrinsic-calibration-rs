cargo zigbuild --release --target=aarch64-unknown-linux-gnu
cp ./target/aarch64-unknown-linux-gnu/release/ccrs ./ccrs
tar czvf ccrs-aarch64-unknown-linux-gnu.tar.gz ./ccrs
shasum -a 256 ccrs-aarch64-unknown-linux-gnu.tar.gz > ccrs-aarch64-unknown-linux-gnu.tar.gz.sha256
rm ./ccrs
maturin build -r --target aarch64-unknown-linux-gnu --zig