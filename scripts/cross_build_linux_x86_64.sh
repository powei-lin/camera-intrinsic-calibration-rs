cargo zigbuild --release --target=x86_64-unknown-linux-gnu
cp ./target/x86_64-unknown-linux-gnu/release/ccrs ./ccrs
tar czvf ccrs-x86_64-unknown-linux-gnu.tar.gz ./ccrs
shasum -a 256 ccrs-x86_64-unknown-linux-gnu.tar.gz > ccrs-x86_64-unknown-linux-gnu.tar.gz.sha256
rm ./ccrs
maturin build -r --target x86_64-unknown-linux-gnu --zig