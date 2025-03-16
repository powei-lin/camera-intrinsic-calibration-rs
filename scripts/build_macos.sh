cargo build -r
cp ./target/release/ccrs ./ccrs
tar czvf ccrs-aarch64-apple-darwin.tar.gz ./ccrs
shasum -a 256 ccrs-aarch64-apple-darwin.tar.gz > ccrs-aarch64-apple-darwin.tar.gz.sha256
rm ./ccrs
maturin build -r