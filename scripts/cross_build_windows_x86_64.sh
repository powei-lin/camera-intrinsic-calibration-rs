cargo zigbuild --release --target=x86_64-pc-windows-gnu
cp ./target/x86_64-pc-windows-gnu/release/ccrs.exe ./ccrs.exe
zip ccrs-x86_64-pc-windows-gnu.zip ./ccrs.exe
shasum -a 256 ccrs-x86_64-pc-windows-gnu.zip > ccrs-x86_64-pc-windows-gnu.zip.sha256
rm ./ccrs.exe
maturin build -r --target x86_64-pc-windows-gnu --zig