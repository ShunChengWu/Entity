# 3RScan
ffmpeg -framerate 8 -i data/output/frame-%06d.jpg -c:v libx264 -pix_fmt yuv420p data/out.mp4
# ScanNet
ffmpeg -framerate 8 -start_number 0  -i 'data/scene0000_00_all/%d.jpg' -c:v libx264 -pix_fmt yuv420p data/scene0000_00_all.mp4
