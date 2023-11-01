# FFmpeg command
ffmpeg  -framerate 5 -pattern_type glob -i "*.png" -vf "scale=1024:1174:force_original_aspect_ratio=decrease,pad=1024:1174:-1:-1:color=black" -b:v 8M -pix_fmt yuv420p out.mp4
