from utils import *


# Extract frames from the video
video_path = 'data/stable/lib_stable.mp4'
interval = 24
frames = extract_frames(video_path, interval)
