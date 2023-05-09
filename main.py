from utils import *


# Extract frames from the video
video_path = 'data/stable/lib_stable.mp4'
interval = 24
frames = extract_frames(video_path, interval)

# Convert the frames to grayscale using cv
gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]


