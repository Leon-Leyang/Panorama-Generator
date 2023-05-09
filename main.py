from utils import *


# Extract frames from the video
video_path = 'data/stable/lake_stable.mp4'
interval = 24
frames = extract_frames(video_path, interval)

# Extract SIFT features for the frames
keypoints, descriptors = extract_sift_features(frames)

# Display the frames with keypoints
display_frames_with_keypoints(frames, keypoints)