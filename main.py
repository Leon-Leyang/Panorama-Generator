from utils import *


# Extract frames from the video
video_path = 'data/stable/lib_stable.mp4'
interval = 24
width = 608
height = 1080
frames = extract_frames(video_path, interval, width, height)

# Extract SIFT features for the frames
keypoints, descriptors = extract_sift_features(frames)

# Calculate the homographies between frames
homographies = calc_homographies(keypoints, descriptors)
