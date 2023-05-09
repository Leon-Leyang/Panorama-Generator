from utils import *


# Extract frames from the video
video_path = 'data/stable/garden_stable.mp4'
interval = 24
# width = 608
# height = 1080
width = 1920
height = 1080
frames = extract_frames(video_path, interval, width, height)

# Extract SIFT features for the frames
keypoints, descriptors = extract_sift_features(frames)

# Calculate the homographies between adjacent frames
adjacent_homographies = calc_adjacent_homographies(keypoints, descriptors)

# Cumulate the homographies
cumulative_homographies = cumulate_homographies(adjacent_homographies)

# Warp all frames to the coordinate system of the first frame
warped_frames = warp_frames(frames, cumulative_homographies)

display_frames(warped_frames)
