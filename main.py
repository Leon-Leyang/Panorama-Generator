import time
import os
from utils import *


# # Extract frames from the video
# video_path = 'data/stable/garden_stable.mp4'
# interval = 24
# # width = 608
# # height = 1080
# width = 1920
# height = 1080
# print('Extracting frames from video...')
# s_time = time.time()
# frames = extract_frames(video_path, interval, width, height)
# print(f'Done! Extracted {len(frames)} frames. Take {time.time() - s_time:.2f}s.\n')
#
# # Extract SIFT features for the frames
# print('Extracting SIFT features...')
# s_time = time.time()
# keypoints, descriptors = extract_sift_features(frames)
# print(f'Done! Extracted keypoints and descriptors for {len(keypoints)} frames. Take {time.time() - s_time:.2f}s.\n')
#
# # Calculate the homographies between adjacent frames
# print('Calculating adjacent homographies...')
# s_time = time.time()
# adjacent_homographies = calc_adjacent_homographies(keypoints, descriptors)
# print(f'Done! Calculated {len(adjacent_homographies)} adjacent homographies. Take {time.time() - s_time:.2f}s.\n')
#
# # Cumulate the homographies
# print('Cumulating homographies...')
# s_time = time.time()
# cumulative_homographies = cumulate_homographies(adjacent_homographies)
# print(f'Done! Got {len(cumulative_homographies)} cumulated homographies. Take {time.time() - s_time:.2f}s.\n')
#
# # Warp all frames to the coordinate system of the first frame
# print('Warping frames...')
# s_time = time.time()
# warped_frames = warp_frames(frames, cumulative_homographies)
# print(f'Done! Warped all frames to the coordinate system of the first frame. Take {time.time() - s_time:.2f}s.\n')

import pickle
with open('./warped_frames.pkl', 'rb') as f:
    warped_frames = pickle.load(f)
video_path = 'data/stable/garden_stable.mp4'

# Generate panorama from the warped frames
print('Generating panorama...')
num_levels = 3
result_path = os.path.join('result', video_path.split('/')[-1].split('.')[0] + '.jpg')
s_time = time.time()
panorama = PanoramaGenerator.gen_panorama(warped_frames, num_levels, result_path.replace('garden_stable', 'garden_stable_3'))
print(f'Done! Saved the panorama in {result_path}. Take {time.time() - s_time:.2f}s.\n')
