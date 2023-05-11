import time
import os
from utils import *


# Extract frames from the video
video_path = 'data/stable/square_stable.mp4'
interval = 72
# width = 608
# height = 1080
width = 1920
height = 1080
print('Extracting frames from video...')
s_time = time.time()
frames = extract_frames(video_path, interval, width, height)
print(f'Done! Extracted {len(frames)} frames. Take {time.time() - s_time:.2f}s.\n')

# Extract SIFT features for the frames
print('Extracting SIFT features...')
s_time = time.time()
keypoints, descriptors = extract_sift_features(frames)
print(f'Done! Extracted keypoints and descriptors for {len(keypoints)} frames. Take {time.time() - s_time:.2f}s.\n')

# Calculate the homographies between the reference frame and each frame
print('Calculating homographies...')
ref_frame_idx = -1
if ref_frame_idx < 0:
    ref_frame_idx = len(keypoints) + ref_frame_idx
s_time = time.time()
homographies = calc_homographies(keypoints, descriptors, ref_frame_idx)
print(f'Done! Calculated {len(homographies)} homographies. Take {time.time() - s_time:.2f}s.\n')

# Warp all frames to the coordinate system of the first frame
print('Warping frames...')
s_time = time.time()
warped_frames = warp_frames(frames, homographies)
print(f'Done! Warped all frames to the coordinate system of frame {ref_frame_idx}. Take {time.time() - s_time:.2f}s.\n')

# Generate panorama from the warped frames
print('Generating panorama...')
num_levels = 3
s_time = time.time()
panorama = PanoramaGenerator.gen_panorama(warped_frames, num_levels)
print(f'Done! Generated panorama. Take {time.time() - s_time:.2f}s.\n')

# Crop the black borders of the panorama
print('Cropping the panorama...')
s_time = time.time()
panorama = crop_black_border(panorama)
print(f'Done! Cropped the panorama. Take {time.time() - s_time:.2f}s.\n')

# Save the panorama
print('Saving the panorama...')
result_path = os.path.join('result', video_path.split('/')[-1].split('.')[0] + '.jpg')
cv2.imwrite(result_path, panorama)
print(f'Done! Saved the panorama in {result_path}.\n')