import time
import argparse
from utils.utils import *
from utils.panorama_generator import PanoramaGenerator

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to the video')
ap.add_argument('-i', '--interval', type=int, default=72, help='interval between sampled frames')
ap.add_argument('-w', '--width', type=int, default=1920, help='width of the sampled frames')
ap.add_argument('-he', '--height', type=int, default=1080, help='height of the sampled frames')
ap.add_argument('-r', '--ref_frame_idx', type=int, default=0, help='index of the reference frame')
ap.add_argument('-l', '--num_levels', type=int, default=3, help='number of levels in the pyramid of multi-band blending'
                )
ap.add_argument('-c', '--crop', action='store_true', help='whether to crop the black borders')
ap.add_argument('-o', '--output', default='output.jpg', help='path to the output panorama')

# Parse the arguments
arg = ap.parse_args()

# Extract frames from the video
print('Extracting frames from video...')
s_time = time.time()
frames = sample_frames(arg.video, arg.interval, arg.width, arg.height)
print(f'Done! Sampled {len(frames)} frames. Take {time.time() - s_time:.2f}s.\n')

# Extract SIFT features for the frames
print('Extracting SIFT features...')
s_time = time.time()
keypoints, descriptors = extract_sift_features(frames)
print(f'Done! Extracted keypoints and descriptors for {len(keypoints)} frames. Take {time.time() - s_time:.2f}s.\n')

# Calculate the homographies between the reference frame and each frame
print('Calculating homographies...')
if arg.ref_frame_idx < 0:
    arg.ref_frame_idx = len(keypoints) + arg.ref_frame_idx
s_time = time.time()
homographies = calc_homographies(keypoints, descriptors, arg.ref_frame_idx)
print(f'Done! Calculated {len(homographies)} homographies. Take {time.time() - s_time:.2f}s.\n')

# Warp all frames to the coordinate system of the first frame
print('Warping frames...')
s_time = time.time()
warped_frames = warp_frames(frames, homographies)
print(f'Done! Warped all frames to the coordinate system of frame {arg.ref_frame_idx}. '
      f'Take {time.time() - s_time:.2f}s.\n')

# Generate panorama from the warped frames
print('Generating panorama...')
s_time = time.time()
panorama = PanoramaGenerator.gen_panorama(warped_frames, arg.num_levels)
print(f'Done! Generated panorama. Take {time.time() - s_time:.2f}s.\n')

# Crop the black borders of the panorama
print('Cropping the panorama...')
s_time = time.time()
panorama = crop_black_border(panorama)
print(f'Done! Cropped the panorama. Take {time.time() - s_time:.2f}s.\n')

# Save the panorama
print('Saving the panorama...')
cv2.imwrite(arg.output, panorama)
print(f'Done! Saved the panorama in {arg.output}.\n')
