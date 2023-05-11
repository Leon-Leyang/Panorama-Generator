import cv2
import numpy as np


# Function to extract frames with a given interval from a video in the given path
def extract_frames(video_path, interval, width=1920, height=1080):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_height, frame_width = frame.shape[:2]
            x_start = int((frame_width - width) / 2)
            y_start = int((frame_height - height) / 2)
            frame = frame[y_start:y_start + height, x_start:x_start + width]
            frames.append(frame)
        frame_count += 1
    cap.release()

    return frames


# Function to extract SIFT features for a list of frames
def extract_sift_features(frames):
    # Convert the frames to grayscale
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Extract SIFT features iteratively
    keypoints = []
    descriptors = []
    for frame in frames:
        kp, des = sift.detectAndCompute(frame, None)
        keypoints.append(kp)
        descriptors.append(des)

    return keypoints, descriptors


# Function to calculate the homographies between consecutive frames
def calc_adjacent_homographies(keypoints, descriptors):
    homographies = []

    # Create a BFMatcher object
    bfm = cv2.BFMatcher()

    for i in range(len(keypoints) - 1):
        # Find the matches between the descriptors of the current frame and the next frame
        matches = bfm.knnMatch(descriptors[i], descriptors[i + 1], k=2)

        # Apply Lowe's ratio test to select good matches
        # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        good_matches = []
        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                good_matches.append(m1)

        # Extract the coordinates of the matched keypoints
        src_pts = [keypoints[i][m.queryIdx].pt for m in good_matches]
        dst_pts = [keypoints[i + 1][m.trainIdx].pt for m in good_matches]

        # Calculate the homography between the current frame and the next frame
        h, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC)
        homographies.append(h)

    return homographies


# Function to cumulate the homographies
def cumulate_homographies(homographies):
    # Initialize the cumulative homographies
    cumulative_homographies = [np.identity(3)]

    # Cumulate the homographies iteratively
    for h in homographies:
        cumulative_homographies.append(cumulative_homographies[-1] @ h)

    return cumulative_homographies


# Function to warp all frames to the coordinate system of the first frame
def warp_frames(frames, cumulative_homographies):
    warped_frames = []

    # Warp all frames iteratively
    for i in range(len(frames)):
        # Get the inverse of the cumulative homography
        h_inv = np.linalg.inv(cumulative_homographies[i])

        # Warp the current frame
        size = tuple(map(int, (frames[i].shape[1] * 3, frames[i].shape[0] * 1.5)))
        warped_frame = cv2.warpPerspective(frames[i], h_inv, size)
        warped_frames.append(warped_frame)

    return warped_frames


# Function to display a list of frames
def display_frames(frames, scale=1):
    for frame in frames:
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        cv2.imshow('frame', frame)
        cv2.waitKey(0)


# Function to display a list of frames with keypoints
def display_frames_with_keypoints(frames, keypoints):
    for i in range(len(frames)):
        frame = frames[i]
        kp = keypoints[i]
        frame = cv2.drawKeypoints(frame, kp, None)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)


# Class to generate a panorama from warped frames
class PanoramaGenerator:
    @staticmethod
    def gen_panorama(warped_frames, num_levels, result_path):
        panorama = warped_frames[0]
        for i in range(1, len(warped_frames)):
            panorama = PanoramaGenerator.__blend_frames_multiband(panorama, warped_frames[i], num_levels)

        cv2.imwrite(result_path, panorama)

        return panorama

    @staticmethod
    # Function to generate a Gaussian pyramid
    def __gen_gaussian_pyramid(frame, num_levels):
        pyramid = [frame]
        for _ in range(num_levels - 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)

        return pyramid

    @staticmethod
    # Function to generate a Laplacian pyramid
    def __gen_laplacian_pyramid(frame, num_levels):
        frame = frame.astype(np.float32)
        gaussian_pyramid = PanoramaGenerator.__gen_gaussian_pyramid(frame, num_levels)
        laplacian_pyramid = []
        for i in range(num_levels - 1):
            expanded_frame = cv2.pyrUp(gaussian_pyramid[i + 1])
            # Resize the expanded frame to the size of the current level
            expanded_frame = cv2.resize(expanded_frame, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            laplacian_pyramid.append(gaussian_pyramid[i] - expanded_frame)

        laplacian_pyramid.append(gaussian_pyramid[num_levels - 1])

        return laplacian_pyramid

    @staticmethod
    # Function to blend two laplacian pyramids
    def __blend_laplacian_pyramids(pyramid_1, pyramid_2, pyramid_mask):
        blended_pyramid = []
        for i in range(len(pyramid_1)):
            blended_pyramid.append(pyramid_1[i] * (1 - pyramid_mask[i]) + pyramid_2[i] * pyramid_mask[i])

        return blended_pyramid

    @staticmethod
    # Function to reconstruct a frame from a laplacian pyramid
    def __reconstruct_from_laplacian_pyramid(laplacian_pyramid):
        frame = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            expanded_frame = cv2.pyrUp(frame)
            # Resize the expanded frame to the size of the current level
            expanded_frame = cv2.resize(expanded_frame, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
            frame = laplacian_pyramid[i] + expanded_frame

        return frame.astype(np.uint8)

    @staticmethod
    # Function to generate a smooth mask of the same size as the frame
    def __gen_smooth_mask(frame1, frame2):
        # Generate a mask and initialize it to 0
        # Then set the non-overlapping area of frame1 to 0 and the non-overlapping area of frame2 to 1
        mask = np.zeros_like(frame1, dtype=np.float32)
        mask[(frame1 > 0) & (frame2 == 0)] = 0
        mask[(frame1 == 0) & (frame2 > 0)] = 1

        # Apply a Gaussian blur to the mask
        ksize = 55
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Normalize the mask to be in the range [0, 1]
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

        return mask

    # Function to generate a feathered mask of the same size as the frame
    @staticmethod
    def __gen_feathered_mask(frame1, frame2):
        # Convert frames to grayscale
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Create a binary mask with non-overlapping areas set to 1 and 0
        mask = np.zeros_like(frame1, dtype=np.float32)
        mask[(frame1_gray > 0) & (frame2_gray == 0)] = 0
        mask[(frame1_gray == 0) & (frame2_gray > 0)] = 1

        # Create a linear gradient in the overlapping area
        overlap_mask = np.logical_and(frame1_gray > 0, frame2_gray > 0)
        rows, cols = np.where(overlap_mask)

        if cols != []:
            min_col, max_col = np.min(cols), np.max(cols)

            for col in range(min_col, max_col + 1):
                mask[overlap_mask[:, col], col, :] = (col - min_col) / (max_col - min_col)

        return mask

    @staticmethod
    # Function to get the mask of the valid region of a frame
    def __get_valid_mask(frame):
        vld_mask = np.any(frame != 0, axis=-1)

        # Stack the mask three times to get a 3-channel mask
        vld_mask = np.stack((vld_mask, vld_mask, vld_mask), axis=-1).astype(np.uint8)
        return vld_mask

    @staticmethod
    # Function to blend two frames using multiband blending
    def __blend_frames_multiband(frame_1, frame_2, num_levels):
        mask = PanoramaGenerator.__gen_feathered_mask(frame_1, frame_2)

        # Generate the Gaussian and Laplacian pyramids for the two frames
        laplacian_pyramid_1 = PanoramaGenerator.__gen_laplacian_pyramid(frame_1, num_levels)
        laplacian_pyramid_2 = PanoramaGenerator.__gen_laplacian_pyramid(frame_2, num_levels)
        mask_pyramid = PanoramaGenerator.__gen_gaussian_pyramid(mask, num_levels)

        # Blend the two laplacian pyramids
        blended_laplacian_pyramid = PanoramaGenerator.__blend_laplacian_pyramids(laplacian_pyramid_1,
                                                                                 laplacian_pyramid_2,
                                                                                 mask_pyramid)

        # Reconstruct the blended frame from the blended laplacian pyramid
        blended_frame = PanoramaGenerator.__reconstruct_from_laplacian_pyramid(blended_laplacian_pyramid)

        return blended_frame


if __name__ == '__main__':
    video_path = 'data/stable/lib_stable.mp4'
    interval = 24

    frames = extract_frames(video_path, interval)
    print(len(frames))

    # Display the frames
    display_frames(frames)
