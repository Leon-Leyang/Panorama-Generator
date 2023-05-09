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
def display_frames(frames):
    for frame in frames:
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


if __name__ == '__main__':
    video_path = 'data/stable/lib_stable.mp4'
    interval = 24

    frames = extract_frames(video_path, interval)
    print(len(frames))

    # Display the frames
    display_frames(frames)
