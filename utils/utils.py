import cv2
import imutils
import numpy as np


# Function to extract frames with a given interval from a video in the given path
def sample_frames(video_path, interval, width=1920, height=1080):
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


# Function to calculate the homographies between the reference frame and each frame
def calc_homographies(keypoints, descriptors, ref_frame_idx):
    homographies = [np.identity(3) if i == ref_frame_idx else None for i in range(len(keypoints))]
    bfm = cv2.BFMatcher()

    for i in range(ref_frame_idx + 1, len(keypoints)):
        matches = bfm.knnMatch(descriptors[i-1], descriptors[i], k=2)
        good_matches = []
        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                good_matches.append(m1)
        src_pts = [keypoints[i-1][m.queryIdx].pt for m in good_matches]
        dst_pts = [keypoints[i][m.trainIdx].pt for m in good_matches]
        h, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC)
        homographies[i] = homographies[i-1] @ h

    for i in range(ref_frame_idx - 1, -1, -1):
        matches = bfm.knnMatch(descriptors[i], descriptors[i+1], k=2)
        good_matches = []
        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                good_matches.append(m1)
        src_pts = [keypoints[i][m.queryIdx].pt for m in good_matches]
        dst_pts = [keypoints[i+1][m.trainIdx].pt for m in good_matches]
        h, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC)
        homographies[i] = homographies[i+1] @ np.linalg.inv(h)

    return homographies

    return homographies


# Function to warp all frames to the coordinate system of the first frame
def warp_frames(frames, homographies):
    warped_frames = []

    # Warp all frames iteratively
    for i in range(len(frames)):
        # Get the inverse of the cumulative homography
        h_inv = np.linalg.inv(homographies[i])

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


# Function to crop the black borders of a panorama
def crop_black_border(image):
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    min_rect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)
        sub = cv2.subtract(min_rect, thresh)

    cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    cropped_image = image[:y + h, x:x + w]

    return cropped_image


if __name__ == '__main__':
    video_path = '../data/stable/lib_stable.mp4'
    interval = 24

    frames = sample_frames(video_path, interval)
    print(len(frames))

    # Display the frames
    display_frames(frames)
