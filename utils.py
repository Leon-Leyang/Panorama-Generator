import cv2


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
