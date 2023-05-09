import cv2


# Function to extract frames with a given interval from a video in the given path
def extract_frames(video_path, interval):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames


if __name__ == '__main__':
    video_path = 'data/stable/lib_stable.mp4'
    interval = 24

    frames = extract_frames(video_path, interval)
    print(len(frames))

    # Display the frames
    for frame in frames:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
