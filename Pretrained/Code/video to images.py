import cv2


def save_video_images(video_path, save_path, frames=240):
    cap = cv2.VideoCapture(video_path)
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    count = 0
    while cap.isOpened():
        count += 1
        for i in range(frames):
            ret, frame = cap.read()
            if not ret:
                break
            pass
        print("Saving")
        cv2.imwrite(f"{save_path}\\Image{count}.png", frame)


def main():
    video_path = r"..\Dataset\Video\Raanana.webm"
    save_path = r"..\Dataset\Images_from_video"
    save_video_images(video_path, save_path)


if __name__ == '__main__':
    main()
