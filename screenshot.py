import cv2
import tools
import numpy as np





class Video_Crop():
    def __init__(self):
        self.pictures = []

    def video_read(self):
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("相机故障")
            exit()
        while 1:
            ret, frame = cap.read()
            if not ret:
                print("没有读到视频流")
                break

            cv2.namedWindow("video", cv2.WINDOW_NORMAL)
            cv2.imshow("video", frame)
            wait = cv2.waitKey(1)  # 设置图像显示时间为25毫秒
            if wait == ord("c"):
                self.video_shot(frame)
            elif wait == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def video_shot(self, frame):
        self.pictures.append (frame)


if __name__ == "__main__":
    video = Video_Crop()
    video.video_read()
    picture_sum=np.hstack(video.pictures)
    print(picture_sum)
    tools.imshow("name", picture_sum)
