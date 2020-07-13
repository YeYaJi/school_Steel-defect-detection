import cv2
import tools
import numpy as np
import tools
import detect

yolo_detect = detect.Detect("./yolo/coco.names", "./yolo/yolov3-tiny.cfg", "./yolo/yolov3-tiny.weights")


class Video_Crop():
    def __init__(self):
        self.pictures = np.zeros(0)

    def video_read(self):
        i = 0
        cap = cv2.VideoCapture(0)
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
            wait = cv2.waitKey(1)  # 设置图像显示时间为1毫秒
            if wait == ord("c"):
                i += 1

                detect_img = yolo_detect.run_one(frame)
                picture_sum = np.hstack([frame, detect_img])
                tools.imshow("num  [{}]  picture".format(i), picture_sum)



            elif wait == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video = Video_Crop()
    video.video_read()

    # picture_sum = np.hstack(video.pictures)
    # print(picture_sum)
    # tools.imshow("name", picture_sum)
