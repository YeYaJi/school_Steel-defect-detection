import cv2
import numpy as np
import tools
import detect

from multiprocessing import Process, Queue


class Video_Crop():
    def __init__(self):
        self.pictures = np.zeros(0)

    def video_read(self, queue_picture, queue_idx):
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
                test_img = frame
                queue_picture.put(test_img)
                print("第-{}-张截图传入".format(i))
                queue_idx.put(i)
                print("索引传入")
            elif wait == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    qnum1 = Queue()
    qnum2 = Queue()
    video = Video_Crop()
    video.video_read(qnum1, qnum2)

    # picture_sum = np.hstack(video.pictures)
    # print(picture_sum)
    # tools.imshow("name", picture_sum)
