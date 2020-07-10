import cv2


class Video_Crop():

    def video_read(self):
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

            if cv2.waitKey(1)==ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def videoshot(self,frame):

        # if cv2.waitKey(0)==ord("w")
        cv2.fr


if __name__ == "__main__":
    video = Video_Crop()
    video.video_read()
