import cv2
import numpy as np
import tools
import detect
import screenshot
import detect
from multiprocessing import Process, Queue

video = screenshot.Video_Crop()
detect=detect.Detect("./yolo/coco.names", "./yolo/yolov3.cfg", "./yolo/yolov3.weights")


Q_picture = Queue(1)
Q_idx = Queue(1)
p1 = Process(target=video.video_read, args=(Q_picture, Q_idx))

p2 = Process(target=detect.show_detect, args=(Q_picture, Q_idx))
p1.start()
p2.start()

p1.join()
p2.terminate()