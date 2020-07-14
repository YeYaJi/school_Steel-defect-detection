import numpy as np
import argparse
import time
import cv2
import os
import tools


class Detect():
    def __init__(self, labels_path, cfg_path, weight_path, confidence=0.4, threshhold=0.4):
        self.labels = labels_path
        self.cfg = cfg_path
        self.weight = weight_path
        self.confidence = confidence
        self.threshold = threshhold
        self.text = []

    # load the COCO class labels our YOLO model was trained on
    def load_labels(self):
        LABELS = open(self.labels).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        return COLORS, LABELS

    def load_yolo_net(self, picture):
        print("正在加载yolo网络")
        net = cv2.dnn.readNetFromDarknet(self.cfg, self.weight)
        (self.H, self.W) = picture.shape[:2]
        blob = cv2.dnn.blobFromImage(picture, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # 仅确定我们需要YOLO的*输出*层名称
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        print("YOLO检测花费时间{:.6f} seconds".format(end - start))
        return layerOutputs

    def openCV_detect(self, layerOutputs):
        boxes = []
        predict_confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                predict_confidence = scores[classID]
                if predict_confidence > self.confidence:  # confidence是最小概率
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")  # 获得(x,y,w,h)
                    # 算出边界框的左上角坐标
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    predict_confidences.append(float(predict_confidence))
                    classIDs.append(classID)
        return boxes, predict_confidences, classIDs

    def box_img(self, img, boxes, predict_confidences, classIDs):
        image = img.copy()
        idxs = cv2.dnn.NMSBoxes(boxes, predict_confidences, self.confidence, self.threshold)
        # 确保现在有至少一个框框
        if len(idxs) > 0:
            print("当前目置信度下检测到的目标个数为{}".format(len(idxs)))
            text_list = []
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                COLORS, LABELS = self.load_labels()
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], predict_confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text_list.append(text)
            return image, text_list
        else:
            print("当前置信度下没有检测到任何目标")
        return image, ["NULL"]

    def run_one(self, img):
        boxes, predict_confidences, classIDs = self.openCV_detect(self.load_yolo_net(img))
        detect_picture, text_list = self.box_img(img, boxes, predict_confidences, classIDs)
        return detect_picture, text_list

    def show_detect(self, queue_picture, queue_idx):
        while True:
            img = queue_picture.get()
            idx = queue_idx.get()
            # cv2.imwrite('./idx.jpg({})'.format(idx), img)
            print("获取图片和索引-{}-".format(idx))
            detect_img, text_list = self.run_one(img)
            picture_sum = np.hstack([img, detect_img])
            tools.imshow("num  [{}]  picture".format(idx), picture_sum)
            for i in range(len(text_list)):
                print("num  [{}]  picture result:{}".format(idx, text_list[i]))


if __name__ == "__main__":
    img = cv2.imread("./yolo/office.jpg", 1)
    test_detect = Detect("./yolo/coco.names", "./yolo/yolov3-tiny.cfg", "./yolo/yolov3-tiny.weights")
    detect_picture, text_list = test_detect.run_one(img)
    print(text_list)
