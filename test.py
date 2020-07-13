import os
labelsPath = os.path.sep
print(labelsPath)
import cv2
import tools
img=cv2.imread("./yolo/test.jpg")
tools.imshow("sd",img)
image=img.copy()
tools.imshow("sdfsdf",image)