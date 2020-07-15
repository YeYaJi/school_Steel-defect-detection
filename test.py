# import os
# labelsPath = os.path.sep
# print(labelsPath)
# import cv2
# import tools
# img=cv2.imread("./yolo/test.jpg")
# tools.imshow("sd",img)
# image=img.copy()
# tools.imshow("sdfsdf",image)


# from multiprocessing import Pool
# import os
# import time
# import random
# def long_time_task(name):
#     print('Run task %s (%s)...' % (name, os.getpid()))
#     start = time.time()
#     time.sleep(random.random() * 3)
#     end = time.time()
#     print('Task %s runs %0.2f seconds.' % (name, (end - start)))
#
#
# if __name__ == '__main__':
#     print('Parent process %s.' % os.getpid())
#     p = Pool(80)
#     for i in range(81):
#         p.apply_async(long_time_task, args=(i,))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     print('All subprocesses done.')


# import subprocess
#
# print('$ nslookup www.python.org')
# r = subprocess.call(['nslookup', 'www.python.org'])
# print('Exit code:', r)

from multiprocessing import Pool, Process
import subprocess
import os
import random

# m = random.randint(1, 100)
#
#
# def test(i):
#     print(i)
#
#
# # subprocess.run(["ls", "-l"])
#
# q1 = Process(target=test, args=(m,))
# m = random.randint(1, 100)
# q2 = Process(target=test, args=(m,))
# m = random.randint(1, 100)
# q3 = Process(target=test, args=(m,))
# m = random.randint(1, 100)
# q4 = Process(target=test, args=(m,))
#
# q1.start()
# q2.start()
# q3.start()
# q4.start()
#
# q1.join()
# q2.join()
# q3.join()
# q4.join()
import cv2

# def video_read(i):
#
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("相机故障")
#         exit()
#     while 1:
#         ret, frame = cap.read()
#         if not ret:
#             print("没有读到视频流")
#             break
#
#         cv2.namedWindow("video"+str(i), cv2.WINDOW_NORMAL)
#         cv2.imshow("video"+str(i), frame)
#         wait = cv2.waitKey(1)  # 设置图像显示时间为1毫秒
#
#         if wait == ord("q"):
#             break
#
# q1=Process(target=video_read,args=(1,))
# q2=Process(target=video_read,args=(2,))
#
# q1.start()
# q2.start()

# q.join()
# q.join()


from multiprocessing import Process, Queue

#
# img=cv2.imread("./yolo/office.jpg",1)
# dst=cv2.GaussianBlur(img,(189,189),100)
# cv2.imwrite("123.jpg",dst)
#
# while True:
#
#     a = input("输入over结束传入截图")
#     b = a
#     print(b)
#     if b == "200":
#         break
#
# print("结束")


# import time
# a=time.localtime()
# print(a)
# time_tuple = time.localtime( time.time())
# print(time_tuple)

