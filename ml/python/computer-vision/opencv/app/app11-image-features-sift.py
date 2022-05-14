import cv2
import numpy as np

"""
sift算法
在opencv4版本以后, SIFT算法就进入了付费使用, 专利保护. 如果需要的话, 需要下降版本到4以前的版本, 以前的方法为cv2.xfeatures2d.SIFT_create()
Opencv的SIFT算法使用的包是 opencv-contrib-python 如果需要调整版本, 可以尝试只调整这个. 但是可能会出现版本不兼容的情况
2020年opencv4也进入了正式sift库, 所以可以调用SIFT_create()方法了
"""
filename = 'E:\Workspace\ml\code-ml\ml\python\computer-vision\data\chessboard.jpeg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""得到特征点"""
sift = cv2.SIFT_create()

kp = sift.detect(gray, None)  # 获取关键点
img = cv2.drawKeypoints(gray, kp, img)  # 将关键点绘制到图片上

cv2.imshow('drawKeypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 计算特征
kp, des = sift.compute(gray, kp)
print(np.array(kp).shape)

print(des.shape)
print(des[0])
