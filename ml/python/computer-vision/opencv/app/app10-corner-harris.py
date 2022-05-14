import cv2
import numpy as np

"""
角点检测
"""
filename = 'E:\Workspace\ml\code-ml\ml\python\computer-vision\data\chessboard.jpeg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)  # 转化为float32
# 输入图像必须是float32，最后一个参数在0.04 到0.06 之间
dst = cv2.cornerHarris(gray, 2, 3, 0.06)
print(img.shape)  # (450, 600, 3)
print(dst.shape)  # (450, 600) 每个点平移操作后的计算结果

dst = cv2.dilate(dst, None)
# 如果图像中一个点, 在角点矩阵中的值大于角点矩阵最大值的1%那么这个点就被认为为角点
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 标记为红色
cv2.imshow('dst', img)
cv2.waitKey()
