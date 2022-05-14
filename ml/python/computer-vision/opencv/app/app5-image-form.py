import cv2
import numpy as np


def cv_show(title, cv_image):
    cv2.imshow(title, cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
5. 图像形态学
"""

# 腐蚀操作
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dg.png')
# cv_show("dige", img)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
# cv_show("erosion", erosion)

# 膨胀
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(erosion, kernel, iterations=1)
# cv_show("dilate", dilate)

# 开：先腐蚀，再膨胀
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dg.png')
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv_show("opening", opening)

# 闭：先膨胀，再腐蚀
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dg.png')
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv_show("closing", closing)

# 梯度=膨胀-腐蚀
pie = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\pie.png')
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)

res = np.hstack((dilate, erosion))
# cv_show("res", res)
gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
# cv_show("gradient", gradient)


# 礼帽
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dg.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv_show('tophat', tophat)

# 黑帽
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dg.png')
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv_show('blackhat', blackhat)
