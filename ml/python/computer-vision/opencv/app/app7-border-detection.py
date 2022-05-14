import cv2
import numpy as np


def cv_show(title, cv_image):
    cv2.imshow(title, cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
7. 边界检测
"""

img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\Lenna.jpg')
cv_show('image', img)

# minVal和maxVal
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show('res', res)

img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\car.jpg", cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show('res', res)
