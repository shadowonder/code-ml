import cv2
import numpy as np


def cv_show(title, cv_image):
    cv2.imshow(title, cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
6. 图像算子
"""

img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\pie.png')
cv_show('image', img)

# 当计算矩阵的时候会使用右边减左边的操作, 因此此时中心的颜色为白色, 右边的颜色为黑色, 因此左侧的圆弧就会出现.
# 但是在右侧的圆弧中,由于右边减左边的操作是黑色减白色, 因此数据就会出现负数的情况(白色为255). 因此我们可以对数据进行absolute处理
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# 进行绝对值处理
sobelx1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx1 = cv2.convertScaleAbs(sobelx1)

sobelx2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx2 = cv2.convertScaleAbs(sobelx2)

# 直接设定进行计算的结果并不准确, 不推荐直接计算
sobelx3 = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelx3 = cv2.convertScaleAbs(sobelx3)

res = np.hstack((img, sobelx, sobelx1, sobelx2, sobelx3))  # 展示
cv_show('show', res)

# 由于不推荐直接计算, 我们会进行叠加计算
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\Lenna.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('img', img)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy', sobelxy)

# 不同算子的差异
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\Lenna.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
cv_show('res', res)
