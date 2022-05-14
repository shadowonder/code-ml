import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(title, cv_image):
    cv2.imshow(title, cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
4. 阈值,平滑操作
"""

cat = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
dog = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dog.jpg')

cat_gray = cv2.cvtColor(cat, cv2.COLOR_RGB2GRAY)

ret, thresh1 = cv2.threshold(cat_gray, 127, 255, cv2.THRESH_BINARY)  # 大于127的都为白否则都是黑
ret, thresh2 = cv2.threshold(cat_gray, 127, 255, cv2.THRESH_BINARY_INV)  # 和上面的相反
ret, thresh3 = cv2.threshold(cat_gray, 127, 255, cv2.THRESH_TRUNC)  # 大于127的都为127, 否则不变
ret, thresh4 = cv2.threshold(cat_gray, 127, 255, cv2.THRESH_TOZERO)  # 大于127的不变, 否则变为0
ret, thresh5 = cv2.threshold(cat_gray, 127, 255, cv2.THRESH_TOZERO_INV)  # 色差翻转, 和上面的颜色翻转

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [cat_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

"""
平滑处理. 当图片存在噪音的时候可以对图片进行平滑处理
"""
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\lenaNoise.png')
cv_show("image", img)

# 均值滤波, 简单的平均卷及操作
# 其原理就是, 所有的点点变换为周围点的均值
# 方法就是构建3x3的全1矩阵然后矩阵相乘最终求平均值 也就是相加对除
# 整个图将会变得模糊
blur = cv2.blur(img, (3, 3))  # 每个点相对于3x3的矩阵进行运算
# cv_show("blur", blur)

# 方框滤波
# 基本和均值一样, 可以选择归一化, 一旦进行了归一化均值就和方框是一样的
# 当越界现象就会直接使用255
box = cv2.boxFilter(img, -1, (3, 3), normalize=False)  # 每个点相对于3x3的矩阵进行运算
# cv_show("box", box)

# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
# 越接近取值就越相似, 原理就是使用矩阵, 创建一个从0.0到1.0的权重型矩阵, 中心就是本pixel,
#   当出现与本属性相似的值的时候权重就高, 否则如果数值相差较大权重就地, 然后使用权重矩阵与数据矩阵相乘
gaus = cv2.GaussianBlur(img, (5, 5), 1)
# cv_show("gaus", gaus)

# 中值滤波
# 相当于用中值代替, 去周围像素点的中间值对比
median = cv2.medianBlur(img, 5)
# cv_show("median", median)

# 展示所有的
res = np.hstack((img, blur, gaus, median))  # 横向拼接
# res = np.vstack((img, blur, gaus, median))  # 竖着拼接
# print (res)
cv2.imshow('median vs average', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
