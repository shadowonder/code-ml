import cv2
import matplotlib.pyplot as plt

"""
3. 图像的数值计算
"""

cat = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
dog = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\dog.jpg')

cat2 = cat + 10  # 相当于整体灰度增加, 每一个像素点的颜色都加10
print(cat[:5, :, 0])
print(cat2[:5, :, 0])
# 两个图片叠加一下,只打印前5行, 这里注意当相加的时候超过255的会从0计算. 注意相加维度必须相同
# 原因是类型是uint8最大就是255
print((cat + cat2)[:5, :, 0])

# cv2的相加属性, 超过255的都视为255
print(cv2.add(cat, cat2)[:5, :, 0])  # [[ 255,255,255 ... 255,255]]

# 图像融合, 简单的数值相加
# print(cat + dog)  # Error operands could not be broadcast together with shapes (259,194,3) (198,255,3) # 形状不同无法相加
print(cat.shape)  # (259, 194, 3)
dog2 = cv2.resize(dog, (194, 259))  # 对图像进行resize
print(dog2.shape)  # (259, 194, 3)
print(cat + dog2)

# 图像resize进行倍数压缩, 图像的x轴乘以3
res = cv2.resize(dog, (0, 0), fx=3, fy=1)
plt.imshow(res)
res = cv2.resize(dog, (0, 0), fx=1.5, fy=0.5)  # 横轴乘以1.5倍, 纵轴减半
plt.imshow(res)

# 图像融合
# 将cat的所有像素属性乘以0.4, dog的所有像素属性乘以0.5, 然后叠加最后所有像素加上6
res = cv2.addWeighted(cat, 0.4, dog2, 0.5, 6)
plt.imshow(res)
