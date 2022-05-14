import cv2
import matplotlib.pyplot as plt

"""
1. 图片的基本操作
"""

# 需要注意的是cv2默认读取图片格式是BGR而不是rgb格式,因此读取颜色的时候要注意
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
print(img)  # 一个numpy中ndarray的结构[h,w,c], type是uint8, 因为最小值为0, 最大值为255


# [[[183 157 150]
#   [193 167 160]
#   [203 177 170]
#   ...

# # 展示图片
# cv2.imshow('image', img)  # 第一个参数表示的是标题
# # 图片会马上关闭, 需要安排等待时间, 毫秒, 0表示手动终止
# cv2.waitKey(0)
# cv2.destroyAllWindows()  # 终止的时候直接关闭窗口
def cv_show(title, cv_image):
    cv2.imshow(title, cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(img.shape)  # 展示数据的维度(h,w,c)
# (259, 194, 3)

# 读取灰度图像, 黑白图像
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)  # (259, 194)
# cv_show("gray",img)

# 保存图片
cv2.imwrite('testCat.png', img)

print(type(img))  # 属性
print(img.size)  # 大小
print(img.dtype)  # dtype uint8

cat = img[0:50, 0:50]  # 截取图片x:0-50 y:0-50
# cv_show("cat", cat)

# 将三个通道导出到3个变量中
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
b, g, r = cv2.split(img)
print(b)
print(r.shape)  # (259, 194)

# 复制一个图像, 将绿色蓝色定义为0, 也就是删除其他颜色的比重
red_img = img.copy()
red_img[:, :, 0] = 0
red_img[:, :, 1] = 0
# cv_show("R", red_img)

green_img = img.copy()
green_img[:, :, 0] = 0
green_img[:, :, 2] = 0
# cv_show("R", green_img)

blue_image = img.copy()
blue_image[:, :, 1] = 0
blue_image[:, :, 2] = 0
# cv_show("R", blue_image)

# 便捷填充, 将图像扩大
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# 定义五种不同的填充方法, 分别使用不同的常量定义. 按照什么样的方式进行填充
# copyMakeBorder: 复制图像并添加border
# BORDER_REPLICATE: 复制, 将最边缘的像素点填充到当前行和列
# BORDER_REFLECT: 反射, 将像素点反向填充,图像像素类似: fedcba|abcdefg|hgfedcb
# BORDER_REFLECT101: 另一种反射 gfedcb|abcdefgh|gfedcba
# BORDER_WRAP: 包装: cdefgh|abcdefgh|abcdefg
# BORDER_CONSTANT: 常量填充, 这个就是颜色
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                              value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('BORDER_REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('BORDER_REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('BORDER_REFLECT101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('BORDER_WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('BORDER_CONSTANT')

plt.show()
