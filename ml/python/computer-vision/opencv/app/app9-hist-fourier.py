import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
9. 直方图/傅里叶变换
"""
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg', 0)  # 0表示灰度图.
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist.shape)
plt.hist(img.ravel(), 256)
# plt.show()

img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
color = ('b', 'g', 'r')  # 这里的格式是bgr的格式绘图进入plotlib顺序不一样, 因此在这里定义
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
# plt.show()

# # 创建一个掩码, 默认为None
# # 掩码就是一个黑或者白的像素属性
# img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)  # 创建需要使用同样的大小
# mask[50:200, 50:150] = 255  # 需要把需要展示的指定为白色
# cv_show(mask, 'mask')

# # 这里就类似于一个截取动作
# img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg')
# masked_img = cv2.bitwise_and(img, img, mask=mask)
# hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])  # 不待掩码的时候
# hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()

# # 直方图的均衡化
# img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\cat.jpg", 0)
# plt.subplot(221).hist(img.ravel(), 256)
# plt.subplot(223).imshow(img)
#
# equ = cv2.equalizeHist(img)
# plt.subplot(222).hist(equ.ravel(), 256)
# plt.subplot(224).imshow(equ)
# plt.show()
#
# # 自适应均衡化
# img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\car.jpg", 0)
# equ = cv2.equalizeHist(img)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 切分为8*8的格子
# res_clahe = clahe.apply(img)
# res = np.hstack((img, equ, res_clahe))
# cv_show(res, 'res')

# # 2d 直方图
# img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\car.jpg")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# plt.imshow(hist, interpolation='nearest')  # nearest插值参数
# plt.show()

"""
傅里叶变换
"""

# DFT 变换
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\Lenna.jpg', 0)  # 读取灰度图像
img_float32 = np.float32(img)  # 输入图像先转换成32格式, 这是opencv的要求.

# 1. 执行傅里叶变换, 得到一个频谱图
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# 2. 在numpy中有fftshift方法, 执行shift操作将低频的值拉到中心
dft_shift = np.fft.fftshift(dft)

# 再次转换一下 使用映射公式, 对两个通道进行转换magnitude方法可以直接对通道进行操作
# 但是结果有点小, 我们需要扩展到0-255的区间, 下面的公式就是直接映射到0-255区间的公式: 20 * np.log
# 从而获得DFT的变换结果, 也就是频率结果. 可以看做是向外发散的频率图.
magnitude_spectrum = 20 * np.log(cv2.magnitude((dft_shift[:, :, 0]), dft_shift[:, :, 1]))

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Imput Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title(' Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 低通以及高通滤波
img = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\Lenna.jpg', 0)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # 时域转换到频域
dft_shift = np.fft.fftshift(dft)  # 将低频部分拉到中心处, 有利于展示和抓取高亮位置.

rows, cols = img.shape  # 计算图像宽高
crow, ccol = int(rows / 2), int(cols / 2)  # 确定掩膜的中心位置坐标

# # 低通滤波
# # 构建一个长宽一样的全零图像
# mask = np.zeros((rows, cols, 2), np.uint8)
# # 然后中间位置的上下左右30位置全部置为1, 其他位置为0
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
#
# # IDFT, 将dft转换为图像
# fshift = dft_shift * mask  # 去掉高频部分，只显示低频部分
# f_ishift = np.fft.ifftshift(fshift)  # 将低频部分从中心点处还原
# img_back = cv2.idft(f_ishift)  # 从频域逆变换到时域
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 该函数通过实部和虚部用来计算二维矢量的幅值, 反向处理
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_back, cmap='gray')
# plt.title('Result'), plt.xticks([]), plt.yticks([])
# plt.show()

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # 做反向操作, 只保留高频, 也就是外部的滤波

# IDFT, 将dft转换为图像
fshift = dft_shift * mask  # 去掉高频部分，只显示低频部分
f_ishift = np.fft.ifftshift(fshift)  # 将低频部分从中心点处还原
img_back = cv2.idft(f_ishift)  # 从频域逆变换到时域
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 该函数通过实部和虚部用来计算二维矢量的幅值, 反向处理
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()
