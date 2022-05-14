import cv2
import numpy as np


def cv_show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
2. 票据识别
使用ocr进行票据检测
"""


# 轮廓是一个点矩阵(pts), 根据点的位置数值, 获取上下左右点的坐标
# :pts 轮廓坐标
def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 在原图的基础上将轮廓进行轮廓拉扯
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值, 比如上宽度和下宽度, 然后进行比较找出较大的值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    # 矩阵变换的原理就是通过旋转翻转等动作将图像拉伸.
    # 1. 将二维的图像变成三维的图像,[x,y,1], 增加了一个维度, 但是不改变坐标值
    # 2. 然后创建一个3x3的矩阵与上面的矩阵进行乘法运算, 原图4个点, 目标图4个点来求解方程,加上1就可以得到3x3的阵列
    # https://chowdera.com/2021/03/20210315203507951h.html
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 变换

    # 返回变换后结果
    return warped


# 通过height进行缩放操作
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 读取输入
image_location = "E:\Workspace\ml\code-ml\ml\python\computer-vision\opencv\examples\\page.jpg"

# 读取图像并对图像进行缩放操作, 这里缩放到height为500
image = cv2.imread(image_location)
# 坐标也会相同变化
ratio = image.shape[0] / 500.0  # 计算出缩放的比例
orig = image.copy()

image = resize(orig, height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波操作,去除噪音
edged = cv2.Canny(gray, 75, 200)  # 边缘检测

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
# 需要注意的是在cv2 3.0的版本中获取的轮廓index需要时1, 因为返回值在4.0为 contours, hierarchy, 而3.0返回三个值.
# 因此选择0就好了
# contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
print(cv2.contourArea)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # 前5个轮廓

# 遍历轮廓
for c in contours:
    # 计算轮廓近似, 因为轮廓未必会有4个点, 这里做一个近似处理, 找到需要的轮廓
    peri = cv2.arcLength(c, True)
    # C表示输入的点集, 也就是轮廓数据
    # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数. 从原始轮廓到目标轮廓转换的距离, 源码称为精准度. 越小越精致, 越大越粗狂
    # True表示封闭的
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果, 在原图上绘制我们需要的轮廓
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
# 第一个参数是原始图, 第二个参数是轮廓, 由于是原始图, 需要精度重新计算一次
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)  # 结果输出到目标目录
# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height=650))
cv2.imshow("Scanned", resize(ref, height=650))
cv2.waitKey(0)
