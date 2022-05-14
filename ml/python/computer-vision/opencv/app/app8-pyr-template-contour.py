import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
8. 轮廓处理方法
"""

# 读取数据, 转换为灰度图, 可以更好地进行边缘检测
img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\contours.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv_show(thresh, 'thresh')

# 绘制轮廓, 需要注意旧版中可以获得原图像作为返回值的第一个值, 新版中被省略
# binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv_show(binary, 'binary')  # 和上面的图像相同
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(contours)  # 轮廓的信息

# 绘制轮廓
# 传入绘制图像, 轮廓, 轮廓索引, 轮廓颜色模式, 线条厚度
# 轮廓索引是包括内圈,外圈以及不同轮廓的索引, -1表示展示全部
draw_img = img.copy()  # 需要注意, 轮廓会直接绘制在原图上
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show(draw_img, 'draw_img')

# 轮廓特征计算
# 获取轮廓的特征
cnt = contours[0]
print(cv2.contourArea(cnt))  # 计算面积
print(cv2.arcLength(cnt, True))  # 计算周长, true表示计算闭合的

# 轮廓近似
# 生成轮廓
img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\contours2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
# cv_show(res, 'res')

# 阈值是0.1倍的周长, 也就是垂线的阈值, 越小的话边框越精细
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
# cv_show(res, 'res')

# 创建一个外接矩形, 起始绘图点, 长宽, 颜色, 笔刷粗细
x, y, w, h = cv2.boundingRect(cnt)
outer_rec = img.copy()
img = cv2.rectangle(outer_rec, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv_show(outer_rec, "outer_rec")

img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\contours.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]  # 0外接,1内切,2+换其他图

x, y, w, h = cv2.boundingRect(cnt)
rec_img = img.copy()
rec_img = cv2.rectangle(rec_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv_show(rec_img, "img")

area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
react_area = w * h
extent = float(area) / react_area
print('轮廓面积与边界矩形比:', extent)

# 外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
circle_image = img.copy()
circle_image = cv2.circle(circle_image, center, radius, (0, 255, 0), 2)
# cv_show(circle_image, "img")

"""
高斯金字塔
"""
img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\\fun.jpg")
# cv_show(img, 'img')
print(img.shape)

# 向上采样, 扩大
up = cv2.pyrUp(img)
# cv_show(up, 'up')
print(up.shape)

# 向下采样
down = cv2.pyrDown(img)
# cv_show(down, 'down')
print(down.shape)

# 原始采样修改
up = cv2.pyrUp(img)
up_down = cv2.pyrDown(up)
print(up_down.shape)
up_down = np.hstack((img, up_down, img - up_down))  # 尝试一下减法
# cv_show(up_down, 'up_down')

# 拉普拉斯金字塔
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
# 注意这里可能会出现图像大小不同的情况
# lv_1 = img - down_up
# cv_show(lv_1, 'lv_1')

"""
模板匹配
"""
img = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\Lenna.jpg")
template = cv2.imread("E:\Workspace\ml\code-ml\ml\python\computer-vision\data\lena-cut.jpg")
h, w = template.shape[:2]
print(img.shape)  # (316, 316, 3)
print(template.shape)  # (125, 101, 3)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img, template, cv2.TM_CCORR)
print(res.shape)  # (192, 216)

# opencv中给出了这个函数,可以找到最大最小的位置和最大最小值
# 表示用当前方式给出的结果. 由于我们给定的计算方法结果是使用的归一化, 因此minloc其实是计算出来的位置, 因此我们可以调取
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val, max_val, min_loc, max_loc)  # 0.0008121237624436617 0.40212324261665344 (120, 113) (41, 191)

fig, axs = plt.subplots(3, 4)

for m_index in range(len(methods)):
    m = methods[m_index]
    img2 = img.copy()
    # 匹配方法的真值
    method = eval(m)
    print(m_index, method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    # 绘图, 放入plot中
    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    # plt.subplot(122), plt.imshow(img2, cmap='gray')
    # plt.xticks([]), plt.yticks([])
    # plt.suptitle(m)
    # plt.show()
    sub_set_index = m_index % 2
    sub_set_index_first = sub_set_index * 2
    sub_set_index_second = sub_set_index * 2 + 1
    print(int(m_index / 2), sub_set_index_first, sub_set_index_second)

    axs[int(m_index / 2), sub_set_index_first].imshow(res, cmap='gray')
    axs[int(m_index / 2), sub_set_index_first].set_xticks([])
    axs[int(m_index / 2), sub_set_index_first].set_yticks([])
    axs[int(m_index / 2), sub_set_index_first].set_title(m)
    axs[int(m_index / 2), sub_set_index_second].imshow(img2, cmap='gray')
    axs[int(m_index / 2), sub_set_index_second].set_xticks([])
    axs[int(m_index / 2), sub_set_index_second].set_yticks([])
    axs[int(m_index / 2), sub_set_index_second].set_title(m)

# plt.show()

# 多模板匹配
img_rgb = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\mario-coin.jpg', cv2.IMREAD_GRAYSCALE)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)  # 返回每一个窗口的结果值
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 1)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)
