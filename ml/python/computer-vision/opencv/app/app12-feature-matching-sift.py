import cv2
import numpy as np


def cv_show(title, picture):  # 图像显示函数
    cv2.imshow(title, picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
特征匹配
当获得了图像特征以后, 比如在当前案例我们获得了sift向量特征就可以对图像进行对比, 也就是对比向量之间的差异
"""

img1 = cv2.imread("E:\\Workspace\\ml\\code-ml\\ml\\python\\computer-vision\\data\\book.png")
img2 = cv2.imread("E:\\Workspace\\ml\\code-ml\\ml\\python\\computer-vision\\data\\books.png")

cv_show('img1', img1)
cv_show('img2', img2)

sift = cv2.SIFT_create()  # 将特征构造出来
# 构造出两幅图片的特征
kp1, des1 = sift.detectAndCompute(img1, None)  # kp1为关键点，des1为对应的特征向量
kp2, des2 = sift.detectAndCompute(img2, None)

"""
暴力匹配, 将两个图像的特征向量算出来, 然后进行匹配. 会进行归一化
"""
# crossCheck表示两个特征点相互匹配, 例如A中的第i个特征点与B中的第j个特征点最近，并且B中的第j个特征点到A中的第i个特征点也是
# NORM_L2：归一化数组的(欧几里得距离)，如果其他特征计算方法需要考虑不同的匹配计算方法
bf = cv2.BFMatcher(crossCheck=True)  # 让其两幅图的特征向量相互计算, crosscheck

# 1对1的匹配, 传入两个向量组
matches = bf.match(des1, des2)
# 排序一次 有利于计算
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# cv_show('img3', img3)

"""
k对最佳匹配, 类似k紧邻, 一个点相对于多个点进行匹配
制定一个过滤方法
"""
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 指定一个过滤方法, 由于是多个点,这里定义一个点A到另一个点B的距离为m, 到另一个点C的距离为n
# 当距离m和距离n相对于两个点的距离之比小于0.75, 那么这个点是我们想要的
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# cv_show('img3', img4)

cv_show("img", np.hstack((img3, img4)))
