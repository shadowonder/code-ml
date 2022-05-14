from Stitcher import Stitcher
import cv2
import matplotlib.pyplot as plt


def cv_show(title, picture):  # 图像显示函数
    cv2.imshow(title, picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
图像拼接
>1. 获取一个矩阵H
>2. 原始点$x_q,y_q$和矩阵进行运算生成投影$x_p,y_p$
>3. 进行变换运算
>4. 根据结果可以定义一个损失函数
"""

img1 = cv2.imread("E:\\Workspace\\ml\\code-ml\\ml\\python\\computer-vision\\data\\stitch_left.jpeg")
img2 = cv2.imread("E:\\Workspace\\ml\\code-ml\\ml\\python\\computer-vision\\data\\stitch_right.jpeg")

stitcher = Stitcher()

(result, vis) = stitcher.stitch([img1, img2], showMatches=True)

cv_show("result", result)

plt.subplot(221), plt.imshow(img1)
plt.subplot(222), plt.imshow(img2)
plt.subplot(212), plt.imshow(result)
plt.show()
