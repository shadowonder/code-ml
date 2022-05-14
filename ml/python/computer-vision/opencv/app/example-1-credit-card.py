import cv2
from imutils import contours
import numpy as np


def cv_show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


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


"""
1. 银行卡案例
"""
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# ap.add_argument("-r", "--reference", required=True,
#                 help="path to reference OCR-A image")
# args = vars(ap.parse_args())
#
# # define a dictionary that maps the first digit of a credit card
# # number to the credit card type
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

reference_number_location = 'E:\Workspace\ml\code-ml\ml\python\computer-vision\opencv\examples\\reference.png'
card_image_location = 'E:\Workspace\ml\code-ml\ml\python\computer-vision\opencv\examples\\cc-trio.jpg'

# 读取一个模板图像
img = cv2.imread(reference_number_location)
# cv_show('img', img)
# 展示模板的灰度图
gray_ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref', gray_ref)
# 二值图像
ref = cv2.threshold(gray_ref, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('二值图像', ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，
# 即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
# 返回值中我们只需要轮廓contours
ref_contours, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, ref_contours, -1, (0, 0, 255), 3)  # -1表示获取所有轮廓, 然后渲染到红色
cv_show('轮廓', img)  # 展示一下轮廓

print("ref中轮廓的数量:", np.array(ref_contours).shape)
# 排序所有的坐标, 从左到右单项排序. 从而获得0-9的数字和坐标的index对其
ref_contours = sort_contours(ref_contours, method="left-to-right")[0]  # 排序，从左到右，从上到下
digits = {}

# 遍历每一个轮廓, c=轮廓 i=index
for (i, c) in enumerate(ref_contours):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)  # 通过轮廓切出想要的矩形
    roi = ref[y:y + h, x:x + w]  # 切出来
    roi = cv2.resize(roi, (57, 88))  # resize一下

    # 每一个数字对应每一个模板
    digits[i] = roi  # 字典保存模板

# 初始化卷积核, 也就是定义坐标pixel矩阵的大小
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像，预处理
image = cv2.imread(card_image_location)
cv_show('image', image)
image = resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 这里只使用了左侧的sobel算子, 没有使用y算子, 可以同时是用来获取边界锯齿数据. 但是这里只是用x作为参考系
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize=-1相当于用3*3的
                  ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起. 相比于其他的图片就不会被过滤
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)
# 二值化处理, THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
# 设置为0的时候, 由于是双峰, 系统会自动的判断, 不是匹配判断
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 再来一个闭操作, 这样目标数据就展示的更加清晰
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)  # 再来一个闭操作
cv_show('thresh', thresh)

# 计算轮廓
# thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
thresh_contours, hierarchy = cv2.findContours(thresh.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

cnts = thresh_contours
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('contoursAfter', cur_img)  # 将经过各种处理以后的轮廓画到原始图像中
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    # 计算合适的矩形比例, 通过常量获取
    if ar > 2.5 and ar < 4.0:

        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)

    # 预处理, 二值化
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)

    # 计算每一组的轮廓
    digit_contours, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = contours.sort_contours(digit_contours, method="left-to-right")[0]  # 排序一次

    # 计算每一组中的每一个数值
    for c in digit_contours:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)  # 获取当前digit的得分
            scores.append(score)

        # 得到最合适的数字, 也就是最大的得分
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card #: {}".format("".join(output)))
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
cv2.imshow("Image", image)
cv2.waitKey(0)
