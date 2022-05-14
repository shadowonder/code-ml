import cv2
import numpy as np

"""
源代码抄录自
https://github.com/shekkizh/ImageProcessingProjects/blob/master/PythonProjects/ImageManipulation/ImageStitching.py
"""


class Stitcher:
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images  # 获取输入图片
        # 检测A,B图片的sift关键特征点, 计算特征描述
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        # 匹配两张图片所有的特征点, 返回匹配的结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        # 如果返回结果为空就是没有任何匹配的特征点, 返回none
        if M is None:
            return None

        # apply a perspective warp to stitch the images
        # together
        # 提取匹配结果, 在这里同时提取H矩阵
        (matches, H, status) = M
        # 进行透视变换
        # 这里的result就是将图片变形后的操作, 这里需要注意的是, 结果附带了第二张图片的位置, 有一部分是给第二张预留的
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        self.cv_show("image", result)

        # 第二张图片嵌入结果
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    # 计算特征
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 使用灰度图
        descriptor = cv2.SIFT_create()  # 创建特征构造器算法
        (kps, features) = descriptor.detectAndCompute(image, None)  # 计算对应的特征向量

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        # 转换成32位的格式, 特征需求
        kps = np.float32([kp.pt for kp in kps])  # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        # 获取暴力匹配
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # 获取knn匹配, k为2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        # 循环所有的点
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            # 过滤操作, 当最近距离更次近距离的本质小于ratio的时候保留配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 储存两个点的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

                # computing a homography requires at least 4 matches

        # 当筛选结束后, 如果匹配点大于4的时候才进行矩阵变换操作
        if len(matches) > 4:
            # construct the two sets of points
            # 获取匹配成功的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            # 计算视角变换矩阵, 过滤掉不合适的特征点, 通过上面4个特征点的形状匹配最好的H矩阵
            # 也就是迭代方程
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

            # return the visualization
        return vis

    def cv_show(self, title, picture):  # 图像显示函数
        cv2.imshow(title, picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class SurfStitcher:
    def __init__(self, image, ratio=0.75, reprojThresh=4.0):
        self.leftImage = image
        self.ratio = ratio
        self.reprojThresh = reprojThresh

        self.surfFeature = cv2.xfeatures2d.SURF_create(500, extended=False, upright=True)
        # HessianThreshold = 500
        # No orientation calculation
        # 64 dimension feature vector

        self.matcher = cv2.DescriptorMatcher_create('BruteForce')

        self.leftKps, self.leftDescriptor = self.detectAndDescribe(image)

    def detectAndDescribe(self, image):
        kps, des = self.surfFeature.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, des

    def stitch(self, image):
        print
        "stitch called"
        # cv2.imshow("StitchImage", utils.image_resize(image, height = 400))
        # cv2.waitKey()
        # cv2.destroyWindow("StitchImage")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rightKps, rightDescriptor = self.detectAndDescribe(gray)
        H = self.getHomography(rightKps, rightDescriptor)
        if H is None:
            return None
        leftImageShape = self.leftImage.shape
        result = cv2.warpPerspective(image, H, (leftImageShape[1] + image.shape[1], image.shape[0]))
        result[0:leftImageShape[0], 0:leftImageShape[1]] = self.leftImage

        #     Update leftImage stats
        print("Stitch done!")
        self.leftImage = result
        self.leftKps = rightKps
        self.leftDescriptor = rightDescriptor
        return

    def getHomography(self, rightKps, rightDescriptor):
        rawMatches = self.matcher.knnMatch(self.leftDescriptor, rightDescriptor, 2)
        matches = []

        for m in rawMatches:
            if (len(m) == 2 and m[0].distance < m[1].distance * self.ratio):
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if (len(matches) >= 4):
            # print(matches)
            ptsB = np.float32([self.leftKps[i] for (_, i) in matches])
            ptsA = np.float32([rightKps[i] for (i, _) in matches])

            # ptsB = H*ptsA
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)
            return H

        return None

    def saveImage(self):
        # cv2.imshow("Image", utils.image_resize(self.leftImage, width = 900))
        # cv2.waitKey()
        cv2.imwrite("stitchedImage.jpg", self.leftImage)
