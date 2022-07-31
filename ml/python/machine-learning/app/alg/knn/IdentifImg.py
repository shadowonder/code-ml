# coding:utf-8

import os

import numpy as np

# 此方法将每个文件中32*32的矩阵数据，转换到1*1024一行中
from .KNNDateOnHand import classify


def img2vector(filename):
    # 创建一个1行1024列的矩阵
    returnVect = np.zeros((1, 1024))
    # 打开当前的文件
    fr = open(filename, "rb")
    # 每个文件中有32行，每行有32列数据，遍历32个行，将32个列数据放入1024的列中
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def IdentifImg():
    labels = []
    # 读取训练集 TrainData目录下所有的文件和文件夹
    trainingFileList = os.listdir('TrainData')
    m = len(trainingFileList)
    # zeros((m,1024)) 返回一个m行 ，1024列的矩阵，默认是浮点型的
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 获取文件名称  0_0.txt
        fileNameStr = trainingFileList[i]
        # 获取文件除了后缀的名称
        fileStr = fileNameStr.split('.')[0]
        # 获取文件"数字"的类别
        classNumStr = int(fileStr.split('_')[0])
        labels.append(classNumStr)
        # 构建训练集, img2vector  每个文件返回一行数据 1024列
        trainingMat[i, :] = img2vector('TrainData/%s' % fileNameStr)
    # 读取测试集数据
    testFileList = os.listdir('TestData')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]  # 0_0.txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('TestData/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, labels, 5)
        print("识别出的数字是: %d, 真实数字是: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\n识别错误次数 %d" % errorCount)
    errorRate = errorCount / float(mTest)
    print("\n正确率: %f" % (1 - errorRate))


if __name__ == '__main__':
    IdentifImg()
