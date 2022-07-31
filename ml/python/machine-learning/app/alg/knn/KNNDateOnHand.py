# coding:utf-8

# matplotlib 绘图模块
import matplotlib.pyplot as plt
import numpy as np


# from array import array
# from matplotlib.font_manager import FontProperties

# normData 测试数据集的某行，  dataSet 训练数据集 ，labels 训练数据集的类别，k k的值
def classify(normData, dataSet, labels, k):
    # 计算行数
    dataSetSize = dataSet.shape[0]
    #     print ('dataSetSize 长度 =%d'%dataSetSize)
    # 当前点到所有点的坐标差值  ,np.tile(x,(y,1)) 复制x 共y行 1列
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet
    # 对每个坐标差值平方
    sqDiffMat = diffMat ** 2
    # 对于二维数组 sqDiffMat.sum(axis=0)指 对向量每列求和，sqDiffMat.sum(axis=1)是对向量每行求和,返回一个长度为行数的数组
    # 例如：narr = array([[ 1.,  4.,  6.],
    #                   [ 2.,  5.,  3.]])
    #    narr.sum(axis=1) = array([ 11.,  10.])
    #    narr.sum(axis=0) = array([ 3.,  9.,  9.])
    sqDistances = sqDiffMat.sum(axis=1)
    # 欧式距离 最后开方
    distance = sqDistances ** 0.5
    # x.argsort() 将x中的元素从小到大排序，提取其对应的index 索引，返回数组
    # 例：   tsum = array([ 11.,  10.])    ----  tsum.argsort() = array([1, 0])
    sortedDistIndicies = distance.argsort()
    #     classCount保存的K是魅力类型   V:在K个近邻中某一个类型的次数
    classCount = {}
    for i in range(k):
        # 获取对应的下标的类别
        voteLabel = labels[sortedDistIndicies[i]]
        # 给相同的类别次数计数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # sorted 排序 返回新的list
    #     sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


# 获取文件的x和y
def file2matrix(filename):
    fr = open(filename, "rb")
    # readlines:是一次性将这个文本的内容全部加载到内存中(列表)
    arrayOflines = fr.readlines()
    numOfLines = len(arrayOflines)
    #     print "numOfLines = " , numOfLines
    # numpy.zeros 创建给定类型的数组  numOfLines 行 ，3列
    returnMat = np.zeros((numOfLines, 3))
    # 存结果的列表, 存放y值
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        # 去掉一行的头尾空格
        line = line.decode("utf-8").strip()
        listFromline = line.split('\t')
        returnMat[index, :] = listFromline[0:3]  # 将前三个值复制到目标行
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


'''
    将训练集中的数据进行归一化
    归一化的目的：
        训练集中飞行公里数这一维度中的值是非常大，那么这个纬度值对于最终的计算结果(两点的距离)影响是非常大，
        远远超过其他的两个维度对于最终结果的影响
    实际约会姑娘认为这三个特征是同等重要的
    下面使用最大最小值归一化的方式将训练集中的数据进行归一化
'''


# 将数据归一化
def autoNorm(dataSet):
    #     dataSet.min(0)   代表的是统计这个矩阵中每一列的最小值     返回值是一个矩阵1*3矩阵
    # 例如： numpyarray = array([[1,4,6],
    #                        [2,5,3]])
    #    numpyarray.min(0) = array([1,4,3])    numpyarray.min(1) = array([1,2])
    #    numpyarray.max(0) = array([2,5,6])    numpyarray.max(1) = array([6,5])
    minVals = dataSet.min(0)  # 返回列表, 每一列的最小值, 0表示为空则为默认值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # dataSet.shape[0] 计算行数， shape[1] 计算列数
    m = dataSet.shape[0]

    #     print '行数 = %d' %(m)
    #     print maxVals

    #     normDataSet存储归一化后的数据
    #     normDataSet = np.zeros(np.shape(dataSet))
    # np.tile(minVals,(m,1)) 在行的方向上重复 minVals m次 即复制m行，在列的方向上重复munVals 1次，即复制1列
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    # 归一化以后的数据, 数据范围列表, 最小值矩阵
    return normDataSet, ranges, minVals


def datingClassM():
    rate = 0.1
    # 获取x和y
    datingDataMat, datingLabels = file2matrix('../../../../data/datingTestSet.txt')
    # 将数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 是 ： normMat行数 = 1000
    m = normMat.shape[0]
    #     print 'm =%d 行'%m
    # 取出100行数据测试
    numTestVecs = int(m * rate)
    errorCount = 0.0
    for i in range(numTestVecs):
        # normMat[i,:] 取出数据的第i行,normMat[numTestVecs:m,:]取出数据中的100行到1000行 作为训练集，
        # datingLabels[numTestVecs:m] 取出数据中100行到1000行的类别，4是K
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 5)
        print('模型预测值: %d ,真实值 : %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    errorRate = errorCount / float(numTestVecs)
    print('正确率 : %f' % (1 - errorRate))
    return 1 - errorRate


'''
    拿到每条样本的飞行里程数和玩视频游戏所消耗的时间百分比这两个维度的值，使用散点图
'''


def createScatterDiagram():
    datingDataMat, datingLabels = file2matrix('../../../../data/datingTestSet.txt')
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    # 生成一个新的图像
    fig = plt.figure()
    # matplotlib下, 一个 Figure 对象可以包含多个子图(Axes), 可以使用 subplot() 快速绘制
    # subplot(numRows, numCols, plotNum)图表的整个绘图区域被分成 numRows 行和 numCols 列,按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
    # plt.subplot(111)等价于plt.subplot(1,1,1)
    axes = plt.subplot(111)
    # 设置字体 黑体  ，用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:  # 不喜欢
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:  # 魅力一般
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:  # 极具魅力
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    # 绘制散点图 ，前两个参数表示相同长度的数组序列 ，s 表示点的大小， c表示颜色
    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
    plt.title(u'标题')
    plt.xlabel(u'每年飞行里程数')
    plt.ylabel(u'玩视频游戏所消耗的时间百分比')
    # loc 设置图例的位置 2是upper left
    axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)
    #     plt.scatter(datingDataMat[:,0],datingDataMat[:,1],c = datingLabels)
    plt.show()


def classifyperson():
    resultList = ['没感觉', '看起来还行', '极具魅力']
    # input_man = [14000, 1.2, 0.6]
    input_man = [72993, 10.141740, 1.032955]
    datingDataMat, datingLabels = file2matrix('../../../../data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    result = classify((input_man - minVals) / ranges, normMat, datingLabels, 5)
    print('你即将约会的人是:%s' % resultList[result - 1])


if __name__ == '__main__':
    #     createScatterDiagram观察数据的分布情况
    # createScatterDiagram()
    acc = datingClassM()
    if (acc > 0.9):
        classifyperson()
