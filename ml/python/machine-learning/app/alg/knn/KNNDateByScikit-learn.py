# coding:utf-8

from sklearn.neighbors import NearestNeighbors

from .KNNDateOnHand import file2matrix, autoNorm

if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # n_neighbors=3 表示查找的近邻数，默认是5
    # fit:用normMat作为训练集拟合模型   n_neighbors:几个最近邻 
    # NearestNeighbors 默认使用的就是欧式距离测度
    nbrs = NearestNeighbors(n_neighbors=3).fit(normMat)
    input_man = [28567, 9.968648, 0.731232]
    # input_man = [35948,6.830792,1.213192]
    # 数据归一化
    S = (input_man - minVals) / ranges
    # 找到当前点的K个临近点，也就是找到临近的3个点
    #  distance 返回的是距离数据集中最近点的距离,indices 返回的距离数据集中最近点的坐标的下标。
    distances, indices = nbrs.kneighbors([S])
    print("distances is %s" % distances)
    print("indices is %s" % indices)
    # classCount   K：类别名    V：这个类别中的样本出现的次数
    classCount = {}
    for i in range(3):
        # 找出对应的索引的类别号
        voteLabel = datingLabels[indices[0][i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda tp: tp[1], reverse=True)
    resultList = ['没感觉', '看起来还行', '极具魅力']
    print(resultList[sortedClassCount[0][0] - 1])
