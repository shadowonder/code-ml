# encoding:utf-8import numpy as np# 将每行数据放入一个数组内列表，返回一个二维列表def loadDataSet(fileName):    # 创建空列表    dataMat = []    fr = open(fileName, "rb")  # 打开文件    for line in fr.readlines():        # 按照制表"\t"符切割每行，返回一个列表list        curLine = line.decode("utf-8").strip().split('\t')        # 将切分后的每个列表中的元素，以float形式返回，map()内置函数，返回一个map object【注意，在python2.7版本中map直接返回list】，这里需要再包装个list        fltLine = list(map(float, curLine))        # [[k,v],[k1,v1]...]        dataMat.append(fltLine)    return dataMat# 两点欧式距离def distEclud(vecA, vecB):    # np.power(x1,x2)  对x1中的每个元素求x2次方,不会改变x1。    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))# 随机找到3个中心点的位置坐标，返回一个3*2的矩阵def randCent(dataSet, k):    # 返回dataSet列数,2列    n = np.shape(dataSet)[1]    '''        centerids是一个3*2的矩阵，用于存储三个中心点的坐标    '''    centerids = np.mat(np.zeros((k, n)))    for j in range(n):        # 统计每一列的最小值        minJ = min(dataSet[:, j])        # 每列最大值与最小值的差值        rangeJ = float(max(dataSet[:, j]) - minJ)        # np.random.rand(k,1) 产生k行1列的数组，里面的数据是0~1的浮点型 随机数。        array2 = minJ + rangeJ * np.random.rand(k, 1)        # 转换成k*1矩阵 赋值给centerids        # centerids[:, j] = np.mat(array2)        centerids[:, j] = array2    return centeridsdef kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):    # 计算并获取矩阵所有 行数  m=80    m = np.shape(dataSet)[0]    # zeros((m,2)) 创建一个80行，2列的二维数组    # numpy.mat 将二维数组转换成矩阵  【类别号，当前点到类别号中心点的距离】    clusterAssment = np.mat(np.zeros((m, 2)))    # createCent找到K个随机中心点坐标    centerids = createCent(dataSet, k)    #     print centerids    clusterChanged = True    while clusterChanged:        clusterChanged = False        # 遍历80个数据到每个中心点的距离        for i in range(m):            # np.inf float的最大值，无穷大            minDist = np.inf            # 当前点属于的类别号            minIndex = -1            # 每个样本点到三个中心点的距离            for j in range(k):                # x = centerids[j,:]                # print x                # 返回两点距离的值                distJI = distMeas(centerids[j, :], dataSet[i, :])                if distJI < minDist:                    # 当前最小距离的值                    minDist = distJI                    # 当前最小值属于哪个聚类                    minIndex = j            # 有与上次迭代计算的当前点的类别不相同的点            if clusterAssment[i, 0] != minIndex:                clusterChanged = True            # 将当前点的类别号和最小距离 赋值给clusterAssment的一行            clusterAssment[i, :] = minIndex, minDist        for cent in range(k):            # array = clusterAssment[:,0].A==cent            # result = np.nonzero(clusterAssment[:,0].A==cent)[0]            # clusterAssment[:,0].A 将0列 也就是类别号转换成数组            # clusterAssment[:,0].A==cent 返回的是一列，列中各个元素是 True或者False,True代表的是当前遍历的cent类别            # np.nonzero(clusterAssment[:,0].A==cent)  返回数组中值不为False的元素对应的行号下标数组 和列号下标数组            # currNewCenter 取出的是对应是当前遍历cent类别的 所有行数据组成的一个矩阵            currNewCenter = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]            # numpy.mean 计算矩阵的均值，axis=0计算每列的均值，axis=1计算每行的均值。            # 这里是每经过一次while计算都会重新找到各个类别中中心点坐标的位置  ，axis = 0 是各个列求均值            centerids[cent, :] = np.mean(currNewCenter, axis=0)            # 返回 【 当前三个中心点的坐标】 【每个点的类别号，和到当前中心点的最小距离】    return centerids, clusterAssmentif __name__ == '__main__':    # numpy.mat 将数据转换成80*2的矩阵    dataMat = np.mat(loadDataSet('./testSet.txt'))    k = 3    # distEclud 计算距离的方法    # centerids 三个中心点的坐标。clusterAssment 每个点的类别号|到当前中心点的最小距离    centerids, clusterAssment = kMeans(dataMat, k, distMeas=distEclud, createCent=randCent)    print(centerids)    print(clusterAssment)