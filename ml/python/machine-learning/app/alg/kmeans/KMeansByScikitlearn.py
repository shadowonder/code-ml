# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 建立12*12英寸  新的图像
plt.figure(figsize=(12, 12))
n_samples = 1500
random_state = 170
'''
        make_blobs函数是为聚类产生数据集 ， 产生一个数据集和相应的标签 
        n_samples:表示数据样本点个数,默认值100 
        n_features:表示数据的维度，特征，默认值是2 
        centers:产生数据的中心点，默认值3个 
        shuffle ：洗乱，默认值是True 
        random_state:官网解释是随机生成器的种子 
'''
# x返回的是向量化的数据点，y返回的是对应数据的类别号
x, y = make_blobs(n_samples=n_samples, random_state=random_state)
print('x=', x, type(x), 'y=', y, type(y))
# 使用KMeans去聚类,返回聚好的类别集合, n_clusters聚合成几类
y_pred = KMeans(n_clusters=3).fit_predict(x)
print("y_pred : ", y_pred)
# subplot 绘制多个子图，221 等价于2,2,1 表示两行两列的子图中的第一个
plt.subplot(221)
# scatter 绘制散点图   ,c 指定颜色
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("kmeans01")

transformation = [[0.60834549, -0.63667341],
                  [-0.40887718, 0.85253229]]
# numpy.dot 矩阵相乘
# a1= [[1,2]
#     [3,4]
#     [5,6]]
# a2= [[10,20]
#     [30,40]]
# a1*a2 = [[1*10+2*30,1*20+2*40]
#         [3*10+4*30,3*20+4*40]
#         [5*10+5*30,6*20+6*40]
#            ]
X_aniso = np.dot(x, transformation)
y_pred = KMeans(n_clusters=3).fit_predict(X_aniso)
plt.subplot(222)
# # s 设置点的大小,s=8
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("kmeans02")

# vstack 是合并矩阵，将y=0类别的取出500行，y=1类别的取出100行，y=2类别的取出10行
X_filtered = np.vstack((x[y == 0][:500], x[y == 1][:100], x[y == 2][:200]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
plt.subplot(223)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("kmeans03")

dataMat = []
fr = open("testSet.txt", "rb")
for line in fr.readlines():
    if line.decode("utf-8").strip() != "":
        curLine = line.decode("utf-8").strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
dataArray = np.array(dataMat)
print("dataArray type = %s" % type(dataArray))
# 调用Scikitlearn中的KMeans
# KMeans 中参数 init='k-means++' 默认就是k-means++  如果设置为'random'是随机找中心点
y_pred = KMeans(n_clusters=6).fit_predict(dataArray)  # 聚类为6类
plt.subplot(224)
plt.scatter(dataArray[:, 0], dataArray[:, 1], c=y_pred)
plt.title("kmeans04")
plt.savefig("./kmeans.png")
plt.show()
