import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import silhouette_score


def clusters():
    """
    聚类, 分为4个类别
    :return:
    """
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = "1"

    data = fetch_california_housing()
    x = data.data
    print(x)

    km = KMeans(n_clusters=8)
    km.fit(x)  # 进行分析

    predict = km.predict(x)
    print(predict)  # [2 0 2 ... 2 2 0]

    # 画个图
    plt.figure(figsize=(10, 10))
    # 建立4个颜色的列表
    colored = ['orange', 'green', 'blue', 'purple', 'cyan', 'magenta', 'yellow', 'red']
    color = [colored[i] for i in predict]  # 遍历predict, 每一个定义为colored[i], 组成新的array放入color

    plt.scatter(x[:, 1], x[:, 2], color=color)
    # plt.show()

    # 聚类评估
    print(silhouette_score(x, predict))  # 0.5242608023536229 大于0.5 会是比较不错的模型

    return None


if __name__ == '__main__':
    clusters()
