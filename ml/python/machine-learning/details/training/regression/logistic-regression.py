import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def logistic_regression():
    """
    逻辑分类二分类进行癌症预测, 根据细胞特征
    :return:
    """
    # 读取数据
    """
    pd.read_csv(’’,names=column_names)
        column_names：指定类别名字,['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
        return:数据
        
        replace(to_replace=’’,value=)：返回数据
        dropna():返回数据
    """
    # 构造列名
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column_names)
    print(data)

    # 对缺失值进行处理, 把?换成np.nan
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()  # 删除所有的nan, 这里可以针对每一列进行均值配置, 也可以比较懒, 直接drop

    # 数据分隔
    # train_test_split(data[column_names[1:10]], data['Class'], test_size=0.25)
    # 特征值为第二列到第10列, 结果值为第11列
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                        test_size=0.25)

    # 进行标准化处理, 由于是分类算法, 不需要进行结果标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 逻辑回归
    logr = LogisticRegression()  # 默认 penalty="l2",C=1.0,
    logr.fit(x_train, y_train)

    print(logr.coef_)
    print("准确率:", logr.score(x_test, y_test))
    # [[1.49159213 0.46667743 0.92893529 0.75838182 0.12238663 1.18111883 0.75582525 0.30524351 0.85741416]]
    # 准确率: 0.9590643274853801

    # 计算一下召回率, 这里指定y值的结果, 2代表良性, 4代表恶性
    y_pred = logr.predict(x_test)
    print("召回率:", classification_report(y_test, y_pred, labels=[2, 4], target_names=["良性", "恶性"]))
    # 召回率:            precision    recall  f1-score   support
    #
    #           良性       0.95      0.98      0.96       106
    #           恶性       0.97      0.91      0.94        65
    #
    #     accuracy                           0.95       171
    #    macro avg       0.96      0.94      0.95       171
    # weighted avg       0.95      0.95      0.95       171
    return None


if __name__ == '__main__':
    logistic_regression()
