# 决策树
# 决策树存在一个升级版叫做随机森林
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def discussion_tree_alg():
    """
    泰坦尼克号的数据分类
    https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic.txt

    1: 存活
    0: 死亡
    """
    print("决策树选择过程")
    # 读取数据
    data = pd.read_csv("E:\\Workspace\\ml\\machine-learning-python\\data\\titanic.txt")
    print(data)

    # 处理数据, 找出特征值和目标值
    x = data[['pclass', 'age', 'sex']]  # 特征值的列
    print(x)
    y = data['survived']  # 结果集

    # 处理缺失值 inplace表示替换. 把平均值填入age中
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集, 训练集, 测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程处理
    dict = DictVectorizer(sparse=False)
    # to_dict() 就是将样本转换为字典类型, orient="records"表示每一行为一个字典
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.fit_transform(x_test.to_dict(orient="records"))
    print(dict.get_feature_names())
    print(x_train)
    print(x_test)
    # ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
    # [[31.19418104  0.          0.          1.          1.          0.        ]
    #  [39.          1.          0.          0.          1.          0.        ]
    #  [ 1.          0.          1.          0.          1.          0.        ]
    #  ...
    #  [18.          0.          1.          0.          1.          0.        ]
    #  [45.          0.          1.          0.          1.          0.        ]
    #  [ 9.          0.          0.          1.          1.          0.        ]]

    # 用决策树进行预测
    dec = DecisionTreeClassifier()
    # dec = DecisionTreeClassifier(max_depth=5)  # 深度为5
    dec.fit(x_train, y_train)

    print("预测的准确率:")
    print(dec.score(x_test, y_test))

    export_graphviz(dec, out_file="./tree.dot",
                    feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])
    return None


if __name__ == '__main__':
    discussion_tree_alg()
