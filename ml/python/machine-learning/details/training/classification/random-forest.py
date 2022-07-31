import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV


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
    print(dict.get_feature_names_out())
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

    # 使用随机森林
    # (n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None)
    rfc = RandomForestClassifier()  # 默认数据

    # 网格搜索与交叉验证
    print("正在进行网格参数调优...");
    params = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rfc, param_grid=params, cv=2)  # 网格交叉验证, 配置交叉验证为2

    gc.fit(x_train, y_train)

    print("预测准确率")
    print(gc.score(x_test, y_test))
    print("选择的参数模型:")
    print(gc.best_params_)

    # 预测准确率
    # 0.8297872340425532
    # 选择的参数模型:
    # {'max_depth': 5, 'n_estimators': 300}

    return None


if __name__ == '__main__':
    discussion_tree_alg()
