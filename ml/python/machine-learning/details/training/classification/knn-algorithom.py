import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knn_alg():
    """
    k近邻算法
    k-nearest neighbors algorithm
    """
    # 读取数据
    data = pd.read_csv("E:\\Workspace\\ml\\machine-learning-python\\data\\FBlocation\\train.csv")

    # 缩小数据, 通过query查询数据
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 对时间进行处理
    time_value = pd.to_datetime(data['time'], unit='s')

    # 把日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)
    print(time_value)
    # DatetimeIndex(['1970-01-01 18:09:40', '1970-01-10 02:11:10',
    #                '1970-01-05 15:08:02', '1970-01-06 23:03:03',
    #                '1970-01-09 11:26:50', '1970-01-02 16:25:07',
    #                ...

    # 添加feature
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    # data.loc[:, 'day'] = time_value.day
    # data.loc[:, 'hour'] = time_value.hour
    # data.loc[:, 'weekday'] = time_value.weekday
    print(data)
    #             row_id       x       y  accuracy    time    place_id  day  hour  weekday
    # 600            600  1.2214  2.7023        17   65380  6683426742    1    18        3
    # 957            957  1.1832  2.6891        58  785470  6683426742   10     2        5

    # 从data中删除时间特征, 1表示列, 0表示行
    data = data.drop(['time'], axis=1)
    print(data)
    #             row_id       x       y  accuracy    place_id  day  hour  weekday
    # 600            600  1.2214  2.7023        17  6683426742    1    18        3
    # 957            957  1.1832  2.6891        58  6683426742   10     2        5

    # 把签到数量少于n个的位置删除
    place_count = data.groupby('place_id').count()

    # place_count.row_id就成了count的返回值了, 然后把大于3的index保留住, 也就是过滤掉了小于3的id, 也就是count
    # 然后reset_index()就是把index变为一个列,此处就是place_id,也就是刚刚的groupby的名称设置为一个列
    tf = place_count[place_count.row_id > 3].reset_index()

    # data中的place_id是否在tf.place_id中也就是在data中删除小于3的特征值
    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据当中的目标值和特征值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    x = x.drop(['row_id'], axis=1)

    print(x)

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程(standarlize)
    std = StandardScaler()
    # 对训练集和测试集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 算法, 计算最近的5个点
    # knn = KNeighborsClassifier(n_neighbors=5)
    #
    # knn.fit(x_train, y_train)
    #
    # # 得出预测结果
    # y_predict = knn.predict(x_test)
    #
    # print("预测的目标签到位置: ", y_predict)
    #
    # # 准确率
    # print("预测的准确率", knn.score(x_test, y_test))

    # 使用网格搜索, 需要注意的是不需要给参数否则参数会固定
    knn = KNeighborsClassifier()

    # 构造参数的值进行搜索
    param = {"n_neighbors": [3, 5, 10]}

    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    # 预测准确率
    print("在测试集上的准确率:", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果:", gc.best_score_)
    print("最好的模型(参数):", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果", gc.cv_results_)
    # 在测试集上的准确率: 0.4739952718676123
    # 在交叉验证当中最好的结果: 0.44774590163934425
    # 最好的模型(参数): KNeighborsClassifier(n_neighbors=10)
    # 每个超参数每次交叉验证的结果 ...

    return None


if __name__ == '__main__':
    knn_alg()
