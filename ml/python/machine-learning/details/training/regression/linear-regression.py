from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def linear_regression():
    """
    线性回归直接预测房屋价格
    :return:
    """
    # 获取数据
    data = fetch_california_housing()
    print(data.feature_names)
    # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    print(data.data)
    print(data.target)  # [4.526 3.585 3.521 ... 0.923 0.847 0.894]

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    # 标准化
    # 特征值都必须进行标准化处理, 实例化两个标准化api
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 需要注意的是在0.19版本以后, 标准化只能作用在多维数组上, 不能作用在一维数组上
    std_y = StandardScaler()
    # y_train = std_y.fit_transform(y_train)
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    # y_test = std_y.transform(y_test)
    # y_test = std_y.transform(y_test.reshape(-1, 1))

    # 预测
    # ==========================正规方程求解方式===========================
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("模型参数: ", lr.coef_)  # 获取全部的w值, 也就是回归系数

    # 测试集进行预测
    y_predict = lr.predict(x_test)
    # 使用反转标准化
    print("测试集中每个样本的预测结果:", std_y.inverse_transform(y_predict))
    print("均方误差", mean_squared_error(y_test, std_y.inverse_transform(y_predict)))
    # 均方误差 0.5281526459634236

    # ===========================梯度下降=================================
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("模型参数: ", sgd.coef_)
    y_sgd_predict = sgd.predict(x_test).reshape(-1, 1)  # sgd 返回一维数组
    print("测试集中每个样本的预测结果:", y_sgd_predict)
    print("均方误差", mean_squared_error(y_test, std_y.inverse_transform(y_sgd_predict)))
    # 均方误差 0.5325275391867944

    # ===========================岭回归==================================
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    print("模型参数: ", ridge.coef_)
    y_r_predict = ridge.predict(x_test)
    print("测试集中每个样本的预测结果:", y_r_predict)
    print("均方误差", mean_squared_error(y_test, std_y.inverse_transform(y_r_predict)))

    # 网格搜索与交叉验证f
    print("正在进行网格参数调优...")
    params = {"alpha": [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.5, 0.7, 1, 10, 50, 100, 500]}
    gc = GridSearchCV(ridge, param_grid=params, cv=2)
    gc.fit(x_train, y_train)
    print("预测准确率")
    print(gc.score(x_test, y_test))
    print("选择的参数模型:")
    print(gc.best_params_)

    return None


if __name__ == '__main__':
    linear_regression()
