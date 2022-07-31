import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def linear_regression():
    """
    线性回归直接预测房屋价格
    :return:
    """
    # 获取数据
    data = fetch_california_housing()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))

    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)

    # 保存训练模型
    joblib.dump(sgd, "./export/test.pkl")

    return None


if __name__ == '__main__':
    linear_regression()
