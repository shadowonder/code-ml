import matplotlib.pyplot as plt
import numpy as np


def h(x):
    return w0 * 1 + w1 * x[0] + w2 * x[1]


"""
根据梯度下降
"""
if __name__ == '__main__':
    #  y=3 + 2 * (x1) + (x2)
    rate = 0.001
    # 通过x1,x2获得y, 来计算w0,w1,w2的值
    x_train = np.array([[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]])
    y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
    x_test = np.array([[1, 4], [2, 2], [2, 5], [5, 3], [1, 5], [4, 1]])
    w0 = np.random.normal()
    w1 = np.random.normal()
    w2 = np.random.normal()
    for i in range(7000):
        for x, y in zip(x_train, y_train):
            w0 = w0 - rate * (h(x) - y) * 1
            w1 = w1 - rate * (h(x) - y) * x[0]
            w2 = w2 - rate * (h(x) - y) * x[1]
        plt.plot([h(xi) for xi in x_test])
        # 当最终结果计算完毕, 导数将趋于0
        print("w0导数 = %f ,w1导数 = %f ,w2导数 = %f " % (
            rate * (h(x) - y) * 1, rate * (h(x) - y) * x[0], rate * (h(x) - y) * x[1]))

    print(w0)
    print(w1)
    print(w2)

    result = [h(xi) for xi in x_train]
    print(result)

    result = [h(xi) for xi in x_test]
    # [9,9,12,16,10,12]
    print(result)

    plt.show()
