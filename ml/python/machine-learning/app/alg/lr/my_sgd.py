import numpy as np
import matplotlib.pyplot as plt


def h(x):
    return w0+w1*x[0]+w2*x[1]


if __name__ == '__main__':
    #  y= 5 + 3*x1 + 8*x2    //w0 = 5  ,w1 = 3, w2 = 8
    rate = 0.001
    x_train = np.array([[2, 2], [3, 6], [5, 2], [4, 5], [6, 2], [7, 4], [8, 5], [1, 2], [10, 3], [9, 7]])
    y_train = np.array([27, 62, 36, 57, 39, 58, 69, 24, 59, 88])
    x_test = np.array([[9,2], [4,7], [9,3], [6,4], [2,7], [8,3]])

    w0 = np.random.normal()
    w1 = np.random.normal()
    w2 = np.random.normal()

    for i in range(10):
        for x, y in zip(x_train, y_train):
            w0 = w0 - rate*(h(x)-y)*1
            w1 = w1 - rate*(h(x)-y)*x[0]
            w2 = w2 - rate*(h(x)-y)*x[1]
        plt.plot([h(xi) for xi in x_test])

    print(w0)
    print(w1)
    print(w2)

    result = [h(xi) for xi in x_train]
    print(result)

    result = [h(xi) for xi in x_test]
    print(result)

    plt.show()
