import numpy as np
import tensorflow as tf

x = [[1.]]  # 定义一个1行一列的数组
m = tf.matmul(x, x)  # 得到tf封装的Tensor tf.Tensor([[1.]], shape=(1, 1), dtype=float32)
print(m)  # 由于使用的是tensorflow2.0, 动态图的因素, 可以直接打印架构

x = tf.constant([[1, 9], [3, 6]])
print(x)
# tf.Tensor(
# [[1 9]
#  [3 6]], shape=(2, 2), dtype=int32)

print(x.numpy())  # 也可以转换为numpy直接获取
# [[1 9]
#  [3 6]]

x = tf.cast(x, tf.float32)  # 类型转换
print(x)
# tf.Tensor(
# [[1. 9.]
#  [3. 6.]], shape=(2, 2), dtype=float32)

x_1 = np.ones([2, 2])  # 一个2x2的矩阵
x_2 = tf.multiply(x_1, 2)  # 简单的乘法操作
print(x_2)
# [[2. 2.]
#  [2. 2.]], shape=(2, 2), dtype=float64)
