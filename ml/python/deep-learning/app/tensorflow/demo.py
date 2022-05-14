import tensorflow as tf

"""
1.0 的处理方式, 在tensorflow 1.0 的数据中, 可以使用session进行数据获取以及处理
"""

# tf.compat.v1.disable_eager_execution()  # windows的bug,会进行主动搜寻的处理, 这里关掉
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭tensorflow的log, 调整log等级为2
#
# a = tf.constant(5.0)  # tf.Tensor(6.0, shape=(), dtype=float32)
# b = tf.constant(6.0)
#
# print(b)
#
# sum1 = tf.add(a, b)  # tf.Tensor(11.0, shape=(), dtype=float32)
# print(sum1)
#
# with tf.compat.v1.Session() as sess:
#     print(sess.run(sum1))  # 读取原始数据

"""
2.0 处理方式.
"""
# 定义两个2×2的常量矩阵
y = tf.constant(1.0)  # 如果需要原始数据的话可以使用v1进行获取
tf.compat.v1.print(y)

x = tf.constant([[1, 9], [3, 6]])
print(x.numpy())

print(tf.autograph.to_graph())
