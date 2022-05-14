# 处理时间数据
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow import keras
import tensorflow as tf
# 不同版本的引用会不一样, 其他版本使用tensorflow.keras
# 也可以直接使用tf.keras.layers
from tensorflow.python.keras import layers

"""
搭建神经网络进行气温预测
"""
# 导入数据, actual就是数据
#
#    year  month  day  week  temp_2  temp_1  average  actual  friend
# 0  2016      1    1   Fri      45      44       44      45      24
# 1  2016      1    2   Sat      46      45       44      45      24
# 2  2016      1    3   Sun      47      46       44      45      24
# 3  2016      1    4   Mon      48      47       44      45      24
# 4  2016      1    5  Tues      49      48       44      45      24
features = pd.read_csv('E:\\Workspace\\ml\\code-ml\\ml\\python\\csv\\temps.csv')
print(features.head())

years = features['year']
month = features['month']
day = features['day']
# zip方法: 将多个list压制为元祖: a=[a,b,c]; b=[1,2,3]; zip(a,b)=[(a,1),(b,2),(c,3)]
# 通过for循环list, 然后返回值构建新的list, 返回的元祖赋值到(year,month,day)中
# 通过string创建返回值"2016-1-2"等等的列表
dates = [str(int(years)) + '-' + str(int(month)) + '-' + str(int(day)) for years, month, day in zip(years, month, day)];
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

################################# 数据可视化展示 #############################

# matplotlib 也提供了几种我们可以直接来用的内建样式
# 导入 matplotlib.style 模块 - 探索 matplotlib.style.available 的内容，里面包括了所有可用的内建样式
# 这些样式可以帮我们改变背景颜色, 改变网格, 消除毛刺等等. 这里我们制定了fivethirtyeight的样式
plt.style.use('fivethirtyeight')

# 设置布局, 获取四格图像的subplot, 2x2的图像矩阵
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''), ax1.set_ylabel('Temperature'), ax1.set_title('Max Temp')

# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''), ax2.set_ylabel('Temperature'), ax2.set_title('Previous Max Temp')

# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'), ax3.set_ylabel('Temperature'), ax3.set_title('Two Days Prior Max Temp')

# 朋友预测
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'), ax4.set_ylabel('Temperature'), ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
# plt.show() # 展示一下

################################### 建模 ###############################
# 1. 特征工程
# 将数据进行特征修改, 将weekday改为数字. 有多种处理方式, 这里使用的是独热处理
features = pd.get_dummies(features)
print(features.head(5))
#    year  month  day  temp_2  temp_1  average  actual  friend  week_Fri  week_Mon  week_Sat  week_Sun  week_Thurs  week_Tues  week_Wed
# 0  2016      1    1      45      45     45.6      45      29         1         0         0         0           0          0         0
# 1  2016      1    2      44      45     45.7      44      61         0         0         1         0           0          0         0
# 2  2016      1    3      45      44     45.8      41      56         0         0         0         1           0          0         0
# 3  2016      1    4      44      41     45.9      40      53         0         1         0         0           0          0         0
# 4  2016      1    5      41      40     46.0      44      41         0         0         0         0           0          1         0

# 处理x和y
labels = np.array(features['actual'])  # y值
features = features.drop('actual', axis=1)  # 删除actual column, drop()方法默认删除行, axis表示这里删除的是列
# 获取column的名称单独保存到另一个参数中.
# columns参数获取全部的column名称, 同样可以使用index和values获取需要的数据, index是行坐标, values是全部数据
feature_list = list(features.columns)
features = np.array(features)  # 将feature的DataFrame类型转换为ndArray类型

# 对数据进行预处理, 还是之前sklearn的standardscaler
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features)
# [[ 0.         -1.5678393  -1.65682171 ... -0.40482045 -0.41913682 -0.40482045]
#  [ 0.         -1.5678393  -1.54267126 ... -0.40482045 -0.41913682 -0.40482045]
#  [ 0.         -1.5678393  -1.4285208  ... -0.40482045 -0.41913682 -0.40482045]
#  ...

# 2. 构建模型
# 这里我们使用的是dense mode全连接层, 其中也包含卷积层cropping等等
# <https://tensorflow.google.cn/api_docs/python/tf/keras>
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )
model = tf.keras.Sequential()  # 创建模型
model.add(layers.Dense(16))  # 第一层16个神经元, 也就是16个特征
model.add(layers.Dense(32))  # 第二层32个神经元
model.add(layers.Dense(1))  # 输出单元1

# 初始化网络模型
# model.compile(optimizer='sgd', loss='mse')
# optimizer 优化迭代器. 损失函数的迭代器. 这里可以使用上面的简写, adam也是常用的迭代器.
# loss 损失函数, mse是非常常见的损失函数, 不同损失函数对最终结果影响很大.
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')

# 训练, tensorflow1.0中需要创建session等等, 在tensorflow2.0中只需要基础属性就可以了
# validation_split 就是测试集分解
# epochs 运算迭代的次数, 这里制定了迭代10次
# batch_size 每一次优化器迭代多少个样本, 这里选择的是64个样本, 越大越好64/128/256
model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)
# Epoch 10/10
# 5/5 [==============================] - 0s 6ms/step - loss: 50.4199 - val_loss: 724.1822
# 可以看到在训练集中的损失值为724.182, 但是训练集的损失为50.4199, 说明模型出现了过拟合的状态.

model.summary()
# Model: "sequential"
# -----------------------------------------------------------------
#  Layer (type)                Output Shape              Param #
# =================================================================
#  module_wrapper (ModuleWrapp  (None, 16)               240  er)  一共14个特征, 第一层有16个神经元, 总共16x14=224, 再加上偏置函数16为240
#  module_wrapper_1 (ModuleWra  (None, 32)               544  pper) 32*16=512, 再加上32个偏置函数总共544
#  module_wrapper_2 (ModuleWra  (None, 1)                33  pper) 32 * 1 + 1 = 33
# =================================================================
# Total params: 817
# Trainable params: 817
# Non-trainable params: 0


# 3. 重新构建模型, 尝试不同的结果
# 使用random_normal初始化函数, 高斯分布
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal'))
model.add(layers.Dense(32, kernel_initializer='random_normal'))
model.add(layers.Dense(1, kernel_initializer='random_normal'))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)
# 好一丢丢, 但是复杂的数据可能会出现不同的结果
# Epoch 10/10
# 5/5 [==============================] - 0s 6ms/step - loss: 53.7003 - val_loss: 692.4907

# 使用正则化修正, L2的正则化, lambda的值设置为0.03
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=100, batch_size=64)
# 正则化会非常稳定
# Epoch 10/10
# 5/5 [==============================] - 0s 6ms/step - loss: 37.9034 - val_loss: 583.6664
# Epoch 100/100
# 5/5 [==============================] - 0s 6ms/step - loss: 52.9017 - val_loss: 31.6005

##########################################结果预测####################################
# 放入测试集
predict = model.predict(input_features)  # 这里不应该把预测结果直接放入, 因为测试集会完美符合, 但是, 时间有限只能上车了
print(predict.shape)
# 转换日期
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, month, day)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()

# 图名
plt.xlabel('Date'), plt.ylabel('Maximum Temperature (F)'), plt.title('Actual and Predicted Values')
plt.show()
