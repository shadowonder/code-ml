import gzip
from pathlib import Path
import pickle

import numpy as np
import requests
import tensorflow as tf
from tensorflow.python.keras import layers

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
print(PATH)  # data\mnist

PATH.mkdir(parents=True, exist_ok=True)  # 创建文件夹
URL = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/'
FILENAME = 'mnist.pkl.gz'

if not (PATH / FILENAME).exists():
    print("downloading...")
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    print("unzipping...")
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

"""
数据集, 以及常用函数
"""
input_data = np.arange(16)
print(input_data)  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

# 直接将ndarray类型的数据传入tensorflow, 得到的flow数据是可以遍历的
dataset = tf.data.Dataset.from_tensor_slices(input_data)
for data in dataset:
    print(data)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # tf.Tensor(2, shape=(), dtype=int32)
    # ...

# repeat操作
# 将dataset中的数据repeat一份, 同样序列
dataset = tf.data.Dataset.from_tensor_slices(input_data)
dataset = dataset.repeat(2)
for data in dataset:
    print(data)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # ...
    # tf.Tensor(15, shape=(), dtype=int32)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # ...
    # tf.Tensor(15, shape=(), dtype=int32)

# batch 操作, 批量操作, 将数据分批打包
dataset = tf.data.Dataset.from_tensor_slices(input_data)
dataset = dataset.batch(4)  # 每4个组成一个batch
for data in dataset:
    print(data)
    # tf.Tensor([0 1 2 3], shape=(4,), dtype=int32)
    # tf.Tensor([4 5 6 7], shape=(4,), dtype=int32)
    # tf.Tensor([ 8  9 10 11], shape=(4,), dtype=int32)
    # tf.Tensor([12 13 14 15], shape=(4,), dtype=int32)

# shuffle操作, 打乱顺序
# buffer_size 缓存区, 随机构建的时候使用缓存区进行抽取, 然后从缓存区中随机抽取构建序列. 类似于随机滑动窗口.
# 比如buffer_size为10, 那么就将1-10放入缓存区, 然后随机抽取, 抽取一个以后将11放入随机抽取.
# 此时, 11绝对不会出现在第一个位置, 而第一个位置也必然只能是1-10中的一个
# 因此, 如果buffer_size为1的时候就是不进行乱序操作, 而buffer_size为数据长度相同时就是全局随机排列.
dataset = tf.data.Dataset.from_tensor_slices(input_data).shuffle(buffer_size=10).batch(4)
for data in dataset:
    print(data)
    # tf.Tensor([ 1  0 11  6], shape=(4,), dtype=int32)
    # tf.Tensor([13  5 15  2], shape=(4,), dtype=int32)
    # tf.Tensor([ 8  7  3 14], shape=(4,), dtype=int32)
    # tf.Tensor([10  9  4 12], shape=(4,), dtype=int32)

# 基于数据集的重新训练
train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).repeat()
valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32).repeat()
model.fit(train, epochs=5, steps_per_epoch=100, validation_data=valid, validation_steps=100)
