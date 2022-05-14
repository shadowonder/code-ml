# 处理时间数据
import gzip
from pathlib import Path
import pickle

from matplotlib import pyplot
import requests
import tensorflow as tf
# 不同版本的引用会不一样, 其他版本使用tensorflow.keras
# 也可以直接使用tf.keras.layers
from tensorflow.python.keras import layers

"""
分类任务
"""
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
print(PATH)  # data\mnist

PATH.mkdir(parents=True, exist_ok=True)  # 创建文件夹
URL = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/'
FILENAME = 'mnist.pkl.gz'

# url下载到本地
if not (PATH / FILENAME).exists():
    print("downloading...")
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# 解压缩, 然后抽取train和test
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    print("unzipping...")
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)))
pyplot.show()
print(x_train.shape)  # 展示一波
print(y_train[0])  # 5, 这里的y不是onehot类型,属于单一输出

############################ 模型构建 ############################
# 数据是一个28*28的图片, 得到的784个pixel
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 分为10类

# 由于输入的结果值y属于属性方法, 而不是onehot类型, 我们可以使用不同的损失函数
# 比如 CategoricalCrossentropy 需要一个onehot形式 (we expect labels to be provided in a one_hot representation.)
# 损失函数可能会变得非常离谱, 所以如果损失值出现问题, 检查损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])  # 展示准确率, 准确率的计算metric
model.fit(x_train, y_train,
          validation_split=0.25, epochs=5, batch_size=64,
          validation_data=(x_valid, y_valid))
