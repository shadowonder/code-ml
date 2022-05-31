from keras import layers
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow import keras

"""
卷积神经网络文本分类
使用的是影评的数据进行训练
"""
num_features = 3000
sequence_length = 300  # 文章的长度, 不满300得用0填充
embedding_dimension = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)
print(x_train.shape)  # (25000,)
print(x_test.shape)  # (25000,)
print(y_train.shape)  # (25000,)
print(y_test.shape)  # (25000,)

# 填充操作
x_train = pad_sequences(x_train, maxlen=sequence_length)
x_test = pad_sequences(x_test, maxlen=sequence_length)
print(x_train.shape)  # (25000, 300)
print(x_test.shape)  # (25000, 300)
print(y_train.shape)  # (25000,)
print(y_test.shape)  # (25000,)

# 多种卷积核，相当于单词数, 3个词,四个词,五个词的卷积核
filter_sizes = [3, 4, 5]


# 卷积
def convolution():
    inn = layers.Input(shape=(sequence_length, embedding_dimension, 1))  # 3维的, 获取上一层的输出作为这里的输入
    cnns = []
    for size in filter_sizes:  # 遍历卷积核
        # 创建卷积核, 我们针对每一个卷积创建64个特征图, 也就是64个卷积核
        conv = layers.Conv2D(filters=64, kernel_size=(size, embedding_dimension),
                             strides=1, padding='valid', activation='relu')(inn)
        # 需要将多种卷积后的特征图池化成一个特征
        # 由于我们定义的步长为1, 同时没有padding, 所以最终结果将是一个("总长度" - 卷积核大小 - 1)的一维向量
        # 然后使用2位pooling得到一个单一的值
        # 由于我们针对每一个 100 * size的卷积核, 我们使用64个不同的特征图. 因此我们最终得到64个一维向量. 然后通过pooling得到64个值
        pool = layers.MaxPool2D(pool_size=(sequence_length - size + 1, 1), padding='valid')(conv)
        cnns.append(pool)  # 将每一个卷积64个pooling后的值append到list中
    # 将得到的特征拼接在一起, 形成一个3*64=192个输出特征
    outt = layers.concatenate(cnns)

    model = keras.Model(inputs=inn, outputs=outt)
    return model


def cnn_mulfilter():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,
                         input_length=sequence_length),  # 将所有的词都映射为向量
        layers.Reshape((sequence_length, embedding_dimension, 1)),  # 变为3维的操作, 用1来填充
        convolution(),  # 卷积
        layers.Flatten(),
        layers.Dense(10, activation='relu'),  # 全连接
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


model = cnn_mulfilter()
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 300, 100)          300000   # 300个词100个维度
#  reshape (Reshape)           (None, 300, 100, 1)       0
#  model (Functional)          (None, 1, 1, 192)         76992  # 自定义模型卷积pooling后得到 3*64的特征
#  flatten (Flatten)           (None, 192)               0
#  dense (Dense)               (None, 10)                1930
#  dropout (Dropout)           (None, 10)                0
#  dense_1 (Dense)             (None, 1)                 11
# =================================================================
# Total params: 378,933
# Trainable params: 378,933
# Non-trainable params: 0
# _________________________________________________________________

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
# Epoch 1/5
# 352/352 [==============================] - 33s 93ms/step - loss: 0.5019 - accuracy: 0.7399 - val_loss: 0.3359 - val_accuracy: 0.8568
# ...

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()
