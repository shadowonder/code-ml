import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# 下载服装数据集
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 输出类别,我们输出的类别
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)  # (60000, 28, 28)
print(len(train_labels))  # 60000
print(test_images.shape)  # (10000, 28, 28)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()  # 添加颜色热度
plt.grid(False)  # 关闭网格
# plt.show()

#### 处理图像
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# 展示图像
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

## 训练模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 将图像进行拉伸, 成为784的feature, 同时定义输入层
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

####评估模型
# 指定测试集的x和测试集的y就可以了, 直接出结果
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)  # Test accuracy: 0.8847000002861023

predictions = model.predict(test_images)
print(predictions.shape)  # (10000, 10) # 10000个结果对应10各类别的概率值
print(predictions[0])  # 获取第一个测试样本的每一个类别概率值, 一般我们只是用最大的一个
# [7.6147963e-08 5.5699688e-11 2.4855829e-10 6.9418427e-09 4.1508761e-09
#  2.9993954e-04 1.7000576e-07 5.8553129e-02 2.8515702e-08 9.4114667e-01]

print(np.argmax(predictions[0]))  # 9  也是最大的样本概率, 第一个测试样本9号类别概率最高


### 可视化结果
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
# plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
# plt.show()

### 模型的保存
model.save('fashion_model.h5')

# 将网络模型输入到文件中
config_json = model.to_json()
with open('config.json', 'w') as json:
    json.write(config_json)
# '{"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}'

# 获取权重参数
weights = model.get_weights()
print(weights)
# 也可以存入文件
model.save_weights("fashion.weights")
