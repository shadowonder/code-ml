import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

"""
将目标图像数据封装成tfrecord的数据
"""

# 展示数据
image_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\sample\\'
images = glob.glob(image_path + '*.png')

for fname in images:
    image = mpimg.imread(fname)
    f, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    f.subplots_adjust(hspace=.2, wspace=.05)

    ax1.imshow(image)
    ax1.set_title('Image', fontsize=20)
    # plt.show()

image_labels = {
    'dog': 0,
    'kangaroo': 1,
}

# 打开一张图像, 制定一个标签
# 读数据，binary格式
image_string = open('E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\sample\\dog.png', 'rb').read()
label = image_labels['dog']


def _bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Return a float_list form a float/double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Return a int64_list from a bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 创建图像数据的Example, 保存哪些image
def image_example(image_string, label):
    # 通过tfdecode工具 获取image shape
    image_shape = tf.image.decode_png(image_string).shape

    # 储存多个feature, 同时储存原始数据
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# 打印部分信息 打印feature标签
image_example_proto = image_example(image_string, label)
for line in str(image_example_proto).split('\n')[:15]:
    print(line)
print('...')
# features {
#   feature {
#     key: "depth"
#     value {
#       int64_list {
#         value: 3
#       }
#     }
#   }
#   feature {
#     key: "height"
#     value {
#       int64_list {
#         value: 576
#       }
# ...


# 制作 `images.tfrecords`.
image_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\sample\\'
images = glob.glob(image_path + '*.png')  # 获取全部图片的路径
record_file = 'images.tfrecord'
counter = 0

# 打开tfrecord文件io, 写入tfrecord数据
with tf.io.TFRecordWriter(record_file) as writer:
    for fname in images:
        # 对于每一个文件, 打开文件io
        with open(fname, 'rb') as f:
            image_string = f.read()  # 读取文件
            label = image_labels[os.path.basename(fname).replace('.png', '')]  # 读取标签, 删除文件名的ext

            # `tf.Example` 获取features
            tf_example = image_example(image_string, label)

            # 将`tf.example` 写入 TFRecord
            writer.write(tf_example.SerializeToString())

            # logging
            counter += 1
            print('Processed {:d} of {:d} images.'.format(
                counter, len(images)))

# Wrote 2 images to images.tfrecord
print(' Wrote {} images to {}'.format(counter, record_file))

# 加载tfrecord数据
raw_train_dataset = tf.data.TFRecordDataset('images.tfrecord')
# <TFRecordDatasetV2 element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>
print(raw_train_dataset)
# example数据都进行了序列化, 还需要解析一下之前写入的序列化string
#  使用tf.io.parse_single_example(example_proto,feature_description)可以解析一条example

# 解析的格式需要跟之前创建example时一致, 和上面输入的一致
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def parse_tf_example(example_proto):
    # 解析出来, example_proto就是序列化的原始数据, image_feature_description就是解析后的数据结构
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)

    # 预处理
    x_train = tf.image.decode_png(parsed_example['image_raw'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))  # resize一下
    x_train /= 255.

    lebel = parsed_example['label']
    y_train = lebel

    return x_train, y_train  # 返回x和y


# 使用map方法调用parse操作, 将所有的数据进行parse操作
train_dataset = raw_train_dataset.map(parse_tf_example)
# <MapDataset element_spec=(TensorSpec(shape=(416, 416, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>
print(train_dataset)

"""
制作训练集
"""
num_epochs = 10
# 构建一个batch数据,2条, 重复10次
train_ds = train_dataset.shuffle(buffer_size=10000).batch(2).repeat(num_epochs)
print(train_ds)

# 打印一下batch数据
for batch, (x, y) in enumerate(train_ds):
    print(batch, x.shape, y)
    # 0 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 1 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 2 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 3 (2, 416, 416, 3) tf.Tensor([1 0], shape=(2,), dtype=int64)
    # 4 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 5 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 6 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 7 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)
    # 8 (2, 416, 416, 3) tf.Tensor([1 0], shape=(2,), dtype=int64)
    # 9 (2, 416, 416, 3) tf.Tensor([0 1], shape=(2,), dtype=int64)

# 定义一个模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')
])

# 训练规则
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 训练
model.fit(train_ds, epochs=num_epochs)
