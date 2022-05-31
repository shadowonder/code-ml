import numpy as np
import tensorflow as tf

"""
tfrecords序列化
"""


## 对于不同的类型构建数据
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


# 不同数据的函数进行转换
# tf.train.BytesList
print(_bytes_feature(b'test_string'))
print(_bytes_feature('test_string'.encode('utf8')))

# tf.train.FloatList
print(_float_feature(np.exp(1)))  # value: 2.7182817459106445

# tf.train.Int64List
print(_int64_feature(True))  # int64_list {value: 1}
print(_int64_feature(1))  # int64_list {value: 1}


# tfrecord的制作方法
def serialize_example(feature0, feature1, feature2, feature3):
    """
    创建tf.Example
    """
    # 转换成相应类型
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    # 使用tf.train.Example来创建, 将feature作为参数传入
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    # SerializeToString方法返回转换为二进制的字符串
    return example_proto.SerializeToString()


# 数据量, 定义一个数据样本, 10000
n_observations = int(1e4)

# Boolean feature, 随机创建10000个 False 或者 true, 作为feature0
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, 创建10000个 0-4的整形数据
feature1 = np.random.randint(0, 5, n_observations)

# String feature 然后将0-4的整形数据转换为字符串
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature
feature3 = np.random.randn(n_observations)

filename = 'tfrecord-1'

# 写入tfrecord-1中, 打开文件io, 然后写入数据
with tf.io.TFRecordWriter(filename) as writer:
    # 写入循环10000次的4个feature传入序列化方法, 序列化方法会将4个特征序列化为二进制写入文件
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

# 文件中含有的就是二进制的文件
filenames = [filename]

# 读取, 读取出来的时候就是处理好了的数据样本
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)  # <TFRecordDatasetV2 element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>
