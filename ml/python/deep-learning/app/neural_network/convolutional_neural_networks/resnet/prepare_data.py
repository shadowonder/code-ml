import pathlib

import tensorflow as tf

from . import config
from .config import image_height, image_width, channels

"""
主要用于预处理以及特征工程

入口函数为generate_datasets()
主要根据输入的路径文件夹获取文件夹下归类好的文件路径, 读取并将所有图像文件读入内存. 其中包含特征工程.
"""


def load_and_preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # 所有的标签名字, 比如dog cat
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
                       all_image_path]

    return all_image_path, all_image_label


def get_dataset(dataset_root_dir):
    # 获取所有路径和标签
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count


# 获取dataset
def generate_datasets():
    train_dataset, train_count = get_dataset(dataset_root_dir=config.train_dir)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=config.valid_dir)
    test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)

    # read the original_dataset in the form of batch
    # batch size 在config文件中定义为8
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
