from __future__ import absolute_import, division, print_function

import math

import tensorflow as tf

from . import config
from .models import resnet50, resnet101, resnet152, resnet34
from .prepare_data import generate_datasets

"""
fork自https://github.com/calmisential/TensorFlow2.0_ResNet
"""


def get_model():
    model = resnet50()
    if config.model == "resnet34":
        model = resnet34()
    if config.model == "resnet101":
        model = resnet101()
    if config.model == "resnet152":
        model = resnet152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()

    tf.keras.utils.plot_model(model, to_file='model.png')
    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    # 通过prepare_data.py文件中的generate_datasets()函数获取定义文件夹下的全部文件并归类输出
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    # 在models文件夹中的resnet.py文件中定义了ResNetType的I型和II型类. 通过get_model()函数, 返回指定的类
    # I型和II型继承了tensorflow的Model类, 所以是tensorflow的计算模型, 可以直接使用tensorflow api
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta()

    # 损失函数
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


    # 装饰器, 将funciton放入图中, 可以让执行训练更快
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)  # 基于标签和预测计算损失
        # 如果有多个损失函数可以使用tape.gradient([loss1,loss2], model.trainable_variables)
        gradients = tape.gradient(loss, model.trainable_variables)
        # 更新当前的梯度
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)


    # start training
    # 训练集和验证的操作
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(
                                                                                         train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  config.EPOCHS,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

    model.save_weights(filepath=config.save_model_dir, save_format='tf')
