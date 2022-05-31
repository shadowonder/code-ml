import tensorflow as tf

from .residual_block import build_res_block_2
from ..config import NUM_CLASSES

"""
只要维度不同就需要添加convolution block
"""


class ResNet50(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNet50, self).__init__()

        # 创建输入, 同时创建64个7x7的卷积核, 得到64个特征图
        self.pre1 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding='same')
        self.pre2 = tf.keras.layers.BatchNormalization()
        self.pre3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        # 对64个特征图执行3x3的max pooling池化. 此时我们还是只有64个特征, 从而成为下一层block的输入
        self.pre4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2)

        """
        不同的残差区块,没一个区块的目的是实现一个卷积区块和一个short cut. 
        这里的输入层将是一个64维的卷积层但是最终输出将会是一个512维的特征, 因此需要对数据进行叠加. 
        数据扩充的方法就是使用1x1的卷积512个filter得到目标特征数量
        """
        self.layer1 = build_res_block_2(filter_num=64,
                                        blocks=3)
        self.layer2 = build_res_block_2(filter_num=128,
                                        blocks=4,
                                        stride=2)
        self.layer3 = build_res_block_2(filter_num=256,
                                        blocks=6,
                                        stride=2)
        self.layer4 = build_res_block_2(filter_num=512,
                                        blocks=3,
                                        stride=2)

        # 最终的层, 包括全连接的层
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.relu)
        self.drop_out = tf.keras.layers.Dropout(rate=0.5)
        self.fc2 = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        pre1 = self.pre1(inputs)
        pre2 = self.pre2(pre1, training=training)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        l1 = self.layer1(pre4, training=training)
        l2 = self.layer2(l1, training=training)
        l3 = self.layer3(l2, training=training)
        l4 = self.layer4(l3, training=training)
        avgpool = self.avgpool(l4)
        fc1 = self.fc1(avgpool)
        drop = self.drop_out(fc1)
        out = self.fc2(drop)

        return out
