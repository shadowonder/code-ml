import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu = tf.nn.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2, training=training)

        # 将两个路径相加
        output = tf.nn.relu(tf.keras.layers.add([identity, bn2]))
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, with_downsample=True):
        super(BottleNeck, self).__init__()

        """
        正常的卷积层
        """
        self.with_downsample = with_downsample
        # 64的1x1的卷积层
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 64的3x3的卷积层
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 256的1x1的卷积层, 输出256个特征
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        """
        shortcut路径
        原始Conv2d的卷积256个1x1的卷积核, 得到256个特征, 也是原始特征的翻倍
        """
        # 新建Sequential
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)  # 获取映射, 输出一个256特征的下采样卷积结果

        # 执行定义的三个卷积
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu1 = tf.nn.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2, training=training)
        relu2 = tf.nn.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3, training=training)

        # 将映射层和卷积层进行相加
        if self.with_downsample == True:
            output = tf.nn.relu(tf.keras.layers.add([identity, bn3]))
        else:
            output = tf.nn.relu(tf.keras.layers.add([inputs, bn3]))

        return output


def build_res_block_1(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def build_res_block_2(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()  # 创建层
    # 当前层加上目标层结构
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1, with_downsample=False))

    return res_block
