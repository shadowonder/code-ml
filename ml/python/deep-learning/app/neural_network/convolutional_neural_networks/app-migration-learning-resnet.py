import os
import warnings

from keras import Model, layers
from keras.applications.resnet import ResNet101
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
import tensorflow as tf

"""
迁移学习
"""
# 数据所在文件夹
base_dir = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\dogs-and-cats'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# 训练集
train_cats_dir = os.path.join(train_dir, 'cat')
train_dogs_dir = os.path.join(train_dir, 'dog')
# 验证集
validation_cats_dir = os.path.join(validation_dir, 'cat')
validation_dogs_dir = os.path.join(validation_dir, 'dog')

# 导入模型
# 模型都存放于tf.keras.applications.resnet中 , 当前版本也可以直接调用keras.applications.resnet
# 这里是下载数据的全部参数, 包括所有的数据参数集
# 下载好的数据会保存在C:\Users\<用户名>\.keras\models文件夹中, 第一次下载的时候会展示一个网址,可以手动下载.
pre_trained_model = ResNet101(
    input_shape=(75, 75, 3),  # 输入大小, 在网页要求中,输入大小的长宽必须高于32. 当然不同的网络要求是不一样的
    include_top=False,  # 不要最后的全连接层, 这里说的是要不要提取别人模型的FC, 一般我们都是自己训练全连接
    weights='imagenet'  # 使用什么样的权重, imagenet就是获得冠军的权重
)

# 可以选择训练哪些层, 这里循环所有的层, 然后不更新所有的层
for layer in pre_trained_model.layers:
    layer.trainable = False


## callback的作用
# 相当于一个监视器，在训练过程中可以设置一些自定义项，比如提前停止，改变学习率等
# callbacks = [
#   如果连续两个epoch还没降低就停止：
#   tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#   可以动态改变学习率：
#   tf.keras.callbacks.LearningRateScheduler
#   保存模型：
#   tf.keras.callbacks.ModelCheckpoint
#   自定义方法：
#   tf.keras.callbacks.Callback
# ]

# 自定义方法, 集成tf.keras.callbacks.Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# 为全连接层准备
# 可以直接获取我们所需要的哪一层 get_layer = pre_trained_model.get_layer(<层的名字>)
# 然后将那一层放入我们的输入或者中间层等等
x = layers.Flatten()(pre_trained_model.output)  # 将resnet层结果的输出作为放入
# 加入全连接层，这个需要重头训练的
x = layers.Dense(1024, activation='relu')(x)  # 将x作为输入放入全连接层
x = layers.Dropout(0.2)(x)
# 输出层
x = layers.Dense(1, activation='sigmoid')(x)
# 构建模型序列
model = Model(pre_trained_model.input, x)

model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(75, 75))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(75, 75))

# 训练模型
# 加入Callback()模块
callbacks = myCallback()
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=50,
    epochs=20,
    validation_steps=25,
    verbose=2,
    callbacks=[callbacks])

# 绘图展示

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.subplot(121).plot(epochs, acc, 'b', label='Training accuracy')
plt.subplot(121).plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.subplot(121).set_title('Training and validation accuracy')
plt.subplot(121).legend()

plt.subplot(122).plot(epochs, loss, 'b', label='Training Loss')
plt.subplot(122).plot(epochs, val_loss, 'r', label='Validation Loss')
plt.subplot(122).set_title('Training and validation loss')
plt.subplot(122).legend()

plt.show()
