import os

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

"""
猫狗训练模型
在kaggle网站下载猫狗图片, 然后将前10000张图片作为训练集, 后2500张图片作为测试集, 放入文件夹中
<https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset>
文件架结构: 
 dogs-and-cats
   - train
     - cat
     - dog
   - validation
     - cat
     - dog 
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

# 查看模型类型. 如果想使用gpu那么需要安装gpu版本. pip install tensorflow-gpu
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

# 构建模型
model = tf.keras.models.Sequential([
    # 如果训练慢，可以把数据设置的更小一些
    # 这里的input shape是将数据resize到目标大小
    # 这里是最终得到32个特征图, 使用3x3的核, relu激活函数
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),  # 池化 一般都是2x2

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # 为全连接层准备, 平铺为一个向量
    tf.keras.layers.Flatten(),

    # 创建全连接层
    tf.keras.layers.Dense(512, activation='relu'),
    # 二分类sigmoid就够了
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
# Model: "sequential"
# ______________________________________________________________________
#  Layer (type)                     Output Shape              Param #
# ====================================================================== 32个3x3x3的核
#  conv2d (Conv2D)                  (None, 62, 62, 32)        896   -> 3x3x3 * 32 + 32 = 896
#  max_pooling2d (MaxPooling2D)     (None, 31, 31, 32)        0
#  conv2d_1 (Conv2D)                (None, 29, 29, 64)        18496
#  max_pooling2d_1 (MaxPooling2D)   (None, 14, 14, 64)        0
#  conv2d_2 (Conv2D)                (None, 12, 12, 128)       73856
#  max_pooling2d_2 (MaxPooling2D)   (None, 6, 6, 128)         0
#  flatten (Flatten)                (None, 4608)              0
#  dense (Dense)                    (None, 512)               2359808
#  dense_1 (Dense)                  (None, 1)                 513
# =================================================================
# Total params: 2,453,569
# Trainable params: 2,453,569
# Non-trainable params: 0
# _________________________________________________________________

# 配置训练器
model.compile(loss='binary_crossentropy',  # 二分类损失计算
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['acc'])  # 由于是二分类, 直接使用准确率作为评估标准

# 数据的归一化
#
# 数据预处理:
# - 读进来的数据会被自动转换成tensor(float32)格式，分别准备训练和验证
# - 图像数据归一化（0-1）区间
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 创建生成器, 回一个batch一个batch的读取数据. 目标大小必须是64x64的
# 生成器的原因是不会将数据读入内存
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 文件夹路径
    target_size=(64, 64),  # 指定resize成的大小
    batch_size=20,
    class_mode='binary')  # 如果one-hot就是categorical，二分类用binary就可以
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='binary')
# Found 20002 images belonging to 2 classes.
# Found 4998 images belonging to 2 classes.

# 训练网络模型
# 直接fit也可以，但是通常咱们不能把所有数据全部放入内存，fit_generator相当于一个生成器，动态产生所需的batch数据
# steps_per_epoch相当给定一个停止条件，因为生成器会不断产生batch数据，说白了就是它不知道一个epoch里需要执行多少个step
# 如果2000个训练数据, 如果batch是20, 那么会有100次才能完成epoch. 最好自己计算
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2) # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。

# 此处, 训练集的结果为0.98, 拟合程度很高, 但是测试集中, 拟合程度却很低, 只有0.84, 说明模型过拟合了
# Epoch 201/201
# 100/100 - 7s - loss: 0.0486 - acc: 0.9870 - val_loss: 0.5077 - val_acc: 0.8480 - 7s/epoch - 70ms/step

model.save('dog_cat_model.h5')

## 展示结果
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.subplot(121).plot(epochs, acc, 'bo', label='Training accuracy')
plt.subplot(121).plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.subplot(121).set_title('Training and validation accuracy')

plt.subplot(122).plot(epochs, loss, 'bo', label='Training Loss')
plt.subplot(122).plot(epochs, val_loss, 'b', label='Validation Loss')
plt.subplot(122).set_title('Training and validation loss')

plt.legend()

plt.show()
