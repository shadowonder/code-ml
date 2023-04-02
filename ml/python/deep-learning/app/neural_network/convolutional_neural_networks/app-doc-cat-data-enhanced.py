import os
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

"""
变换操作后继续计算
"""
# 数据所在文件夹
base_dir = "E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\dogs-and-cats"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
# 训练集
train_cats_dir = os.path.join(train_dir, "cat")
train_dogs_dir = os.path.join(train_dir, "dog")
# 验证集
validation_cats_dir = os.path.join(validation_dir, "cat")
validation_dogs_dir = os.path.join(validation_dir, "dog")

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Dropout(0.5), 防止过拟合 随机部分神经元权重为0
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 全连接
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy", # 多分类的情况下可以使用 'categorical_crossentropy'
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=["acc"], # 多分类也是可以使用accuracy的
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=20, class_mode="binary"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(64, 64), batch_size=20, class_mode="binary"
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2,
)
