import glob
import os

from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

"""
将图像进行操作, 变为多个图像
"""


# 展示结果
def print_result(path):
    names = glob.glob(path)
    fig = plt.figure(figsize=(12, 16))
    for i in range(3):
        img = Image.open(names[i])
        sub_img = fig.add_subplot(131 + i)
        sub_img.imshow(img)
    plt.show()


# 删除之前输出文件剩余的文件
def remove_files_in_folder(p):
    del_list = os.listdir(p)
    for f in del_list:
        file_path = os.path.join(p, f)
        if os.path.isfile(file_path):
            os.remove(file_path)


img_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\img\\superman\\*'
in_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\img\\'
out_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\output\\'
name_list = glob.glob(img_path)
# print_result(img_path) # 展示一下原始数据


remove_files_in_folder(out_path + 'resize')
### 指定转换后所有图像都变为相同大小
# - in_path 源路径
# - batch_size 批数量
# - shuffle 是否随机化
# - save_to_dir 存储路径位置
# - save_prefix 存储文件增加前缀
# - target_size 转换后的统一大小
datagen = image.ImageDataGenerator()
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False,
                                       save_to_dir=out_path + 'resize',
                                       save_prefix='gen', target_size=(224, 224))

# 生成操作
for i in range(3):
    gen_data.next()
# print_result(out_path + 'resize\\*')

### 角度变换
remove_files_in_folder(out_path + 'rotation_range')
# 图像旋转45度的生成器
datagen = image.ImageDataGenerator(rotation_range=45)
# 获取可迭代数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
# 数据转换
datagen.fit(np_data)  # 将resize的图像放入生成器中
# 构建输入生成器
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False,
                                       save_to_dir=out_path + 'rotation_range', save_prefix='gen',
                                       target_size=(224, 224))
# 生成角度旋转后的数据
for i in range(3):
    gen_data.next()
# print_result(out_path + 'rotation_range\\*')

"""
图片平移操作
"""
remove_files_in_folder(out_path + 'shift')
# 数据高度平移，数据宽度平移, 传入的是相对长度宽度的比例值
datagen = image.ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
# 生成数据
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'shift',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
# print_result(out_path + 'shift\\*')

"""
图片缩放操作
"""
remove_files_in_folder(out_path + 'zoom')
# 随机缩放幅度 （图像的部分区域）
datagen = image.ImageDataGenerator(zoom_range=0.5)

# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])

datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'zoom',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
# print_result(out_path + 'zoom\\*')

"""
颜色通道变换
"""
remove_files_in_folder(out_path + 'channel')
# 随机通道偏移的幅度
datagen = image.ImageDataGenerator(channel_shift_range=15)
# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
# 生成数据
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'channel',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
# print_result(out_path + 'channel\\*')

"""
翻转
"""
remove_files_in_folder(out_path + 'horizontal')
# 进行水平翻转
datagen = image.ImageDataGenerator(horizontal_flip=True)
# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(
    in_path,
    batch_size=1,
    class_mode=None,
    shuffle=True,
    target_size=(224, 224)
)
np_data = np.concatenate([data.next() for i in range(data.n)])
# 生成数据
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'horizontal',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
# print_result(out_path + 'horizontal\\*')

"""
rescale变换
"""
remove_files_in_folder(out_path + 'rescale')
# 归一化到 0~1的数据返回
datagen = image.ImageDataGenerator(rescale=1 / 255)
# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
# 生成数据
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rescale',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
# print_result(out_path + 'rescale\\*')

"""
## 填充方法
 - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
 - 'nearest': aaaaaaaa|abcd|dddddddd
 - 'reflect': abcddcba|abcd|dcbaabcd
 - 'wrap': abcdabcd|abcd|abcdabcd
"""
remove_files_in_folder(out_path + 'fill_mode')
datagen = image.ImageDataGenerator(fill_mode='wrap', zoom_range=[4, 4])
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'fill_mode',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
# print_result(out_path + 'fill_mode\\*')

#### 使用最近点进行填充
remove_files_in_folder(out_path + 'nearest')
datagen = image.ImageDataGenerator(fill_mode='nearest', zoom_range=[4, 4])
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'nearest',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'nearest\\*')
