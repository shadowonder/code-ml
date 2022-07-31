# coding:utf-8

import numpy as np
from scipy import fft
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression

"""
使用logistic regression处理音乐数据,音乐数据训练样本的获得和使用快速傅里叶变换（FFT）预处理的方法需要事先准备好
1. 把训练集扩大到每类100个首歌,类别仍然是六类:jazz,classical,country, pop, rock, metal
2. 同时使用logistic回归训练模型
3. 引入一些评价的标准来比较Logistic测试集上的表现 
"""

# str(3).zfill(5) => 00003; str(123).zfill(7) => 0000123;
# 准备音乐数据
# def create_fft(g, n):
#     rad = "i:/genres/" + g + "/converted/" + g + "." + str(n).zfill(5) + ".au.wav"
#     # sample_rate 音频的采样率，X代表读取文件的所有信息
#     (sample_rate, X) = wavfile.read(rad)
#     # 取1000个频率特征  也就是振幅
#     fft_features = abs(fft(X)[:1000])
#     # zfill(5) 字符串不足5位，前面补0
#     save = "i:/trainset/" + g + "." + str(n).zfill(5) + ".fft"  # 转换的结果写入硬盘
#     np.save(save, fft_features)


# -------create fft 构建训练集--------------
# 读取全部的测试文件, 转换为fft文件然后写入硬盘
# genre_list = ["classical", "jazz", "country", "pop", "rock", "metal", "hiphop"]
# for g in genre_list:
#     for n in range(100):
#         create_fft(g, n)
#         print('running...')
# print('all is finished')

# =========================================================================================
# 加载训练集数据,分割训练集以及测试集,进行分类器的训练
# 构造训练集！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# -------read fft--------------
genre_list = ["classical", "jazz", "country", "pop", "rock", "metal", "hiphop"]
X = []
Y = []
for g in genre_list:
    for n in range(100):
        rad = "i:/trainset/" + g + "." + str(n).zfill(5) + ".fft" + ".npy"  # 从硬盘中读取训练集数据
        # 加载文件
        fft_features = np.load(rad)
        X.append(fft_features)
        # genre_list.index(g) 返回匹配上类别的索引号, 返回下标
        Y.append(genre_list.index(g))

# 构建的训练集
X = np.array(X)
# 构建的训练集对应的类别
Y = np.array(Y)

# 接下来，我们使用sklearn，来构造和训练我们的两种分类器 
# ------train logistic classifier--------------
model = LogisticRegression()
# 需要numpy.array类型参数
model.fit(X, Y)

print('Starting read wavfile...')
# prepare test data-------------------
sample_rate, test = wavfile.read("i:/heibao-wudizirong-remix.wav")
print(sample_rate, test, len(test))
testdata_fft_features = abs(fft(test))[:1000]
# model.predict(testdata_fft_features) 预测为一个数组，array([类别])
type_index = model.predict([testdata_fft_features])

print(type_index, type(type_index))
print(genre_list[type_index[0]])
