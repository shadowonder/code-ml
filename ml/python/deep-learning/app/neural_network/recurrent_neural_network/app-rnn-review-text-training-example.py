from collections import Counter
import logging
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

folder_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\text-class\\'
train_file = folder_path + 'train.txt'
test_file = folder_path + 'test.txt'
vocab_folder = folder_path + 'vocab'
vocab_word_file = vocab_folder + "\\" + 'word.txt'
vec_pre_trained = folder_path + 'glove.6B.50d.txt'
vec_words_npy_file = folder_path + "words.npy"

# 下载数据影评数据, 也可以手动下载
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()

print(x_train.shape)  # (25000,)
# tensorflow中已经将数据修改为词向量. 可以直接使用计算.
print(x_train[0])
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941,...]

# 下载原始的索引表, 索引表是字典序列
_word2idx = tf.keras.datasets.imdb.get_word_index()
word2idx = {w: i + 3 for w, i in _word2idx.items()}
# 文本的特征字符flag
word2idx['<pad>'] = 0
word2idx['<start>'] = 1
word2idx['<unk>'] = 2
idx2word = {i: w for w, i in word2idx.items()}


# 排序方法
def sort_by_len(x, y):
    x, y = np.asarray(x), np.asarray(y)
    idx = sorted(range(len(x)), key=lambda i: len(x[i]))
    return x[idx], y[idx]


# 排序后重新定义
x_train, y_train = sort_by_len(x_train, y_train)
x_test, y_test = sort_by_len(x_test, y_test)


# 将重新排序和定以后的数据写入本地文件, 以便后来使用
def write_file(f_path, xs, ys):
    with open(f_path, 'w', encoding='utf-8') as f:
        for x, y in zip(xs, ys):
            f.write(str(y) + '\t' + ' '.join([idx2word[i] for i in x][1:]) + '\n')


write_file(train_file, x_train, y_train)
write_file(test_file, x_test, y_test)

print("File write completed.")

# 创建语料表, 基于词频来进行统计
counter = Counter()
# 打开文档, 读取全部语句和词频
with open(train_file, encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        label, words = line.split('\t')
        words = words.split(' ')
        counter.update(words)

# 如果词频出现小于10, 那么久无视掉
words = ['<pad>'] + [w for w, freq in counter.most_common() if freq >= 10]
print('Vocab Size:', len(words))

# 创建文件夹, 准备存放语料库
Path(vocab_folder).mkdir(exist_ok=True)

# 语料库写入文件
with open(vocab_word_file, 'w', encoding='utf-8') as f:
    for w in words:
        f.write(w + '\n')

print("Word write completed.")

# 新的word2id映射表
word2idx = {}
with open(vocab_word_file, encoding='utf-8') as f:
    # 打开文件, 读取每一行, 然后放入字典, 行数就是id
    for i, line in enumerate(f):
        line = line.rstrip()
        word2idx[line] = i

"""
embedding层, 可以基于网络来训练，也可以直接加载别人训练好的，一般都是加载预训练模型
"""
# 做了一个大表，里面有20598个不同的词，【20599*50】
embedding = np.zeros((len(word2idx) + 1, 50))  # + 1 表示如果不在语料表中，就都是unknow
# 读取别人训练好的模型文件, 这里是一个50维的训练模型
# 输出的结果就是一个 id - vec 的词典
with open(vec_pre_trained, encoding='utf-8') as f:  # 下载好的
    count = 0
    # 遍历所有的word, 然后切分
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print('- At line {}'.format(i))  # 打印处理了多少数据
        line = line.rstrip()
        sp = line.split(' ')
        word, vec = sp[0], sp[1:]
        if word in word2idx:
            count += 1
            # 转换成索引,然后获取位置
            embedding[word2idx[word]] = np.asarray(vec, dtype='float32')  # 将词转换成对应的向量

# 打印处理的结果
# [19676 / 20598] words have found pre-trained values
print("[%d / %d] words have found pre-trained values" % (count, len(word2idx)))
np.save(vec_words_npy_file, embedding)
print('Saved ' + vec_words_npy_file)

"""
构建训练数据
数据生成器 

将tensor沿其第一个维度切片，返回一个含有N个样本的数据集，
这样做的问题就是需要将整个数据集整体传入，然后切片建立数据集类对象，比较占内存:
tf.data.Dataset.from_tensor_slices(tensor)

从一个生成器中不断读取样本:
tf.data.Dataset.from_generator(data_generator,output_data_type,output_data_shape)
"""


# 定义一个生成器函数
# f_path: 数据文件的路径
# params: 参数(word2idx,max_len)
# 生成器其实是一种特殊的迭代器，但是不需要像迭代器一样实现__iter__和__next__方法，只需要使用关键字yield就可以。
# 一般不需要将所有的数据都放入生成器, 因为还要考虑到数据的io问题
def data_generator(f_path, params):
    with open(f_path, encoding='utf-8') as f:  # 打开训练数据
        print('Reading', f_path)
        for line in f:
            line = line.rstrip()
            label, text = line.split('\t')  # 切分x和y
            text = text.split(' ')
            # 在之前封装好的word2idx中获取对应的id
            x = [params['word2idx'].get(w, len(word2idx)) for w in text]  # 得到当前词所对应的ID
            if len(x) >= params['max_len']:  # 截断操作
                x = x[:params['max_len']]
            else:
                x += [0] * (params['max_len'] - len(x))  # 补齐操作, 用0补齐也就是<pad>
            y = int(label)
            yield x, y


# 数据集
# is_training: 是否训练, 在验证或者测试的时候可以设置为false
# params: 参数{max_len:最大长度, train_path:训练集的路径, num_samples, batch_size,test_path}
def dataset(is_training, params):
    _shapes = ([params['max_len']], ())
    _types = (tf.int32, tf.int32)
    if is_training:
        # 使用from_generator获取数据
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['train_path'], params),
            output_shapes=_shapes,  # 输出的数据形式
            output_types=_types, )  # 输出的数据类型
        ds = ds.shuffle(params['num_samples'])
        ds = ds.batch(params['batch_size'])
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)  # 设置缓存序列，根据可用的CPU动态设置并行调用的数量，说白了就是加速
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['test_path'], params),  # 获取训练集数据
            output_shapes=_shapes,
            output_types=_types, )
        ds = ds.batch(params['batch_size'])
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


"""
自定义网络模型

创建一个简单的模型. 在init方法中定义当前模型使用的所有layer
然后再call方法中调用, 可以执行前向传播的过程. model就定义好了. 
这里并没有反向传播, 因为模型继承了keras的model, 反向传播会自动计算.
"""


# 函数集成tensorflow的model模型
class Model(tf.keras.Model):
    # 初始化方法, 把我们需要的网络层放入计算中
    def __init__(self, params):
        super().__init__()
        self.params = params

        # 词嵌入层
        self.embedding = tf.Variable(np.load(vec_words_npy_file),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False, )
        # 定义drop out层, 防止过拟合
        self.drop1 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop3 = tf.keras.layers.Dropout(params['dropout_rate'])

        # rnn层, 会很慢, 因为rnn的层不是并行的. Bidirectional就是双向rnn
        # 第一个参数, 传入多少个节点
        # 第二个参数, 当前层是否获取序列(w), 一般上一层结果会成为下一层的输入, 因此最后一层不需要返回序列
        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=False))

        # 全连接层
        self.drop_fc = tf.keras.layers.Dropout(params['dropout_rate'])
        # 由于是双向rnn因此得到两倍的节点数量
        self.fc = tf.keras.layers.Dense(2 * params['rnn_units'], tf.nn.elu)

        self.out_linear = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)

        batch_sz = tf.shape(inputs)[0]
        rnn_units = 2 * self.params['rnn_units']

        # 1. embedding层中获取词向量表, 然后通过计算inputs, 转换为向量
        # 输出的长度会是batchSize x max_len x 50 (因为有50个维度)
        x = tf.nn.embedding_lookup(self.embedding, inputs)

        # 2. 输入到rnn中
        x = self.drop1(x, training=training)
        x = self.rnn1(x)

        x = self.drop2(x, training=training)
        x = self.rnn2(x)

        x = self.drop3(x, training=training)
        x = self.rnn3(x)

        # 3. 输入到全连接中中
        x = self.drop_fc(x, training=training)
        x = self.fc(x)

        x = self.out_linear(x)
        return x


# ========================================版本2=====================================
# 和上面的模型使用的是同样的处理思路, 但是使用的是不同的call方法. 相比与上面的方法速度更快
#  rnn3的return_sequences置为true,因为不需要获取最后一个序列, 因为压缩以后的结果会是一个, 毕竟最后得到的也是特征
class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        self.embedding = tf.Variable(np.load(vec_words_npy_file),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False, )

        self.drop1 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop3 = tf.keras.layers.Dropout(params['dropout_rate'])

        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))

        self.drop_fc = tf.keras.layers.Dropout(params['dropout_rate'])
        self.fc = tf.keras.layers.Dense(2 * params['rnn_units'], tf.nn.elu)

        self.out_linear = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)

        batch_sz = tf.shape(inputs)[0]
        rnn_units = 2 * params['rnn_units']

        x = tf.nn.embedding_lookup(self.embedding, inputs)

        # 相比与上面的方法, 添加了reduce_max方法, 返回x个最大特征
        # 这里就是一个特征压缩, 为了更快的处理
        x = tf.reshape(x, (batch_sz * 10 * 10, 10, 50))
        x = self.drop1(x, training=training)
        x = self.rnn1(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_sz * 10, 10, rnn_units))
        x = self.drop2(x, training=training)
        x = self.rnn2(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_sz, 10, rnn_units))
        x = self.drop3(x, training=training)
        x = self.rnn3(x)
        x = tf.reduce_max(x, 1)

        x = self.drop_fc(x, training=training)
        x = self.fc(x)

        x = self.out_linear(x)
        return x


# 设置参数
params = {
    'vocab_path': vocab_word_file,  # 定义文件路径
    'train_path': train_file,
    'test_path': test_file,
    'num_samples': 25000,
    'num_labels': 2,
    'batch_size': 32,
    'max_len': 1000,
    'rnn_units': 200,
    'dropout_rate': 0.2,
    'clip_norm': 10.,  # 当梯度波动过大时让其控制在一个最大的范围内, 不要太大
    'num_patience': 3,
    'lr': 3e-4,
}


# 判断是否提前停止, 传入准确率, 如果准确率已经达标了那么久可以提前停止了
# 如果连续3次学习没有进步则停下来
def is_descending(history: list):
    history = history[-(params['num_patience'] + 1):]
    for i in range(1, len(history)):
        if history[i - 1] <= history[i]:
            return False
    return True


# id映射, 从本地文件中读取语料表
word2idx = {}
with open(params['vocab_path'], encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        word2idx[line] = i
params['word2idx'] = word2idx
params['vocab_size'] = len(word2idx) + 1

# 构建模型
model = Model(params)
model.build(input_shape=(None, None))  # 设置输入的大小，或者fit时候也能自动找到
# pprint.pprint([(v.name, v.shape) for v in model.trainable_variables])

# 学习率的衰减, 我们希望学习率不要保持在一个值 如果这个值过大的时候可能很难收敛
# 链接：https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay?version=stable
# return initial_learning_rate * decay_rate ^ (step / decay_steps)
decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)  # 相当于加了一个指数衰减函数
optim = tf.optimizers.Adam(params['lr'])  # 优化器
global_step = 0  # 当前迭代次数

history_acc = []
best_acc = .0  # 当前最好的准确率

t0 = time.time()
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

"""
====================================================================
开始训练
====================================================================
"""
while True:
    # 训练模型
    # 获取数据
    for texts, labels in dataset(is_training=True, params=params):
        # 梯度带，记录所有在上下文中的操作，并且通过调用.gradient()获得任何上下文中计算得出的张量的梯度
        with tf.GradientTape() as tape:
            logits = model(texts, training=True)  # 前向传播获得预测结果logits
            # 计算损失值, 通过损失函数softmax
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # 获取平均损失值
            loss = tf.reduce_mean(loss)

        # 更新学习率, 衰减学习率
        optim.lr.assign(decay_lr(global_step))
        # 将损失值和模型的参数数据放入梯度带中, 这里做的就是获取默认数值
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, params['clip_norm'])  # 阶段梯度, 将梯度限制一下，有的时候回更新太猛，防止过拟合
        optim.apply_gradients(zip(grads, model.trainable_variables))  # 更新梯度, 将更新的梯度放入模型中

        # 每50次打印结果
        if global_step % 50 == 0:
            logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
                global_step, loss.numpy().item(), time.time() - t0, optim.lr.numpy().item()))
            t0 = time.time()
        global_step += 1

    # 验证集效果
    m = tf.keras.metrics.Accuracy()

    # 在模型中获取数据, 由于是训练所以is_training设置为false
    for texts, labels in dataset(is_training=False, params=params):
        logits = model(texts, training=False)  # 由于是测试, 直接获取预测值, 不需要进行训练, 不训练模型就不修改
        y_pred = tf.argmax(logits, axis=-1)  # 返回的是准确率, 通过softmax获取最终预测的类别
        m.update_state(y_true=labels, y_pred=y_pred)  # 计算准确率

    # 展示准确率
    acc = m.result().numpy()
    logger.info("Evaluation: Testing Accuracy: {:.3f}".format(acc))
    history_acc.append(acc)

    if acc > best_acc:
        best_acc = acc
    logger.info("Best Accuracy: {:.3f}".format(best_acc))

    # 如果准确率没有上升, 那么停止, 如果准确率超过了我们需要的值 也让他停下来
    if len(history_acc) > params['num_patience'] and is_descending(history_acc):
        logger.info("Testing Accuracy not improved over {} epochs, Early Stop".format(params['num_patience']))
        break

# ...
# INFO:tensorflow:Step 10800 | Loss: 0.2654 | Spent: 77.7 secs | LR: 0.000172
# INFO:tensorflow:Step 10850 | Loss: 0.1829 | Spent: 77.7 secs | LR: 0.000172
# INFO:tensorflow:Step 10900 | Loss: 0.2204 | Spent: 77.7 secs | LR: 0.000172
# Reading ./data/test.txt
# INFO:tensorflow:Evaluation: Testing Accuracy: 0.863
# INFO:tensorflow:Best Accuracy: 0.879
# INFO:tensorflow:Testing Accuracy not improved over 3 epochs, Early Stop
