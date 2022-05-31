import collections
import random
import zipfile

import numpy as np
import tensorflow as tf

"""
word2vec随机采样模型
输入一个词, 预测上下文:
    打: 今天, 篮球 
"""

# 训练参数
learning_rate = 0.1
batch_size = 128
num_steps = 3000000  # 学习多少次
display_step = 10000  # 每10000次打印一次损失值
eval_step = 200000  # 每200000次测试效果

# 测试样例, 计算一下欧氏距离. 查看每一个测试样例附近的词大概是什么
eval_words = ['nine', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec 参数
embedding_size = 200  # 词向量维度, 官方设定为300没有什么特定值可以根据需求修改, 一般50-300
max_vocabulary_size = 50000  # 语料库词语
min_occurrence = 10  # 最小词频, 小于10的都去掉
skip_window = 3  # 左右窗口大小, 上下文窗口, 左窗口为3右窗口也为3, 窗口大小为7
num_skips = 2  # 一次制作多少个输入输出对, 针对于一个窗口定义输出2个
num_sampled = 64  # 负采样, 在50000个语料库中选择64个采样, 然后再64个中做分类操作.

# 加载训练数据，其实什么数据都行
data_path = 'E:\\Workspace\\ml\\code-ml\\ml\\python\\data\\text8.zip'  # 维基百科下载的文本文档zip <http://mattmahoney.net/dc/text8.zip>
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

# 查看一共多少个词
print(len(text_words))  # 17005207

# 创建一个计数器，计算每个词出现了多少次
# 'UNK' unknown, 不存在与语料表中的词设置为-1
count = [('UNK', -1)]
# 基于词频返回max_vocabulary_size个常用词, 也就是获取50000个最常用的词
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))

print(count[0:10])
# [('UNK', -1), (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764), (b'in', 372201), (b'a', 325873), (b'to', 316376), (b'zero', 264975), (b'nine', 250430)]

# 剔除掉出现次数少于'min_occurrence'的词
# 从后到前的每一个index: 49999->49998->49997...
for i in range(len(count) - 1, -1, -1):  # 从start到end每次step多少
    if count[i][1] < min_occurrence:  # 只获取大于10的
        count.pop(i)
    else:
        # 判断时，从小到大排序的，所以跳出时候剩下的都是满足条件的
        break

# 计算语料库大小
vocabulary_size = len(count)
# 每个词都分配一个ID, i就是id
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

print(word2id)

"""
将原始文件所有的词转换为id
"""
data = list()
unk_count = 0
for word in text_words:
    # 全部转换成id
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))  # 配置一个返结构, 通过id获取word的map结构

print("Words count:", len(text_words))  # Words count: 17005207
print("Unique words:", len(set(text_words)))  # Unique words: 253854
print("Vocabulary size:", vocabulary_size)  # Vocabulary size: 47135
print("Most common words:", count[:10])
# Most common words: [('UNK', 444176), (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764), (b'in', 372201), (b'a', 325873), (b'to', 316376), (b'zero', 264975), (b'nine', 250430)]

"""
构建训练数据
"""
data_index = 0


def next_batch(batch_size, num_skips, skip_window):  # 构建输入/输出数据
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 获取batch为ndarrray格式
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one).
    span = 2 * skip_window + 1  # 获取窗口大小, 7为窗口大小，左3右3中间1
    buffer = collections.deque(maxlen=span)  # 创建一个长度为7的队列
    if data_index + span > len(data):  # 如果数据被滑完一遍了, 那么重新开始
        data_index = 0

    # 队列里存的是当前窗口，例如deque([5234, 3081, 12, 6, 195, 2, 3134], maxlen=7)
    buffer.extend(data[data_index:data_index + span])
    data_index += span  # 当前指针向前7步

    ## num_skips表示取多少组不同的词作为输出, 此例为2. 因此循环只需要循环32次
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]  # buffer中上下文索引就是[0, 1, 2, 4, 5, 6]
        words_to_use = random.sample(context_words, num_skips)  # 在上下文里随机选2个候选词

        # 遍历每一个候选词，用其当做输出也就是标签
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]  # 输入, 当前窗口的中间词, 即buffer中的index=3的词
            labels[i * num_skips + j, 0] = buffer[context_word]  # 用当前随机出来的候选词当做标签

        # 窗口滑动动作, 通过队列进行控制
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            # 之前已经传入7个词了，窗口要右移了
            # 例如原来为[5234, 3081, 12, 6, 195, 2, 3134]，现在为[3081, 12, 6, 195, 2, 3134, 46]
            buffer.append(data[data_index])
            data_index += 1

    # 输出batch和label
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# Embedding层操作, 创建一个容器, 每一个词都有200个向量
# with关键字: 打开流, 方法结束后自动关闭
with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))  # 维度：47135, 200
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))  # 负采样层
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))  # 偏置参数


# 通过tf.nn.embedding_lookup函数将索引转换成词向量
def get_embedding(x):
    with tf.device('/cpu:0'):
        # 输入x, 从embedding表中获取x的200个向量
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed


# 损失函数定义
# - 先分别计算出正样本和采样出的负样本对应的output和label
# - 再通过 sigmoid cross entropy来计算output和label的loss
def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=y,
                           inputs=x_embed,
                           num_sampled=num_sampled,  # 采样出多少个负样本
                           num_classes=vocabulary_size))
        return loss


# Evaluation.
# 评估模块, 测试观察模块
# 通过一个词的向量,获取离这个词距离近的词有哪些
def evaluate(x_embed):
    with tf.device('/cpu:0'):
        # Compute the cosine similarity between input data embedding and every embedding vectors
        x_embed = tf.cast(x_embed, tf.float32)  # 获取某一个词的向量
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))  # 归一化
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)  # 全部向量的
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)  # 计算余弦相似度
        return cosine_sim_op


# 优化器SGD
optimizer = tf.optimizers.SGD(learning_rate)


# 迭代优化
def run_optimization(x, y):
    with tf.device('/cpu:0'):
        # 获取词向量, 然后通过x和y计算损失值
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)  # 调用nn的nce lose方法计算损失值

        # 通过损失值计算梯度
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])

        # 更新大表
        optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))


# 待测试的几个词
x_test = np.array([word2id[w.encode('utf-8')] for w in eval_words])

# 训练, 迭代
for step in range(1, num_steps + 1):
    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)  # 获取x和y
    run_optimization(batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))  # 如果到了打印的位置打印当前损失

    # Evaluation, 从上面定义的几个单词中获取8个相似的单词
    if step % eval_step == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k = 8  # 返回前8个最相似的
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)
