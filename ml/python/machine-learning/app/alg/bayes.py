import codecs

from sklearn.feature_extraction.text import CountVectorizer  # 特征工程下的文字处理
from sklearn.naive_bayes import MultinomialNB

"""
邮件分类
贝叶斯算法
数据源<https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset>
"""

print('*************************\nNaive Bayes\n*************************')

if __name__ == '__main__':
    # 读取文本构建语料库
    corpus = []
    labels = []
    corpus_test = []
    labels_test = []
    f = codecs.open("../../../data/sms_spam.txt", "r", "UTF-8")
    count = 0
    while True:
        # readline() 方法用于从文件读取整行，包括 "\n" 字符。
        line = f.readline()
        # 读取第一行，第一行数据是列头，不统计
        if count == 0:
            count = count + 1
            continue
        if line:
            count = count + 1
            line = line.split(",")
            label = line[0]
            sentence = line[1]
            if count <= 5550:  # 5550 以前为训练集
                corpus.append(sentence)
                if "ham" == label:
                    labels.append(0)
                elif "spam" == label:
                    labels.append(1)
            if count > 5550:  # 最后几个作为测试集
                corpus_test.append(sentence)
                if "ham" == label:
                    labels_test.append(0)
                elif "spam" == label:
                    labels_test.append(1)
        else:
            break
    # 文本特征提取：
    #     将文本数据转化成特征向量的过程
    #     比较常用的文本特征表示法为词袋法
    #
    # 词袋法：
    #     不考虑词语出现的顺序，每个出现过的词汇单独作为一列特征
    #     这些不重复的特征词汇集合为词表
    #     每一个文本都可以在很长的词表上统计出一个很多列的特征向量
    # CountVectorizer是将文本向量转换成稀疏表示数值向量（字符频率向量）  vectorizer 将文档词块化,只考虑词汇在文本中出现的频率
    # 词袋
    vectorizer = CountVectorizer()
    # 每行的词向量，fea_train是一个矩阵
    # 此处的处理类似倒排索引, 将每一个字符串向量化
    fea_train = vectorizer.fit_transform(corpus)

    print("vectorizer.get_feature_names is ", vectorizer.get_feature_names())
    print("fea_train is ", fea_train.toarray())

    # vocabulary=vectorizer.vocabulary_ 只计算上面vectorizer中单词的tf(term frequency 词频)
    vectorizer2 = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    fea_test = vectorizer2.fit_transform(corpus_test)
    #     print vectorizer2.get_feature_names()
    #     print fea_test.toarray()

    # create the Multinomial Naive Bayesian Classifier
    # alpha = 1 拉普拉斯估计给每个单词个数加1
    clf = MultinomialNB(alpha=1)
    # fit数据
    clf.fit(fea_train, labels)
    # 进行预测
    pred = clf.predict(fea_test)  # 对最后几个进行预测
    for p in pred:
        if p == 0:
            print("ham-正常邮件")
        else:
            print("spam-垃圾邮件")
