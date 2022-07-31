from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB


def bayes_algorithom():
    """
    朴素贝叶斯分类算法
    :return:
    """
    # 下载新闻数据
    news = fetch_20newsgroups(subset='all')

    # 进行数据分隔, 25%的测试数据
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据及进行特征抽取
    tf = TfidfVectorizer()  # 使用文本特征抽取
    x_train = tf.fit_transform(x_train)  # 针对每篇文章的词进行统计
    x_test = tf.transform(x_test)  # 使用同样的特征抽取测试集, 并进行统计, 这样特征数量是相同的

    print(tf.get_feature_names_out())

    # 进行朴素贝叶斯算法进行预测
    mlt = MultinomialNB(alpha=1.0)
    print(x_train.toarray())
    mlt.fit(x_train, y_train)

    # 得出准确率
    y_predict = mlt.predict(x_test)
    print("预测的文章类别为: ", y_predict)
    print("准确率为: ", mlt.score(x_test, y_test))
    # 预测的文章类别为:  [13 10  7 ... 15 15 10]
    # 准确率为:  0.8552631578947368

    print("每个类别的精确率和召回率\n", classification_report(y_test, y_predict, target_names=news.target_names))
    # 每个类别的精确率和召回率
    #                            precision    recall  f1-score   support
    #
    #              alt.atheism       0.89      0.77      0.83       201
    #            comp.graphics       0.93      0.78      0.85       256
    #  comp.os.ms-windows.misc       0.86      0.81      0.84       261
    # comp.sys.ibm.pc.hardware       0.74      0.85      0.79       255
    #    comp.sys.mac.hardware       0.88      0.86      0.87       231
    # ...

    return None


if __name__ == '__main__':
    bayes_algorithom()
