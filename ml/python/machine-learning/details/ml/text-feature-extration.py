from sklearn.feature_extraction import DictVectorizer


def dictVec():
    """
    字典抽取数据
    :return:
    """
    # 实例化
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{'city': '北京', 'temperature': 100},
                               {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}])
    print(data)
    # sparse = true
    #   (0, 1)	1.0
    #   (0, 3)	100.0
    #   (1, 0)	1.0
    #   (1, 3)	60.0
    #   (2, 2)	1.0
    #   (2, 3)	30.0
    # sparse = false , 也就是ndarray的类型
    # 也被称为one hot编码
    # [[  0.   1.   0. 100.]
    #  [  1.   0.   0.  60.]
    #  [  0.   0.   1.  30.]]

    print(dict.get_feature_names())  # ['city=上海', 'city=北京', 'city=深圳', 'temperature']
    return None


if __name__ == '__main__':
    dictVec()
