import jieba
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

    print(dict.inverse_transform(data))
    # 转换成之前的数据, 但是转换成特征值的类型
    # [{'city=北京': 1.0, 'temperature': 100.0}, {'city=上海': 1.0, 'temperature': 60.0}, {'city=深圳': 1.0, 'temperature': 30.0}]

    print(dict.get_feature_names_out())  # ['city=上海', 'city=北京', 'city=深圳', 'temperature']
    return None


def countVec():
    """
    对文本进行特征值化
    统计所有文章的词, 重复的只计算一次, 作为headers
    针对这个列表, 每一个文章统计单词个数, 每一个文章统计一次, 对于单个字母不统计(字母不会反映文章主题)

    默认不支持中文抽取, 优先进行中文分词
    :return: 
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["life is is short,i like python", "life is too long,i dislike python"])
    print(data.toarray())
    # [[0 2 1 1 0 1 1 0]
    #  [1 1 1 0 1 1 0 1]]

    print(cv.get_feature_names_out())
    # ['dislike' 'is' 'life' 'like' 'long' 'python' 'short' 'too']

    return None


def cutword():
    con1 = jieba.cut("1、今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("2、我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("3、如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    print(content1)
    content2 = list(con2)
    print(content2)
    content3 = list(con3)
    print(content3)

    # 转换成字符串
    c1 = " ".join(content1)
    c2 = " ".join(content2)
    c3 = " ".join(content3)
    return c1, c2, c3


def chinese_vec():
    """
    中文文本抽取
    :return:
    """
    cv = CountVectorizer()
    c1, c2, c4 = cutword()
    data = cv.fit_transform([c1, c2, c4])
    print(cv.get_feature_names_out())
    # ['一种', '不会', '不要', '之前', '了解', '事物', '今天', '光是在', '几百万年', '发出', '取决于', '只用', '后天', '含义', '大部分', '如何', '如果', '宇宙',
    #  '我们', '所以', '放弃', '方式', '明天', '星系', '晚上', '某样', '残酷', '每个', '看到', '真正', '秘密', '绝对', '美好', '联系', '过去', '这样']

    print(data.toarray())
    # [[0 0 1 0 0 0 2 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 2 0 1 0 2 1 0 0 0 1 1 0 0 0]
    #  [0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 1 3 0 0 0 0 1 0 0 0 0 2 0 0 0 0 0 1 1]
    #  [1 1 0 0 4 3 0 0 0 0 1 1 0 1 0 1 1 0 1 0 0 1 0 0 0 1 0 0 0 2 1 0 0 1 0 0]]


def tfidf_vec():
    tfidf_vector = TfidfVectorizer()
    c1, c2, c3 = cutword()
    data = tfidf_vector.fit_transform([c1, c2, c3])
    # data = tfidf_vector.fit_transform(["life is is short,i like python", "life is too long,i dislike python"])
    print(tfidf_vector.get_feature_names_out(data))
    print(data.toarray())
    # 输出结果 -> 数值表示的是重要性
    # [[0.         0.         0.21821789 0.         0.         0.
    #   0.43643578 0.         0.         0.         0.         0.
    #   0.21821789 0.         0.21821789 0.         0.         0.
    #   0.         0.21821789 0.21821789 0.         0.43643578 0.
    #   0.21821789 0.         0.43643578 0.21821789 0.         0.
    #   0.         0.21821789 0.21821789 0.         0.         0.        ]
    #  [0.         0.         0.         0.2410822  0.         0.
    #   0.         0.2410822  0.2410822  0.2410822  0.         0.
    #   0.         0.         0.         0.         0.         0.2410822
    #   0.55004769 0.         0.         0.         0.         0.2410822
    #   0.         0.         0.         0.         0.48216441 0.
    #   0.         0.         0.         0.         0.2410822  0.2410822 ]
    #  [0.15698297 0.15698297 0.         0.         0.62793188 0.47094891
    #   0.         0.         0.         0.         0.15698297 0.15698297
    #   0.         0.15698297 0.         0.15698297 0.15698297 0.
    #   0.1193896  0.         0.         0.15698297 0.         0.
    #   0.         0.15698297 0.         0.         0.         0.31396594
    #   0.15698297 0.         0.         0.15698297 0.         0.        ]]
    return None


def normalization():
    """
    归一化
    :return:
    """
    normalizer = MinMaxScaler()
    data = normalizer.fit_transform([[90, 2, 10, 40],
                                     [60, 4, 15, 45],
                                     [75, 3, 13, 46]])
    print(data)
    # [[1.         0.         0.         0.        ]
    #  [0.         1.         1.         0.83333333]
    #  [0.5        0.5        0.6        1.        ]]

    # 使用2-3缩放
    normalizer = MinMaxScaler(feature_range=(2, 3))
    data = normalizer.fit_transform([[90, 2, 10, 40],
                                     [60, 4, 15, 45],
                                     [75, 3, 13, 46]])
    print(data)
    # [[3.         2.         2.         2.        ]
    #  [2.         3.         3.         2.83333333]
    #  [2.5        2.5        2.6        3.        ]]
    return None


def standarlization():
    """
    标准化缩放
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.],
                              [2., 4., 2.],
                              [4., 6., -1.]])
    print(data)
    # [[-1.06904497 -1.35873244  0.98058068]
    #  [-0.26726124  0.33968311  0.39223227]
    #  [ 1.33630621  1.01904933 -1.37281295]]
    return None


def imputer():
    """
    缺失值处理
    :return:
    """
    # 替换策略:
    # "mean"，使用该列的平均值替换缺失值。仅用于数值数据；
    # "median"，使用该列的中位数替换缺失值。仅用于数值数据；
    # "most_frequent"，使用每个列中最常见的值替换缺失值。可用于非数值数据；
    # "constant"，用fill_value替换缺失值。可用于非数值数据
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform([[1, 2],
                              [np.nan, 3],
                              [7, 6]])
    print(data)
    # [[1. 2.]
    #  [4. 3.]
    #  [7. 6.]]
    return None


def variance():
    """
    删除低方差的特征
    """
    var = VarianceThreshold(threshold=0.00001)  # 删除方差低于0.00001的数据
    data = var.fit_transform([[0, 2, 0, 3],
                              [0, 1, 4, 3],
                              [0, 1, 1, 3]])
    print(data)
    # [[2 0]
    #  [1 4]
    #  [1 1]]
    return None


def pca():
    """
    主成分分析进行数据降维
    :return:
    """
    p = PCA(n_components=0.9)
    data = p.fit_transform([[2, 8, 4, 5],
                            [6, 3, 0, 8],
                            [5, 4, 9, 1]])
    print(data)
    # [[ 1.28620952e-15  3.82970843e+00]
    #  [ 5.74456265e+00 -1.91485422e+00]
    #  [-5.74456265e+00 -1.91485422e+00]]
    return None


if __name__ == '__main__':
    # dictVec()
    # countVec()
    # chinese_vec()
    # tfidf_vec()
    # normalization()
    # standarlization()
    # imputer()
    # variance()
    pca()
