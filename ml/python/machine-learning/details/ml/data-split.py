from sklearn.datasets import load_iris, load_boston

li = load_iris()
print("获取特征值")
print(li.data)
# 获取特征值
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
# ...

print("目标值")
print(li.target)
# 目标值
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

print(li.DESCR)
# 展示花的特征

# # 返回值:训练集x_train,y_train. 测试集x_test,y_test
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
# print("训练集特征值和目标值", x_train, y_train)
# print("测试集特征值和目标值", x_test, y_test)

# news = fetch_20newsgroups(subset='all')
# print(news.data)
# print(news.target)

lb = load_boston()
print(lb.data)  # 特征值
print(lb.target)  # 目标值
print(lb.DESCR)
