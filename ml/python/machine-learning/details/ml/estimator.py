from sklearn.preprocessing import StandardScaler

s = StandardScaler()
s.fit_transform([[1, 2, 3], [4, 5, 6]])

ss = StandardScaler();
ss.fit([[1, 2, 3], [4, 5, 6]])
print(ss.transform([[1, 2, 3], [4, 5, 6]]))
# [[-1. -1. -1.]
#  [ 1.  1.  1.]]

# fit_transform = fit + transform

ss.fit([[1, 2, 3], [4, 5, 7]])  # 此处运算的标准差和方差
print(ss.transform([[1, 2, 3], [4, 5, 6]]))  # 由于标准差fit计算出来的不一样,因此结果不同
# [[-1.  -1.  -1. ]
#  [ 1.   1.   0.5]]0
