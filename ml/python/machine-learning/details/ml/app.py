from sklearn.feature_extraction.text import CountVectorizer

# 实例化
vector = CountVectorizer()
# 转换数据
res = vector.fit_transform(["life is short , i like python", "life is too long, i dislike python"])

print(vector.get_feature_names())
# ['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']

print(res.toarray())
# [[0 1 1 1 0 1 1 0]
#  [1 1 1 0 1 1 0 1]]

print(type(res))  # <class 'scipy.sparse._csr.csr_matrix'>
