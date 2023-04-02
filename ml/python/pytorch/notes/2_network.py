import numpy as np
from matplotlib import pyplot as plt
import torch


# 平面上每一个点都是一个点, x就是横坐标, y就是纵坐标
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# 我们直接对其进行绘图, x和y的值都输入进来, 然后得到一个图像.
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# 我们现在进行预测
# 我们可以随机选择几个初始化点: [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([1., 0.], requires_grad=True)  # 创建一个初始化点, 需要进行梯度信息更新
# 引入优化器
# 优化器的优化目标就是x, 学习率0.001
# x' = x - 0.001 dx
# y' = y - 0.001 dy
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
    pred = himmelblau(x)  # 进行计算
    optimizer.zero_grad()  # 梯度清零
    pred.backward()  # 反向传播#
    optimizer.step()  # 优化梯度

    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {}'
              .format(step, x.tolist(), pred.item()))
