import torch
from torch import nn  # 神经网路蝉蛹

# from torch import functional as F  # 旧版
from torch.nn import functional as F  # 常用函数
from torch import optim  # 优化工具包
import torchvision  # 视觉
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot


# 1. 加载数据集, 通过touchvision工具包直接下载, 并行读取多张图片. 这里的图片是28x28
# 70k图片是用来训练,
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "data/mnist_data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # 数据下载以后不是一个正态分布, 可以使用这个来进行正态分布, 主要影响的是性能, 其他的都不变
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,  # 每次训练加载多少张图片
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "data/mnist_data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=False,
)


# 显示图片
x, y = next(iter(test_loader))
# 如果不添加Normalize, 会均匀地分布在0-1之间, 而现在就变换成了这样, 是的数据在0附近
# torch.Size([512, 1, 28, 28]) torch.Size([512]) tensor(-0.4242) tensor(2.8215)
print(x.shape, y.shape, x.min(), x.max())
# plot_image(x, y, "image sample")  # 打印数据

###
# 创建网络
###
class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ## 新建三层, 每一层wx+b
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入28*28, 输出256
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)  # 输出层

    ## 定义一个计算过程, 输入包含x
    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(wx+b)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3  暂时不加激活函数
        x = self.fc3(x)
        return x


net = Net()

# net.parameters() 当中包含的就是 [w1,b1,w2,b2,w3,b3]
# Momentum: SGD 在 ravines 的情况下容易被困住， ravines 就是曲面的一个方向比另一个方向更陡，
# 这时 SGD 会发生震荡而迟迟不能接近极小值
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []
## 测试/训练
# 训练三遍
for epoch in range(3):
    # 对内部层数迭代一次, 每一次迭代拿到一个batch512个文件
    for batch_inx, (x, y) in enumerate(train_loader):
        # print(x.shape, y.shape)  # torch.Size([512, 1, 28, 28]) torch.Size([512])
        # 将向量flattern平铺拉伸
        x = x.view(x.size(0), 28 * 28)  # => [b, 10]
        out = net(x)
        # 将y转换为Onehot
        y_onehot = one_hot(y)
        # print(y_onehot.shape) # torch.Size([512, 10])
        loss = F.mse_loss(out, y_onehot)  # 计算mse

        # 清零梯度, grad = 0, 否则的话grad就会加载上一次grad的总和上
        optimizer.zero_grad()
        loss.backward()
        # 更新梯度 w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())
        if batch_inx % 10 == 0:
            print(epoch, batch_inx, loss.item())
    """ end batch """
""" end epoch """

# 获得了[w1, b1, w2, b2, w3, b3]
# plot_curve(train_loss) # 打印训练的loss值

# 我们观察一下测试
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out: [512, 10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float()
    total_correct += correct

# 获取所有数据集的数量
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("================================")
print("acc:{}".format(acc))  # acc:0.6884999871253967
print("================================")


# 我们对之前的sample进行预测
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, "test")
