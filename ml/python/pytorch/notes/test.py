# %% [markdown]
# # AutoEncoder
# 
# auto encoder 主要用于降维 dimension reduction. 同时也是unsupervised learning的主要使用工具.
# 
# 主要逻辑就是重构输入reconstruct. 输入层和输出曾是一样大小的. 中间会有一个neck. 可以升维也可以降维, 一般都是降维. 这样我们就包含了语义相关性. neck的前半部分被称作encoder, 后半部分称为decoder.
# 
# ![](./assets/27.png)

# %% [markdown]
# 我们主要的对比就是输入和输出的误差最小. 这里比较的是每一个点的误差. 也就是输出出来的应该是一个特征类似的图谱. 所以可以使用MSE. 当然, 如果输入是一个二进制类型的输入(每一个pixel都是二进制的), 那么我们也可以使用交叉熵计算loss. 可以看出中间的一部分就是整个网络的精髓. 我们只需要创建一个晓得神经网络学习这个精髓. 
# 
# ![](./assets/28.png)
# 
# 由于autoencoder也是一个神经网络, 所以可以使用神经网络的各种特性, 比如dropout.

# %% [markdown]
# **Adversarial Autoencoder 对抗式自编码器**
# 
# 相比于原始的autoencoder, 对抗式自编码器根据对抗网络创建了鉴别器. 在自编码生成器生成数据的时候, 我们训练的特征或者说属性对于每一个训练集都应该存在的, 这种属性的分布应该成一种正态分布. 这个时候我们使用一个鉴别器来查看我们生成的中间neck部分的结果是不是是一个正态分布, 或者我们自定义一种分布. 如果是的话, 那么我们就输出一个1或者0. 这样我们也能处理一些数据不均匀的问题.
# 
# ![](./assets/29.png)

# %% [markdown]
# 对于一个autoencoder, 我们网络中理想就是我们的得到的中间相的"变化程度"越接近输入的"变化程度"(就是相关性性很大). 因此就有了下面的公式, z就是我们的neck的结果, 我们希望给定$x_i$的时候会出现z的结果越大越好, 这样第一项也就是loss就趋近于零(q就是encoder网络的输出分布). 第二项KL曲线就是正态分布的重叠, 范围是0-inf. 如果两个分布相同的时候, KL就会趋近于0. 反之就会越来越大, 最大到正无穷. 因此我们希望如果使用在z出现的时候, x的分布和z的原始分布尽可能的相同, 否则的话KL函数就会越来越大, 这一项也就是误差.
# 
# 总体来说, 下面的公式就是一个loss function, 第一项是来对比当输入x的时候同时得到z的结果概率, 这个概率进行log的结果应该是一个负值. 概率越大越趋近于0, 概率越小越趋近于负无穷. 第二项则是比较x和neck的分布, 如果分布差距过大, 那么第二项也会变大. 总体的error也就是一个正值.
# 
# ![](./assets/30.png)
# 
# KL的计算公式如下, 结果就是两个分布的方差和均值进行计算.
# 
# ![](./assets/31.png)

# %% [markdown]
# 需要注意的是, 当我们在计算分布的时候我们没有办法得到所有的x和h, 因此我们得到的其实是一个抽样的结果, 最终得到一个$\sigma$和一个均值$\mu$. 但是这就出现了一个问题, 我们得到的这个loss function是没有办法反向传播或者微分的. 因此我们可以构建一个函数:
# 
# $$z=\mu+\sigma\cdot\epsilon$$
# 
# 这个函数有一个好处, 均值和bias很像, 而且同时包含了所有的信息. 

# %%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# %%
# 读取数据集
mnist_train = datasets.MNIST(
    "../data",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True,
)
mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

mnist_test = datasets.MNIST(
    "../data",
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True,
)
mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

x, _ = next(iter(mnist_train))
print("x:", x.shape)


# %%
# 创建autoencoder
# 输入的是一个28x28的图片
class MyAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(MyAutoEncoder, self).__init__()
        # [b, 784]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU(),
        )
        # [b, 20]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),  # 将每个像素点压缩到0-1的区间, 使用sigmoid函数比较好
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batchsz, 1, 28, 28)


# %%
device = torch.device("mps")

model = MyAutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criteon = nn.MSELoss()
print(model)


# %%
# 开始训练
import visdom

viz = visdom.Visdom()


for epoch in range(1000):
    for batchidx, (x, _) in enumerate(mnist_train):
        model.train()
        # [b, 1, 28, 28]
        x = x.to(device)

        x_hat = model(x)
        # 这里计算的就是整个图片的loss, 这里直接和原图比较
        loss = criteon(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 直接读取
    print("Epoch : {}, loss: {} ", epoch, loss.item())

    model.eval()
    x = iter(mnist_test).next()
    x.to(device)
    with torch.no_grad():
        x_hat = model(x)
    viz.image(x, nrow=8, win="x", opts=dict(title="x"))
    viz.image(x_hat, nrow=8, win="x_hat", opts=dict(title="x_hat"))



