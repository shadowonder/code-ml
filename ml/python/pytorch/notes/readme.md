# PyTorch

install:

```shell script
pip install torch
# or
pip install torch==1.3.0+cpu torchversion=0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# or gpu
pip install torch===1.3.0 torchversion=0.4.1 -f https://download.pytorch.org/whl/torch_stable.html
```

如果需要使用gpu的话需要安装cuda, cuda是支持Nvidia显卡. AMD显卡使用的是Opencl但是不支持pytorch

使用python命令检查gpu是否可用

```python
import torch
torch.cuda.is_available() ## 返回是否可用
```

## 笔记

在pytorch的nn中, torch.utils.data.DataLoader 的dataset参数所需求的输入类需要实现`map`类型或者`iterable`类型的数据. 默认情况下 torch.datasets 包中的所有测试数据集都实现了map接口, 这里提供一个案例：

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

在Dataloader中就可以直接调用上面的类, 因为实现了map所需要的两个接口方法 `__len__`, `__getitem__`

***

在构建新的类的时候可能需要知道类的建立返回值, 因此就可以在方法中定义返回类型, 一般情况下构造方法的返回类型都是None, 因此直接写上返回类型为None如下：

```python
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
```

***

卷积神经网络中, 图像处理经过一次卷积得出的特征图的大小计算为:

长度: $h_2=\frac{H_1-F_H-2P}{S} + 1$
宽度: $W_2=\frac{W_1-F_W-2P}{S} + 1$

这里$h_1,w_1$表示的是特征图的长宽, P表示的是padding的大小, S表示的是卷积核的步长

**Batch和Epoch有什么区别？**
Batch大小是在更新模型之前处理的多个样本。Epoch数是通过训练数据集的完整传递次数。批处理的大小必须大于或等于1且小于或等于训练数据集中的样本数。可以将epoch设置为1和无穷大之间的整数值。您可以根据需要运行算法，甚至可以使用除固定数量的epoch之外的其他条件来停止算法，例如模型错误随时间的变化（或缺少变化）。它们都是整数值，并且它们都是学习算法的超参数，例如学习过程的参数，而不是学习过程找到的内部模型参数。您必须为学习算法指定batch大小和epoch数。如何配置这些参数没有固定的规则。您必须尝试不同的值，看看哪种方法最适合您的问题。

**实例**
最后，让我们用一个小例子来说明这一点。

> 假设您有一个包含200个样本（数据行）的数据集，并且您选择的Batch大小为5和1,000个Epoch。
>
> 这意味着数据集将分为40个Batch，每个Batch有5个样本。每批五个样品后，模型权重将更新。
>
> 这也意味着一个epoch将涉及40个Batch或40个模型更新。
>
> 有1000个Epoch，模型将暴露或传递整个数据集1,000次。在整个培训过程中，总共有40,000Batch。
