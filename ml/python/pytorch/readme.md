# Pytorch

- forcast 温度预测. 神经网络预测值. 简单实现的神经网络偏导向后传播. 没有测试集切分, 没有复杂的特征工程
- handwriting手写数字的识别, 神经网络预测.
  - 文件的简单io, gzip的处理
  - map方法的执行和解构操作
  - modal包和function包的区别和简单介绍
  - 神经网络的模板.
  - torch.no_grad() 打印损失函数
  - TensorDataset 和 DataLoader

pytorch 的类型一般的属性可以直接替换 `torch.FloatTensor`,`torch.IntTensor`,`torch.LongTensor`. 这些tensor都是cpu的类型, 在换从中都有一个标记

```python
### 0纬标量
a = torch.tensor(2,2)
a.shape # torch.size([])

### 一维标量(张量)
torch.tensor([1.1]) # tensor([1.1000])

torch.tensor([1.1, 2.2]) # 一维的张亮
torch.FloatTensor(1) # 随机出是一个长度为1 的向量
torch.FloatTensor(2) # tensor([3.123e-25, 4.5935e-41])

data=np.ones(2)
torch.from_numpy(data) # 将numpy转换为torch的tensor

## 获取长度
a = torch.ones(2)
a.shape # 获取形状 torch.size([2])

a = torch.randn(2,3) # 2行3列的数据
a.shape # torch.size([2,3])

a.size(0) # 2
a.size(1) # 3
a.shape[1] # 3

## 基本操作

torch.rand(1,2,3)

```
