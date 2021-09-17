
import numpy as np
import pandas as pd
import torch
import scipy
import matplotlib as mpl
import sklearn
from mxnet import nd
import os

print('------------------torch基础使用---------------------------')
print(torch.rand(5, 3))

print('--------------------行向量，变形-------------------------')
# 【1】行向量，向量是张量的一种
# x为一个行向量，其中包含从0开始的12个连续整数
x = nd.arange(12)
# 输出为<NDArray 12 @cpu(0)>，是长度为12的一维数组，且被创建在cpu使用的内存上
print(x)
print(x.shape)
print(x.size)
# 改变该向量的形状为（3，4），即3*4的矩阵，且其中元素不变，输出为<NDArray 3x4 @cpu(0)>
x = x.reshape((3, 4))
print(x)
# 由于x的元素个数已知，因此-1可以借助其和其他维度的大小推断出来 ps：只能写-1，-2则会报错
print(x.reshape(-1, 6))
print(x.reshape(12, -1))

print('----------------张量，多个张量操作-----------------------------')
# 【2】张量
#  多个张量连接在一起
tensor1 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
tensor2 = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# dim=k 即表示在第k维合并，如k=0，代表行合并，k=1，代表列合并
print(torch.cat((tensor1, tensor2), dim=0))
print(torch.cat((tensor1, tensor2), dim=1))
print(tensor1 == tensor2)
print(tensor1.sum())
# 广播机制：形状不同第两个张量操作,前提是两者第行列至少存在一样相同
# 具体操作：两者行最大为3，列数最大为2，则均转换成3*2张量 其中a把第一列复制到第二列，b吧第一行复制到第二行和第三行
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)

# 元素访问
print('----------------张量的元素访问-----------------------------')
elementAccess = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# -1表示最后一个元素
print(elementAccess[-1])
# 1：3 表示从第一个到第三个，其中第三个不归于结果中
print(elementAccess[1:3])
elementAccess[1, 2] = 9
print(elementAccess)
elementAccess[0:2, :] = 12
print(elementAccess)

print('----------------数据预处理-----------------------------')
# 创建一个人工数据集，并存储在csv文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行代表一个数据样本
    f.write('2,Street,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)
print('----------------处理缺失/错误的数据-----------------------------')
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# fillna 取所有值的均值
inputs = inputs.fillna(inputs.mean())
print(inputs)
print('----------------对于的类别值或离散值，将NaN视为一个类别-----------------------------')
inputs1 = pd.get_dummies(inputs, dummy_na=True)
print(inputs1)
# inputs和outputs都是数值类型，因此可以将其转化为张量格式
print('----------------转化为张量格式-----------------------------')
print(torch.tensor(inputs1.values))
print(torch.tensor(outputs.values))


