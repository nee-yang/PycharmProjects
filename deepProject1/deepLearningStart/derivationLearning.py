import torch

x = torch.arange(4.0)

# 在计算y关于x的梯度之前，需要一个地方来存储梯度
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=true)
print(x.grad)  # 默认值为none

y = 2 * torch.dot(x, x)  # dot为计算内积
print(y)
print('----------------调用反向传播函数来自动计算y关于x每个分量的梯度-----------------------------')
print(y.backward())
print(x.grad)
print(x.grad == x * 4)

print('----------------默认情况下，pytorch会累计梯度，因此需要清除之前的值，若不清除前后结果相加----------------------------')
print(x.grad)
x.grad.zero_()
print(x.grad)
y = x.sum()
y.backward()
print(x.grad)

print('----------------对非标量调用backwardxu需要传入一个gradient参数，该参数指定微分函数关于self的函数----------------------------')
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

print('----------------将某些计算移动到记录的计算图之外----------------------------')
x.grad.zero_()
y = x * x
u = y.detach()  # 把u看作常数
z = u * x
z.sum().backward()
print(x.grad)
print(x.grad == u)