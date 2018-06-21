#coding:utf-8
import torch

dtype = torch.float

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, dtype=dtype)
y = torch.randn(N, D_out, dtype=dtype)

w1 = torch.randn(D_in, H, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 这里没有记录中间的隐藏层数据
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # loss是一个shape(1,)的Tensor
    loss = (y_pred - y).pow(2).sum()
    print t, loss.item()
    # 用autograd计算反向传播，这里会根据所有设置了requires_grad=True的Tensor计算loss的梯度，
    # w1.grad和w2.grad将会保存loss对于w1和w2的梯度
    loss.backward()
    # 手动更新weight，需要用torch.no_grad()，因为weight有required_grad=True，但我们不需要在
    # autograd中跟踪这个操作
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 在更新weight后，手动将梯度归零
        w1.grad.zero_()
        w2.grad.zero_()
