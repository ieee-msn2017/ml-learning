#coding:utf-8
import torch

class MyReLU(torch.autograd.Function):
    """
    通过继承Function来实现自定义autograd函数
    """
    @staticmethod
    def forward(ctx, input):
        """
        在前向传播中，我们接收一个Tensor包含输入，返回一个Tensor包含输出。ctx是一个上下文对象，
        可以用来存放反向计算的信息，可以用ctx.save_for_backward方法缓存任意在反向传播中用到的对象
        """
        ctx.save_for_backward(input) # input是x.mm(w1)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播中，我们接收一个Tensor包含loss对于输出的梯度，需要计算loss对于输入的梯度
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.float

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, dtype=dtype)
y = torch.randn(N, D_out, dtype=dtype)

w1 = torch.randn(D_in, H, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2):
    relu = MyReLU.apply

    y_pred = relu(x.mm(w1)).mm(w2)
    # loss是一个shape(1,)的Tensor
    loss = (y_pred - y).pow(2).sum()
    # print t, loss.item()
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 在更新weight后，手动将梯度归零
        w1.grad.zero_()
        w2.grad.zero_()
