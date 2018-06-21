#coding:utf-8
import numpy as np

# N是batch大小；D_in是输入维度，H是隐藏维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建任意输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
# 随机初始化weight
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：计算预测的y
    h = x.dot(w1) # x:64*1000 w1:1000*100 h:64*100
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2) # h_relu:64*100 w2:100*10 y_pred:64*10

    # 计算打印loss
    loss = np.square(y_pred - y).sum()
    print t, loss

    # 反向传播，根据loss计算w1和w2的gradient
    grad_y_pred = 2.0 * (y_pred - y) # grad_y_pred:64*10
    grad_w2 = h_relu.T.dot(grad_y_pred) # h_relu.T:100*64 grad_w2:100*10
    grad_h_relu = grad_y_pred.dot(w2.T) # w2.T:10*100 grad_h_relu:64*100
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0 # grad_h:64*100
    grad_w1 = x.T.dot(grad_h) # x.T:1000*64 grad_w1:1000*100
    # 更新weight
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
