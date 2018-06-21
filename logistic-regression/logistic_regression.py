#coding:utf-8

import numpy as np
import os

training_sample = 'Logistic_Regression-trainingSample.txt'
testing_sample = 'Logistic_Regression-testingSample.txt'

# 从文件中读入训练样本的数据
def loadDataSet(filepath):
    dataMat = []
    labelMat = []
    f = open(filepath)
    for line in f.readlines():
        lineArr = line.strip().split()
        # 三个特征x0, x1, x2, x0=1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))  # 样本标签y
    return dataMat, labelMat

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

# 梯度下降法求回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)             # 转换成numpy中的矩阵, X, 90 x 3
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy中的矩阵, y, 90 x 1
    m, n = np.shape(dataMatrix)  # m=90, n=3
    lr = 0.001  # 学习率
    maxCycles = 1000
    weights = np.ones((n, 1))  # 初始参数, 3 x 1
    for k in range(maxCycles):              # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)     # 模型预测值, 90 x 1
        error = h - labelMat              # 真实值与预测值之间的误差, 90 x 1
        temp = dataMatrix.transpose() * error  # 所有参数的偏导数, 3 x 1
        weights = weights - lr * temp  # 更新权重
    return weights

def predict(y):
    preds = []
    for i in y:
        if i > 0.5:
            preds.append(1)
        else:
            preds.append(0)
    return preds
# 训练
dataArr, labelMat = loadDataSet(training_sample)  # 读入训练样本中的原始数据
A = gradAscent(dataArr, labelMat)  # 回归系数a的值
# 测试
dataArr, labelMat = loadDataSet(testing_sample)
h = sigmoid(np.mat(dataArr)*A)  # 预测结果h(a)的值
print 'real: '
print labelMat
print 'predict: '
print predict(h)
