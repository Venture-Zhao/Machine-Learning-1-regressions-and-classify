import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#定义代价函数对应的梯度函数
def gradient_function(X, y, w):
    diff = np.dot(X, w) - y
    return (1 / y.size) * np.dot(X.transpose(), diff)

#梯度下降法求权重
def gradient_descent(X, y, w):
    tol = 0.00001#阈值
    a = 0.1#步长
    gradient = gradient_function(X, y, w)
    while np.linalg.norm(gradient) >= tol:
        w = w - a * gradient
        gradient = gradient_function(X, y, w)
    return w

#画出回归线及数据散点图
def plot_figure(X, y, w):
    x_1 = np.linspace(-4, 4, 50)
    y_1 = w[1,0] * x_1 + w[0,0]
    plt.scatter(np.array(X[:, [1]]), np.array(y))
    plt.plot(x_1, y_1, '-r', label='y=' + str(w[1, 0]) + 'x+' + str(w[0, 0]))
    plt.xlabel('train_X')
    plt.ylabel('train_y')
    plt.show();

def main(data):
    #数据处理
    data.columns = ['X1', 'y']  # 列索引值
    data.insert(0, 'X0', 1)  # 在data第1列后插入全是1的一列数

    #y=X*w^T
    X = np.mat(data.iloc[:1500, [0, 1]])
    y = np.mat(data.iloc[:1500,[2]])

    #训练学习模型
    #方法一：正规方程组
    w1 =  np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    #方法二：梯度下降
    w2 = np.mat([[0], [40]])
    w2 = gradient_descent(X , y , w2)

    #画出回归线
    plot_figure(X, y, w1)#正规方程组
    plot_figure(X, y, w2)#梯度下降法

    #参数
    print("正规方程组解法：")
    print(w1)
    print("梯度下降解法：")
    print(w2)

if __name__ == '__main__':
    csvpath = '..\..\生成数据集\data\回归模型.csv'
    data = pd.read_csv(csvpath, header=None)
    main(data)