import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 画出回归线及数据散点图
def plot_figure(X, y, w):
    # 直线第一个坐标（x1，y1）
    x1 = 7
    y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
    # 直线第二个坐标（x2，y2）
    x2 = 66
    y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
    plt.plot([x1, x2], [y1, y2], 'r')
    for i in range(y.size):
        if y[i] == 1.00:
            plt.scatter(X[i, 1], X[i, 2], c='g')
        else:
            plt.scatter(X[i, 1], X[i, 2], c='b')
    plt.title('data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

#训练模型获取参数
def get_w(X, y):
    #初始w
    w = np.array([[-1.0],[1.0],[-1.0]])
    s = np.dot(X, w)
    predict_y = np.ones_like(y)  # 预测输出初始化
    loc_n_0 = np.where(s < 0)[0]  # 大于零索引下标
    loc_n_1 = np.where(s>=0)[0]
    predict_y[loc_n_0] = -1
    predict_y[loc_n_1] = 1
    # 第一个分类错误的点
    t = np.where(y != predict_y)[0][0]

    # 更新权重w
    w += y[t] * X[t, :].reshape((3, 1))
    for i in range(100):
        s = np.dot(X, w)
        predict_y = np.ones_like(y)
        loc_n = np.where(s < 0)[0]
        predict_y[loc_n] = -1
        num_fault = len(np.where(y != predict_y)[0])
        print('第%2d次更新，分类错误的点个数：%2d' % (i, num_fault))
        if num_fault == 0:
            break
        else:
            t = np.where(y != predict_y)[0][0]
            w += y[t] * X[t, :].reshape((3, 1))
    return w;

def main(data):
    #数据处理
    data.columns = ['X1', 'X2', 'target']  # 列索引值
    data.insert(0, 'X0', 1)  # 在data第1列后插入全是1的一列数

    # 数据准备
    X = data.iloc[:, :3].values
    y = data.iloc[:, 3].values

    #训练模型，获取w值
    w = get_w(X,y)

    #可视化图形
    plot_figure(X,y,w)

if __name__ == '__main__':
    csvpath = '..\..\生成数据集\data\感知机分类数据.csv'
    data = pd.read_csv(csvpath, header=None)
    main(data)