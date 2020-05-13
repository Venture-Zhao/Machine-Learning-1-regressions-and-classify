import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#画出回归线及数据散点图
def plot_figure(X,y):
    for i in range(y.size):
        if y[i] == 1.00:
             plt.scatter(X[i, 1], X[i, 2],c='r')
        else:
            plt.scatter(X[i, 1], X[i, 2], c='b')
    plt.title('data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show();

#分类器
def classfy(y):
    for i in range(y.size):
        if y[i]<0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y

#计算准确度
def cal_accuracy(test_y, predict_y):
    T_num = 0;#预测正确的数量
    print(test_y)
    print(predict_y)
    for i in range(test_y.size):
        if test_y[i]==predict_y[i]:
            T_num = T_num + 1
    return  T_num/test_y.size

def main(data):
    #数据处理
    data.columns = ['X1', 'X2', 'target']  # 列索引值
    data.insert(0, 'X0', 1)  # 在data第1列后插入全是1的一列数

    #测试集
    data_train = data[:1500]
    train_X = np.mat(data_train.iloc[:, [0, 1, 2]])
    train_y = np.mat(data_train.iloc[:, [3]])

    #训练集
    data_test = data[1500:]
    test_X = np.mat(data_test.iloc[:, [0, 1, 2]])
    test_y = np.mat(data_test.iloc[:, [3]])

    # 画出训练集散点图
    #plot_figure(train_X, train_y)

    #用训练集对模型进行训练
    # 方法：正规方程组
    w = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T, train_X)), train_X.T), train_y)

    #根据权重预测y值，并将y值映射到sigmoid函数上
    predict_y= test_X * w
    g_z = classfy(predict_y)

    # 画出测试集散点图
    plot_figure(test_X, test_y)

    # 画出预测的散点图
    plot_figure(test_X, g_z)

    #计算预测准确度并打印
    accuracy = cal_accuracy(test_y, g_z)
    print(accuracy)

if __name__ == '__main__':
    csvpath = '..\..\生成数据集\data\二分类模型.csv'
    data = pd.read_csv(csvpath, header=None)
    main(data)