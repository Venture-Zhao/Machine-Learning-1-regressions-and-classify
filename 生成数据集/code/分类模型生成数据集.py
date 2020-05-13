import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

# X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2（对应x、y轴）个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X1, Y1 = make_classification(n_samples=2000, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2)

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()

X=np.mat(X1)
y=np.mat(Y1)
data=np.concatenate((X,y.T),axis=1)
np.savetxt('../data/二分类模型.csv', data, delimiter = ',')