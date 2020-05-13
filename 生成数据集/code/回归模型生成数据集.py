import numpy as np
from sklearn.datasets import make_regression
from matplotlib import pyplot
# generate regression dataset
X, y, coef = make_regression(n_samples=2000, n_features=1, noise=16,coef=True)

print(coef)
# plot regression dataset
pyplot.scatter(X,y)
pyplot.show()
print(X,y)

# save data
X=np.mat(X)
y=np.mat(y)
data=np.concatenate((X,y.T),axis=1)
print(data)
np.savetxt('../data/回归模型.csv', data, delimiter = ',')