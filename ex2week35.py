#ex 2 week 35

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split

x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)


X = np.zeros((len(x),3))

for i in range(3):
    X[:,i] = x.T**i

print(X)

DM = pd.DataFrame(X)
DM.columns = ('1', 'x','x^2')

print(DM)

beta = np.linalg.inv(X.T@X)@X.T@y

print(beta)

clf = skl.LinearRegression().fit(X,y)
print('SKL-coefficiants:', clf.coef_)

ytilde = clf.predict(X)
print(ytilde)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

newbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

newytilde = X_train @ newbeta
print("Training R2")
print(R2(y_train,newytilde))
print("Training MSE")
print(MSE(y_train,newytilde))
ypredict = X_test @ newbeta
print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))

x2 = np.linspace(0,1,100)

plt.scatter(x,y, color='m', s=1)
plt.plot(x, X@beta)
plt.show()
