# ex 1 week 36

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split


x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

X = np.zeros((len(x),3))

for i in range(2):
    X[:,i] = x.T**(i+1)

def RSS(l):
    rsslist = []
    for i in l:
        bridge = (X@X.T+i*np.eye(3))**(-1)@X.T@y
        rsslist.append((y-X@beta)).T@(y-X@beta)) + i*bridge.T@bridge)
    return rsslist


l = [0,1,2,3,4,5,6]

rsslist = RSS(l)

plt.plot(l, rsslist)
plt.show()
