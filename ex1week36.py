# ex 1 week 36

"""
a)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

X = np.ones((len(x),3))

for i in range(2):
    X[:,i+1] = x.T**(i+1)

def bridge(l,X):
    bridge = []
    Beta0 = np.array([np.mean(y)])
    for i in l:
        Beta12 = np.linalg.inv(X.T @ X + i*np.eye(2)) @ (X.T@y)
        Beta = np.concatenate((Beta0,Beta12))
        bridge.append(Beta)
    return bridge

def RSS(l, bridge,X):
    rsslist = []
    for i, k in enumerate(l):
        b = bridge[i][1:]
        rsslist.append(((y-X @ b).T @ (y-X @ b)) + k*(b.T@b))
    return rsslist

l = np.logspace(-4, 1, 20)
rsslist = RSS(np.log10(l), bridge(l,X[:,1:]),X[:,1:])
plt.plot(l, rsslist, label='error')
plt.legend()
plt.show()


x2 = np.linspace(0,1,100)
X2 = np.zeros((len(x2),3))

for i in range(2):
    X2[:,i+1] = x2.T**(i+1)

plt.scatter(x,y, color='m', s=1)

for i in bridge(l, X2[:,1:]):
    plt.plot(x2, X2@i, label = f'beta = {i}')
plt.show()

"""
b)
"""
