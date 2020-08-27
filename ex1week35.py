# Ex. 1 week 35

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/mhjensen/MachineLearningMSU-FRIB2020/master/doc/pub/Regression/ipynb/datafiles/EoS.csv'
EoS = pd.read_csv(url, names=('Density','Energy'))
df = pd.DataFrame(EoS)
print(df)

E = df['Energy']
X = np.zeros((len(E),4))

for i in range(4):
    X[:,i] = E**i

print(X)

DM = pd.DataFrame(X)
DM.columns = ('1', 'p','p^2', 'p^3')

print(DM)
