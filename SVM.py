import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("cell_samples.csv")
print(data.columns)

# print(X)

y = data['Class'].values

ax = data[data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
data[data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
# plt.show()

print(data.dtypes)
data = data[pd.to_numeric(data['BareNuc'], errors='coerce').notnull()]

data['BareNuc'] = data['BareNuc'].astype(int)
X = data[['ID', 'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values
print(X)