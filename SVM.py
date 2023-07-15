import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm


data = pd.read_csv("cell_samples.csv")
print(data.columns)

# print(X)

y = data['Class'].values.astype('int')

ax = data[data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
data[data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
# plt.show()

print(data.dtypes)
data = data[pd.to_numeric(data['BareNuc'], errors='coerce').notnull()]

data['BareNuc'] = data['BareNuc'].astype(int)
X = data[['ID', 'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y[0:683], test_size=0.2, random_state=4)

model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print(y_predict)

from sklearn.metrics import f1_score
score = f1_score(y_test, y_predict, average='weighted') 
print("f1 score", score)