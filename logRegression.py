import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import jaccard_score
import itertools



data = pd.read_csv("ChurnData.csv")
print(data.columns)

X = data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
       'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
       'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
       'lninc', 'custcat']].values
print(X)
y = data['churn'].values
print(y)

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)

print(X_train.shape)

lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
y_predict_proba = lr.predict_proba(X_test)
y_predict = lr.predict(X_test)
print(y_predict_proba) # first column = P(class 0), second column = P (class 1)

j_score = jaccard_score(y_test, y_predict, pos_label=0) # intersection over union, doesn't work with predict_proba
print("jaccard score of logistic model: ", j_score)

cm = confusion_matrix(y_test, y_predict, labels=[1,0])

def plot_confusion_matrix(matrix, classes):
    print(matrix)
    plt.imshow(matrix, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.colorbar()
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i,j]))

plot_confusion_matrix(cm, classes = ['1', '0'])
plt.show()
print (classification_report(y_test, y_predict))
