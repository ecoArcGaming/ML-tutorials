import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# change 
data = pd.read_csv("teleCust1000t.csv")
# print(data.describe())

print(data.custcat.value_counts())


print(data.columns)
y = data.custcat.values
print(y[0:5])
X = data[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside',]].values
print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# normalize
X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test)

KKN = KNeighborsClassifier(n_neighbors=4).fit(X_train_norm,y_train)
y_predict = KKN.predict(X_test_norm)
print(y_predict[0:5])
print("accuracy score: %.2f" %  metrics.accuracy_score(y_test,y_predict))

max = 10 
mean_acc = np.zeros(max-1)
for n in range(1,max):
    KKN = KNeighborsClassifier(n_neighbors=n).fit(X_train_norm,y_train)
    y_predict = KKN.predict(X_test_norm)
    mean_acc[n-1]=metrics.accuracy_score(y_test,y_predict)

plt.plot(range(1,max),mean_acc,'g')
plt.yticks(np.arange(0,0.45,0.05))
plt.xlabel('k-value')
plt.ylabel('accuracy score')
plt.show()
