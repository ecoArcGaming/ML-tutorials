import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from scipy import stats
from scipy.stats import norm,skew
pd.set_option("display.max_rows", 1000)

set = datasets.load_diabetes()
x = set.data 
y= set.target
chart = pd.DataFrame(x, columns=set.feature_names)
print(set.DESCR)
print(chart.describe(None))
y = chart['age']

fig, ax = plt.subplots()

ax.set_xlabel("bmi")
ax.set_ylabel('blood sugar')

sns.distplot(np.log1p(chart['age']), fit=norm, label='norm fit')
print(norm.fit(chart['age']))
plt.legend()
fig = plt.figure()
stats.probplot(chart['bmi'],plot=plt)

plt.figure()
cor = chart.corr()
sns.heatmap(chart.corr(), annot=True,cmap=plt.cm.PuBu)
cor_target=abs(cor['bmi'])

from sklearn.model_selection import train_test_split
X = chart.drop('bmi', axis= 1)
Y = chart['bmi']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression 

lr = LinearRegression() 
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)  

print("Actual Age:- ", y_test[0]) 
print("Model Predicted Age:- ", predictions[0])

#number of features = dimension, independent, uppper X
#labels = output, dependent, lower y
#instances = number of rows of each feature
#supervised = input is labeled, know what output is
#regression = output is continuous
#classification = output is binary/discrete
#overfitting = model performs poorly due to too many features on training data 
#fails to generalize on testing data

#