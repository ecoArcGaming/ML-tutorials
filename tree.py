import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import metrics 

data = pd.read_csv("drug200.csv")
# print(data.head)
# print(data.shape)
X = data[['Age','Sex','BP','Cholesterol','Na_to_K']].to_numpy()
y = data.Drug.to_numpy()

enc = preprocessing.LabelEncoder()
enc.fit(['F','M'])
X[:,1] = enc.transform(X[:,1]) # female = 0, male = 1

enc1 = preprocessing.LabelEncoder()
enc1.fit(['LOW','NORMAL','HIGH'])
X[:,2] = enc1.transform(X[:,2]) # low = 0, normal = 1, high = 2


enc2 = preprocessing.LabelEncoder()
enc2.fit(['NORMAL','HIGH'])
X[:,3]=enc2.transform(X[:,3])


print(X[:5],y[:5])
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
print(X_train.shape, y_train.shape)

Tree = DecisionTreeClassifier(criterion='entropy',max_depth=4)
Tree.fit(X_train,y_train)
y_predict = Tree.predict(x_test)
print(y_predict[:5])
print(y_test[:5])
score = metrics.accuracy_score(y_test, y_predict)
print("Accuracy Score of the Tree: ", score)

tree.plot_tree(Tree)
plt.show()