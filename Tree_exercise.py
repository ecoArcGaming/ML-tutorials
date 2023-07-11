from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn.metrics import hinge_loss



data = pd.read_csv("creditcard.csv")
print(data.describe())

big_data = pd.DataFrame(np.repeat(data.values, 10, axis = 0), columns = data.columns)
print(big_data.shape)

labels = big_data.Class.unique()

sizes = big_data.Class.value_counts().values
print(sizes)

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.figure()
plt.hist(big_data.Amount, range = [0, 500])

print("90th percentile:", np.percentile(big_data.Amount.values, 90))


big_data.iloc[:,1:30] = StandardScaler().fit_transform(big_data.iloc[:,1:30])
print(big_data)
X = big_data.values[:,1:30]
y = big_data.values[:,30]
X = normalize(X, norm ="l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       
print(y_train.shape)

weight = compute_sample_weight('balanced',y_train)
print(weight)

Tree = DecisionTreeClassifier(max_depth = 4, random_state = 42)
t0 = time.time()
Tree.fit(X_train, y_train, sample_weight = weight)
t1 = time.time() - t0
print("time elapsed: %.2f" % t1)

y_predict = Tree.predict(X_test) 
tree.plot_tree(Tree, fontsize = 12)
score = roc_auc_score(y_test, y_predict)
print("ROC_AUC score: ", score)

model  = LinearSVC(class_weight='balanced',random_state=31,fit_intercept=False)
t0 = time.time()
model.fit(X_train, y_train)
t1 - time.time() - t0
print("time elapsed:", t1)

y_predict = model.decision_function(X_test)
loss= hinge_loss(y_test, y_predict)
print("model hinge loss:", loss)

