import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.random.seed()

X, y = make_blobs(n_samples=5000, centers=[[3, 5],[-1,3],[6,2]], cluster_std=0.9)
# print(X)
plt.scatter(X[:,0],X[:,1],marker='.')
# plt.show()
k_means = KMeans(init='random',n_clusters=3, n_init=20)

k_means.fit(X)
labels = k_means.labels_
# print(labels)
centers = k_means.cluster_centers_
# print(centers)

fig = plt.figure(figsize=(6,4))
colors = plt.cm.Spectral(np.linspace(0,1,len(set(labels))))
ax = fig.add_subplot(1,1,1)

# print(colors)
for (k,col) in zip(range(len([[3, 5],[-1,3],[6,2]])), colors):
    members = (labels == k)
    center = centers[k]
    ax.plot(X[members, 0], X[members, 1],'w', markerfacecolor = col, marker = '.')
    ax.plot(center[0], center[1], 'o', markerfacecolor = col, markeredgecolor= 'k', markersize  = 6)

ax.set_title('KMeans')


df = pd.read_csv('Cust_Segmentation.csv')
df = df.drop('Address',axis=1)
# standardizing
data = np.nan_to_num(df)
X = data[:,1:]
X_norm = StandardScaler().fit_transform(X)
# print(X_norm)

k = 3
model = KMeans(init = 'k-means++',n_clusters=k, n_init = 20)
model.fit(X_norm)
labels = model.labels_

df['Cluster'] = labels # add labels to dataframe
print(df.head())
print(df.groupby('Cluster').mean())

plt.figure()
area = np.pi*(X[:,1])**2
# print(area)
plt.scatter(X[:,0],X[:,3],s =area,c=labels, alpha = 0.5)
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()