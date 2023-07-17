import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import pylab



X, y = make_blobs(n_samples=500, centers=[[5,6],[2,7],[7,4],[8,6]], cluster_std=0.7)

model = AgglomerativeClustering(n_clusters=4,linkage='complete')
model.fit(X,y)

fig = plt.figure()
X_min, X_max = np.min(X,axis = 0), np.max(X, axis = 0)
X = (X - X_min)/(X_max - X_min)

for i in range(X.shape[0]):
    plt.text(X[i,0],X[i,1], str(y[i]), color=plt.cm.nipy_spectral(model.labels_[i]/10.), fontdict={'weight':'bold','size':9})
plt.scatter(X[:,0],X[:,1], marker='o')

# plt.show()

matrix = distance_matrix(X,X)
print(matrix)

dendrogram = hierarchy.dendrogram(hierarchy.linkage(matrix,'average'))

data = pd.read_csv('cars_clus.csv')

data[['sales','resale','type','price','engine_s',
      'horsepow','wheelbas','width','length','curb_wgt',
      'fuel_cap','mpg','lnsales']]=data[['sales','resale',
                                         'type','price','engine_s','horsepow','wheelbas',
                                         'width','length','curb_wgt','fuel_cap','mpg',
                                         'lnsales']].apply(pd.to_numeric, errors = 'coerce')
data = data.dropna(axis = 0)
data = data.reset_index(drop=True)
print(data.head())

X_df = data[['engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg']]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_df)
print(X)

matrix = euclidean_distances(X, X)
print(matrix)
Z = hierarchy.linkage(matrix, 'complete')
plt.figure(figsize=(8,60))
def leaf_label(id):
    return '[%s %s %s]' % (data['manufact'][id], data['model'][id], int(float(data['type'][id])))
dendro = hierarchy.dendrogram(Z,  leaf_label_func=leaf_label, leaf_rotation=0, leaf_font_size =8, orientation = 'right')
plt.show()