import numpy as np
from sklearn.cluster import DBSCAN 
from sklearn.datasets._samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_data (centroid, num_samples, cluster_deviation):
    X, y = make_blobs(n_samples = num_samples, centers = centroid, cluster_std=cluster_deviation)
    X = StandardScaler().fit_transform(X)
    return X, y

X, y = generate_data([[1,2],[3,5],[9,2]], 1500, 0.7)
# print(X)

r = 0.2
m = 5
model = DBSCAN(eps = r, min_samples=m).fit(X)
labels = model.labels_
# plt.scatter(X[:,0], X[:,1])
# plt.show()
print(labels)

core_sampels_mask = np.zeros_like(labels, dtype=bool)
core_sampels_mask[model.core_sample_indices_] = True
print(core_sampels_mask)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # noisy samples are labelled -1
print(n_clusters, set(labels)) # set() removes repetiton
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels))) #2 is the num of unique labels
for k, col in zip(unique_labels, colors):
    if k ==-1:
        col = 'k' # noise = black
    if_in_class = (labels == k)
    xy = X[if_in_class & core_sampels_mask] # either  true means in cluster
    plt.scatter(xy[:,0],xy[:,1], s=50, c=[col],alpha=0.5)
    xy=X[if_in_class & ~core_sampels_mask] # either false
    plt.scatter(xy[:,0],xy[:,1], s=50, c=[col],alpha=0.5)

plt.show()