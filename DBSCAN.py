import numpy as np
from sklearn.cluster import DBSCAN 
from sklearn.datasets._samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler

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

df = pd.read_csv('weather-stations20140101-20141231.csv')
print(df['Tm'])
df = df[df['Tm'].notnull()] # drop rows for which Tm has no value
df = df.reset_index(drop=True)
print(df.head(), df.shape)
plt.figure()
# set min and max lattitude and longtitude for map
minlong = -140
minlat = 40
maxlong = -50
maxlat = 65
df = df[(df['Long'] > minlong) & (df['Long'] < maxlong) & (df['Lat'] > minlat) & (df['Lat'] < maxlat)]
map = Basemap(projection ='merc',resolution = 'l', area_thresh=1000.0,
               llcrnrlon = minlong, llcrnrlat = minlat,
              urcrnrlon=maxlong, urcrnrlat=maxlat)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='white',alpha=0.3)
map.shadedrelief()
rcParams['figure.figsize'] = (14,10)


df['xm'] = np.asarray(df['Long'])
df['ym'] = np.asarray(df['Lat'])

# for i, j in df.iterrows():
    # map.plot(x.tolist(),y.tolist(),markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
# plt.show()

sklearn.utils.check_random_state(1000)
cluster_df = np.nan_to_num(df[['xm','ym','Tx','Tm','Tn']])
cluster_df = StandardScaler().fit_transform(cluster_df)

# DBSCAN model

db = DBSCAN(eps = 0.5, min_samples=10).fit(cluster_df)
core_sample_mask = np.zeros_like(db.labels_, dtype = bool)
core_sample_mask[db.core_sample_indices_] = True
labels = db.labels_
df['labels']=labels

print(len(set(labels))) # not the real num of clusters because it counts -1, which is outliers
print((len(set(labels)))-(1 if -1 in labels else 0)) # excludes outliers 

colors= plt.get_cmap('jet')(np.linspace(0.0, 1.0, 4))
print(df)
for cluster_number in set(labels):
    col = (([0.4,0.4,0.4] if cluster_number == -1 else colors[int(cluster_number)])) # set different colors for outliers and cluster points
    cluster_set = df[df['labels']== cluster_number]
    map.scatter(cluster_set.xm, cluster_set.ym, color = col, marker = 'o')
    if cluster_number != -1:  # plot centroids
        cenx = np.mean(cluster_set.xm)
        ceny = np.mean(cluster_set.ym)
        plt.text(cenx, ceny, str(cluster_number), fontsize = 25)
plt.show()