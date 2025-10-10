import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

window = 10
step = 5

n_neighbors = 100
idx = 100

scaler = StandardScaler()

min1 = pd.read_csv('data/8,15 8,29 2min.csv', header=2)
min2 = pd.read_csv('data/9,2 9,29 2min.csv', header=2)
min = pd.concat((min1, min2)).to_numpy()

fhr_index = [[i+j for j in range(window)] for i in range(window, len(min)-window, step)]
phr_index = [[i+j for j in range(window)] for i in range(0, len(min)-2*window, step)]

fhr = min[fhr_index, 1:]
phr = min[phr_index, 1:]

fhr = fhr - fhr[:, 0, 4][:, None, None]
phr = phr - phr[:, 0, 4][:, None, None]

fhr = fhr[:, :, 1]
phr = phr[:, :, 1]

fhr_min = fhr.min(axis=1, keepdims=True)
fhr_max = fhr.max(axis=1, keepdims=True)
fhr_norm = (fhr - fhr_min) / (fhr_max - fhr_min + 1e-8)

phr_min = phr.min(axis=1, keepdims=True)
phr_max = phr.max(axis=1, keepdims=True)
phr_norm = (phr - phr_min) / (phr_max - phr_min + 1e-8)

knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(phr_norm)

anchor = np.array([phr[idx]])
dist, indices = knn.kneighbors(anchor)

k_fhr = fhr_norm[indices[0]]

pca = PCA(n_components=1)
k_pca = pca.fit_transform(k_fhr)

sns.histplot(k_pca, bins=10, kde=True)
plt.show()