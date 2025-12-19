import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from ts2vec import TS2Vec

window = 300
step = 300

n_neighbors = 10
idx = 500

# min = pd.read_csv('data/btcusd.csv').to_numpy()
min = np.load('data/btc_test.npy')
log_ret = np.tanh(np.log(min[1:, 41]/min[:-1, 41])/0.0005)
skip = len(min) // 2

# plt.plot(min)
# plt.title("BTC-USD 1-Minute Close Prices")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.show()

fhr_index = [[i+j for j in range(window)] for i in range(skip + window, len(min)-window, step)]
phr_index = [[i+j for j in range(window)] for i in range(skip, len(min)-2*window, step)]

fhr = min[:, 41][fhr_index]
phr = min[phr_index]

fhr = np.log(fhr) - np.log(fhr[:, 0][:, None])

min_vals = phr.min(axis=0, keepdims=True)
max_vals = phr.max(axis=0, keepdims=True)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1
phr = (phr - min_vals) / range_vals

model = TS2Vec(
    input_dims=phr.shape[-1],
    output_dims=384,
    device='mps',
)
model.load('encoder_models/checkpoints/DiT-S.pkl')

repr = model.encode(phr, encoding_window='full_series').cpu().numpy()
print("Representation shape:", repr.shape)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(repr.reshape(repr.shape[0], -1))
explained_variance = pca.explained_variance_ratio_.sum()
print(f'PCA explained variance (2 components): {explained_variance:.4f}')

knn = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan')
knn.fit(pca_result)
distances, indices = knn.kneighbors(pca_result)
print("Indices shape:", indices.shape)
print("Distances shape:", distances.shape)

freturns = fhr[indices[idx, :]]

plt.style.use('Solarize_Light2')
plt.plot(freturns.T, color='blue', alpha=0.3)
plt.title("KNN Future Log Returns")
plt.xlabel("Time")
plt.ylabel("Log Return")
plt.show()


# fhr = fhr - fhr[:, 0][:, None]
# phr = phr - phr[:, 0][:, None]

# fhr_min = fhr.min(axis=1, keepdims=True)
# fhr_max = fhr.max(axis=1, keepdims=True)
# fhr_norm = (fhr - fhr_min) / (fhr_max - fhr_min + 1e-8)
# fhr_slopes = fhr_norm[:, -1] - fhr_norm[:, 0]

# phr_min = phr.min(axis=1, keepdims=True)
# phr_max = phr.max(axis=1, keepdims=True)
# phr_norm = (phr - phr_min) / (phr_max - phr_min + 1e-8)
# phr_slopes = phr_norm[:, -1] - phr_norm[:, 0]

# idx = np.arange(len(fhr_norm))/len(fhr_norm)
# fhr_raw = np.concatenate([idx[:, None], fhr_slopes[:, None]], axis=1)
# phr_raw = np.concatenate([idx[:, None], phr_slopes[:, None]], axis=1)

# pca = PCA(n_components=1)
# f_pca = pca.fit_transform(fhr)
# p_pca = pca.fit_transform(repr.reshape(repr.shape[0], -1))

# print(fhr_slopes.shape, phr_slopes.shape)
# print(fhr_raw.shape, phr_raw.shape)
# print(f_pca.shape, p_pca.shape)

# plt.scatter(p_pca, f_pca, s=5, alpha=0.5)
# plt.xlabel("PHR")
# plt.ylabel("FHR")
# plt.title("2D Visualization of past vs future log returns using PCA")
# plt.tight_layout()
# plt.show()
# exit()

# data = np.vstack([p_pca, f_pca]).T
# print(data.shape)
# np.save('manim/stock_scatter_data.npy', data)

# sns.kdeplot(
#     x=p_pca, y=f_pca,
#     fill=True, thresh=0, levels=10, cmap="mako",
# )
# plt.xlabel("PHR Slopes")
# plt.ylabel("FHR Slopes")
# plt.title("2D Density Plot of FHR and PHR Slopes")
# plt.tight_layout()
# plt.show()


# --- New code to get (x, y, z) coordinates from KDE ---
# data_for_kde = np.vstack([p_pca.T, f_pca.T])

# kde = gaussian_kde(data_for_kde)

# grid_x = np.linspace(p_pca.min(), p_pca.max(), 100)
# grid_y = np.linspace(f_pca.min(), f_pca.max(), 100)
# X, Y = np.meshgrid(grid_x, grid_y)
# positions = np.vstack([X.ravel(), Y.ravel()])

# Z = np.reshape(kde(positions).T, X.shape)

# plt.style.use('Solarize_Light2')
# plt.contourf(X, Y, Z, cmap='mako', levels=10, fill=False)
# plt.colorbar(label='Density')
# plt.title("2D KDE of Past and Future Log Return Slopes")
# plt.xlabel("Past Trendlines")
# plt.ylabel("Future Trendlines")
# plt.tight_layout()
# plt.show()

# xyz_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
# print("Shape of (x, y, z) points array:", xyz_points.shape)
# print("First 5 (x, y, z) points:")
# print(xyz_points[:5])

# np.save('manim/joint_prob_xyz_points.npy', xyz_points)