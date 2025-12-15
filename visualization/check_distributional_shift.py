import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

window = 60
step = 60

min = pd.read_csv('data/btcusd_1-min_data.csv').to_numpy()
skip = len(min) // 2

fhr_index = [[i+j for j in range(window)] for i in range(skip + window, len(min)-window, step)]
phr_index = [[i+j for j in range(window)] for i in range(skip, len(min)-2*window, step)]

fhr = min[fhr_index, 1:]
phr = min[phr_index, 1:]

fhr = fhr - fhr[:, 0, 4][:, None, None]
phr = phr - phr[:, 0, 4][:, None, None]

fhr = fhr[:, :, 1]
phr = phr[:, :, 1]

fhr_min = fhr.min(axis=1, keepdims=True)
fhr_max = fhr.max(axis=1, keepdims=True)
fhr_norm = (fhr - fhr_min) / (fhr_max - fhr_min + 1e-8)
fhr_slopes = fhr_norm[:, -1] - fhr_norm[:, 0]

phr_min = phr.min(axis=1, keepdims=True)
phr_max = phr.max(axis=1, keepdims=True)
phr_norm = (phr - phr_min) / (phr_max - phr_min + 1e-8)
phr_slopes = phr_norm[:, -1] - phr_norm[:, 0]

grid_x = np.linspace(phr_slopes.min(), phr_slopes.max(), 100)
grid_y = np.linspace(fhr_slopes.min(), fhr_slopes.max(), 100)
X, Y = np.meshgrid(grid_x, grid_y)
positions = np.vstack([X.ravel(), Y.ravel()])

Z_list = []
for i in range(84):
    data_for_kde = np.vstack([phr_slopes[i*720:(i+1)*720], fhr_slopes[i*720:(i+1)*720]])
    kde = gaussian_kde(data_for_kde)

    Z = np.reshape(kde(positions).T, X.shape)
    Z = Z / Z.sum()
    Z_list.append(Z)

# heatmap of JS divergences
js_matrix = np.zeros((len(Z_list), len(Z_list)))
for i in range(len(Z_list)):
    for j in range(len(Z_list)):
        js_div = jensenshannon(Z_list[i].ravel(), Z_list[j].ravel(), base=2)
        js_matrix[i, j] = js_div

plt.imshow(js_matrix, cmap='viridis')
plt.colorbar(label='Jensen-Shannon Divergence')
plt.title('Heatmap of Jensen-Shannon Divergences between Months')
plt.tight_layout()
plt.show()

import matplotlib.animation as animation

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.contourf(X, Y, Z_list[frame], cmap='plasma', levels=10)
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_xlabel("Past Trendlines")
    ax.set_ylabel("Future Trendlines")
    ax.set_title(f'2D KDE of Past vs Future Slopes – Month {frame+1}')

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(Z_list),
    interval=50,
    blit=False
)

plt.show()
