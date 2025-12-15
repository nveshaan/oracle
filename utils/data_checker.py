import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/btcusd_1-min_data.csv').to_numpy()[:, 0]
time = data[0]
gaps = []

for t in data:
    if t - time > 60:
        gaps.append((time, t))
    time = t

print(f"Gaps found: {gaps}")

plt.hist([end - start for start, end in gaps], bins=50)
plt.xlabel("Gap Duration (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of Time Gaps in BTC/USD 1-Min Data")
plt.tight_layout()
plt.show()