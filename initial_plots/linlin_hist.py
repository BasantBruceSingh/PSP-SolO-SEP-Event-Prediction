import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SolO_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/Jlinlin_SolO.csv"
df_SolO = pd.read_csv(SolO_csv, parse_dates=["time"])

fig, ax = plt.subplots(figsize=(10,6), ncols=2, nrows=1)

mask = df_SolO["linlin"] > 0
x = df_SolO["linlin"][mask]

x_bins = np.logspace(np.log10(min(x)), np.log10(max(x)), 50)

ax[0].hist(x, bins=x_bins)
ax[0].set_xscale("log")
ax[0].set_xlabel("linlin")
ax[0].set_ylabel("Count")
ax[0].set_title("Distribution of linlin values in SolO dataset")

mask = df_SolO["linlin"] < 100
x = df_SolO["linlin"][mask]
ax[1].hist(x, bins=50)
ax[1].set_xlabel("linlin")
ax[1].set_ylabel("Count")
ax[1].set_title("Distribution of linlin values < 100 in SolO dataset")

plt.savefig("linlin_histogram.png")