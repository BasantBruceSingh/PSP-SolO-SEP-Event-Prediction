from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

combined_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP_to_SolO_times.csv"
df = pd.read_csv(combined_csv, parse_dates=["time"]).sort_values(by=["time", "ARPNUM"])

current_time = None
sum = 0
ar_count = 0
linlins = []
usfluxs = []

for _, row in tqdm(df.iterrows(), desc="Processing rows"):
    time = row["time"]
    if time != current_time:
        if current_time is not None:
            avg = sum / ar_count
            usfluxs.append(avg) # added at end of run for a given time
        current_time = time
        linlin = row["Jlinlin"]
        linlins.append(linlin) # added at start of run for a given time
        sum = row["USFLUXL"]
        ar_count = 1
    else:
        sum += row["USFLUXL"]
        ar_count += 1

avg = sum / ar_count
usfluxs.append(avg) 

linlins = np.array(linlins)
usfluxs = np.array(usfluxs)

mask = (linlins > 0) & (usfluxs > 0)
linlins = linlins[mask]
usfluxs = usfluxs[mask]

x_bins = np.logspace(np.log10(linlins.min()), np.log10(linlins.max()), 50)
y_bins = np.logspace(np.log10(usfluxs.min()), np.log10(usfluxs.max()), 50)

plt.figure(figsize=(10, 6))
plt.hist2d(linlins, usfluxs, bins=(x_bins, y_bins))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Jlinlin")
plt.ylabel("Average USFLUX")
plt.title("Average USFLUX vs Jlinlin")
plt.grid()
plt.savefig("linlin_vs_usflux.png")