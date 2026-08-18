from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

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
            usfluxs.append(sum) # added at end of run for a given time
        current_time = time
        linlin = row["Jlinlin"]
        linlins.append(linlin) # added at start of run for a given time
        sum = row["USFLUXL"]
        ar_count = 1
    else:
        sum += row["USFLUXL"]
        ar_count += 1

usfluxs.append(sum) 
    
plt.figure(figsize=(10, 6))
plt.scatter(linlins, usfluxs, marker='x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Jlinlin")
plt.ylabel("Total USFLUX")
plt.title("Total USFLUX vs Jlinlin")
plt.grid()
plt.savefig("linlin_vs_usflux*ar.png")