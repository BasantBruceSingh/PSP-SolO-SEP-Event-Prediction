import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.dates as mdates
import numpy as np

SHARP_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP.csv"
SolO_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/processed_SolO.csv"
output_img = "active_region_durations_packed.png"

df = pd.read_csv(SHARP_csv, parse_dates=["T_OBS"])
df = df[df["T_OBS"] > pd.Timestamp("2020-06-01")]
df = df.dropna(subset=["T_OBS", "ARPNUM"])
df = df.sort_values(["ARPNUM", "T_OBS"])

max_gap = pd.Timedelta(hours=4)

intervals = []

for ar, g in tqdm(df.groupby("ARPNUM"), desc="Processing active regions"):
    times = g["T_OBS"].sort_values().drop_duplicates()

    if len(times) == 0:
        continue

    start = times.iloc[0]
    prev = times.iloc[0]

    for t in times.iloc[1:]:
        if t - prev > max_gap:
            intervals.append({"ARPNUM": ar, "start": start, "end": prev})
            start = t
        prev = t

    intervals.append({"ARPNUM": ar, "start": start, "end": prev})

interval_df = pd.DataFrame(intervals)
interval_df["duration"] = interval_df["end"] - interval_df["start"]
interval_df = interval_df[interval_df["duration"] > pd.Timedelta(0)]
interval_df = interval_df.sort_values("start").reset_index(drop=True)

# Greedy lane packing
lane_end_times = []
lanes = []

for _, row in tqdm(interval_df.iterrows(), desc="Packing intervals"):
    placed = False

    for lane_idx, lane_end in enumerate(lane_end_times):
        if row["start"] > lane_end:
            lanes.append(lane_idx)
            lane_end_times[lane_idx] = row["end"]
            placed = True
            break

    if not placed:
        lanes.append(len(lane_end_times))
        lane_end_times.append(row["end"])

interval_df["lane"] = lanes

fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

for _, row in tqdm(interval_df.iterrows(), desc="Plotting intervals"):
    start_num = plt.matplotlib.dates.date2num(row["start"])
    width_days = row["duration"].total_seconds() / 86400
    y = row["lane"]

    ax[0].broken_barh(
        [(start_num, width_days)],
        (y - 0.4, 0.8),
    )

ax[0].set_xlabel("Time")
ax[0].set_ylabel("Packed active-region row")
ax[0].set_title("Packed SHARP Active Region Durations")

ax[0].xaxis_date()
ax[0].xaxis.set_major_locator(mdates.YearLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax[0].grid(True, axis="x")

df_solO = pd.read_csv(SolO_csv, parse_dates=["time"])
df_solO = df_solO.sort_values("time")

mask = df_solO["footprint_valid"].astype(bool)

J_plot = df_solO["linlin"].where(mask, np.nan)

ax[1].plot(df_solO["time"], np.log(J_plot))
ax[1].set_xlabel("Time")
ax[1].set_ylabel("log(Jlinlin)")

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(output_img, dpi=200)