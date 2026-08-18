import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import random

cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL", "LAT_FWT", "LON_FWT"]
rms_cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL"]
mean_cols = ["LAT_FWT", "LON_FWT"]

input_csv = "psp_isois_eventlist_v21.csv"
df_catalogue = pd.read_csv(input_csv, parse_dates=[" Start (UTC)     ", " End (UTC)       "])
df_catalogue = df_catalogue.dropna()

combined_csv = "SHARP_to_PSP_times.csv"
df_combined = pd.read_csv(combined_csv, parse_dates=["time"])

df_combined["time"] = pd.to_datetime(
    df_combined["time"].astype(str).str.strip(),
    format="mixed",
    errors="coerce",
)

start_times_str = df_catalogue[" Start (UTC)     "].values
#start_times = np.vectorize(pd.to_datetime)(start_times)
start_times = []

for i, time in enumerate(start_times_str):
    # Quirk of the csv file
    if time == "                 ":
        continue
    start_times.append(pd.to_datetime(start_times_str[i], format=" %Y-%m-%dT%H:%M"))

groups = list(df_combined.groupby("time", sort=True))
times = [t for t, _ in groups]

max_num_ARs = max(len(g) for _, g in groups)
num_features = len(cols)
    
X_time = []
y_time = []
times = []

for time, g in groups:
    rms_features = g[rms_cols].to_numpy(dtype=np.float64)
    mean_features = g[mean_cols].to_numpy(dtype=np.float64)
    n_ar = len(rms_features)

    # Calculate root mean square of each feature across ARs
    X_t_rms = np.sqrt(np.mean(rms_features**2, axis=0))
    X_t_mean = np.mean(mean_features, axis=0)
    X_t = np.concat([X_t_rms, X_t_mean])

    y_t = g["Jlinlin"].iloc[0] * 1000 # Unit conversion from KeV to MeV in denominator

    X_time.append(X_t)
    y_time.append(y_t)
    times.append(time)

X_time = np.stack(X_time)
y_time = np.array(y_time) 
times = np.array(times)

event_blocks = []
non_events = []

event_ptr = 0
block = []
end_flag = False

for i in tqdm(range(len(X_time)), total=len(X_time), desc="Blocking combined dataset"):
    if end_flag:
        non_events.append(i)
        continue

    psp_time = times[i]
    # While psp_time is after 10 day window of event, move to next event
    while psp_time > start_times[event_ptr] + np.timedelta64(240, 'h'):
        event_ptr += 1
        if event_ptr >= len(start_times):
            end_flag = True
            break

    if end_flag:
        non_events.append(i)
        if len(block) > 0:
            event_blocks.append(block)
            block = []
        continue

    if psp_time > start_times[event_ptr]:
        block.append(i)
    else:
        non_events.append(i)
        if len(block) > 0:
            event_blocks.append(block)
            block = []

if len(block) > 0:
    event_blocks.append(block)

non_event_blocks = []
block = []

for i in range(len(non_events)):
    if len(block) > 0:
        start_time = times[block[0]]
        this_time = times[non_events[i]]
        if this_time > start_time + np.timedelta64(240, 'h'):
            non_event_blocks.append(block)
            block = [non_events[i]]
        else:
            block.append(non_events[i])
    else:
        block = [non_events[i]]

if len(block) > 0:
    non_event_blocks.append(block)

random.shuffle(event_blocks)
random.shuffle(non_event_blocks)

train_blocks = event_blocks[:int(len(event_blocks)*0.7)] + non_event_blocks[:int(len(non_event_blocks)*0.7)]
random.shuffle(train_blocks)

val_blocks = event_blocks[int(len(event_blocks)*0.7):int(len(event_blocks)*0.85)] + non_event_blocks[int(len(non_event_blocks)*0.7):int(len(non_event_blocks)*0.85)]
random.shuffle(val_blocks)

test_blocks = event_blocks[int(len(event_blocks)*0.85):] + non_event_blocks[int(len(non_event_blocks)*0.85):]
random.shuffle(test_blocks)

def generate_rms_points(blocks):
    indices = np.concatenate(blocks)
    X = np.stack([X_time[int(i)] for i in indices])
    y = np.array([y_time[int(i)] for i in indices])
    return X, y

X_t_train, y_t_train = generate_rms_points(train_blocks)
X_t_val, y_t_val = generate_rms_points(val_blocks)
X_t_test, y_t_test = generate_rms_points(test_blocks)
np.save("npy/X_t_train_psp", X_t_train)
np.save("npy/X_t_val_psp", X_t_val)
np.save("npy/X_t_test_psp", X_t_test)
np.save("npy/y_t_train_psp", y_t_train)
np.save("npy/y_t_val_psp", y_t_val)
np.save("npy/y_t_test_psp", y_t_test)