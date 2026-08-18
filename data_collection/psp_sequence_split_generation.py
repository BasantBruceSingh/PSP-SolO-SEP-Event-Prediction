import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import random


cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL", "LAT_FWT", "LON_FWT"]

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
mask_time = []
y_time = []

for time, g in groups:
    features = g[cols].to_numpy(dtype=np.float64)
    n_ar = len(features)

    X_t = np.zeros((max_num_ARs, num_features), dtype=np.float64)
    mask_t = np.zeros((max_num_ARs,), dtype=bool)

    X_t[:n_ar, :] = features
    mask_t[:n_ar] = True

    y_t = g["Jlinlin"].iloc[0] > 100.0

    X_time.append(X_t)
    mask_time.append(mask_t)
    y_time.append(y_t)

X_time = np.stack(X_time)
mask_time = np.stack(mask_time)
y_time = np.array(y_time) 

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

def make_windows_from_blocks(X_time, mask_time, y_time, blocks, seq_len):
    X_seq = []
    mask_seq = []
    y_seq = []
    target_indices = []

    for block in blocks:
        # Need seq_len points before first target, all inside the same block
        block_start = block[0]
        block_end = block[-1]
        for target_idx in range(block_start + seq_len - 1, block_end + 1):
            window_start = target_idx - seq_len + 1

            X_seq.append(X_time[window_start:target_idx + 1])
            mask_seq.append(mask_time[window_start:target_idx + 1])
            y_seq.append(y_time[target_idx])
            target_indices.append(target_idx)

    return (
        np.stack(X_seq).astype(np.float32),
        np.stack(mask_seq).astype(bool),
        np.array(y_seq, dtype=np.float32),
        np.array(target_indices),
    )

seq_len = 24

def generate_rms_points(blocks):
    indices = np.concatenate(blocks)
    X = np.stack([X_time[int(i)] for i in indices])
    mask = np.stack([mask_time[int(i)] for i in indices])
    y = np.array([y_time[int(i)] for i in indices])
    return X, mask, y

X_t_train, mask_t_train, y_t_train = generate_rms_points(train_blocks)
X_t_val, mask_t_val, y_t_val = generate_rms_points(val_blocks)
X_t_test, mask_t_test, y_t_test = generate_rms_points(test_blocks)

# Fit feature normalization only on real AR rows from train blocks
train_real = X_t_train[mask_t_train]

X_mean = train_real.mean(axis=0)
X_std = train_real.std(axis=0)
X_std[X_std == 0] = 1.0

# Scale all real AR rows using train statistics
X_time_scaled = X_time.copy()
X_time_scaled[mask_time] = (X_time_scaled[mask_time] - X_mean) / X_std
X_time_scaled = X_time_scaled.astype(np.float32)

# Now build windows from the already-split blocks
X_train, mask_train, y_train, train_idx = make_windows_from_blocks(
    X_time_scaled, mask_time, y_time, train_blocks, seq_len
)

X_val, mask_val, y_val, val_idx = make_windows_from_blocks(
    X_time_scaled, mask_time, y_time, val_blocks, seq_len
)

X_test, mask_test, y_test, test_idx = make_windows_from_blocks(
    X_time_scaled, mask_time, y_time, test_blocks, seq_len
)

print("X_train:", X_train.shape)
print("mask_train:", mask_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("mask_val:", mask_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("mask_test:", mask_test.shape)
print("y_test:", y_test.shape)

np.save("npy/X_seq_train_psp", X_train)
np.save("npy/X_seq_val_psp", X_val)
np.save("npy/X_seq_test_psp", X_test)
np.save("npy/y_seq_train_psp", y_train)
np.save("npy/y_seq_val_psp", y_val)
np.save("npy/y_seq_test_psp", y_test)
np.save("npy/mask_seq_train_psp", mask_train)
np.save("npy/mask_seq_val_psp", mask_val)
np.save("npy/mask_seq_test_psp", mask_test)