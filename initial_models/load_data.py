from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np
import random

combined_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP_to_SolO_times.csv"
window_size_hrs = 240

df = pd.read_csv(combined_csv, parse_dates=["time"]).sort_values(by=["time", "ARPNUM"])
cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL", "LAT_FWT", "LON_FWT"]

groups = list(df.groupby("time", sort=True))
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

    y_t = g["Jlinlin"].iloc[0]

    X_time.append(X_t)
    mask_time.append(mask_t)
    y_time.append(y_t)

X_time = np.stack(X_time)
mask_time = np.stack(mask_time)
y_time = np.array(y_time) 

num_times = len(X_time)

train_end = int(0.7 * num_times)
val_end = int(0.85 * num_times)

train_time_mask = np.zeros(num_times, dtype=bool)
val_time_mask = np.zeros(num_times, dtype=bool)
test_time_mask = np.zeros(num_times, dtype=bool)

train_time_mask[:train_end] = True
val_time_mask[train_end:val_end] = True
test_time_mask[val_end:] = True

def log(x):
    if x == 0:
        return 0
    else:
        return np.log(x)

y_time = np.vectorize(log)(y_time)

# Turn time series into windows
seq_len = 24

def make_nonoverlap_block_split(num_times, seq_len, block_size, train_frac=0.7, val_frac=0.15, seed=0):
    """
    Splits raw time indices into shuffled chronological blocks.
    Returns block ranges assigned to train/val/test.

    Windows should later be built separately within each block.
    """
    rng = random.Random(seed)

    blocks = []
    start = 0

    while start + block_size <= num_times:
        end = start + block_size
        blocks.append((start, end))
        start = end

    rng.shuffle(blocks)

    n_blocks = len(blocks)
    n_train = int(train_frac * n_blocks)
    n_val = int(val_frac * n_blocks)

    train_blocks = blocks[:n_train]
    val_blocks = blocks[n_train:n_train + n_val]
    test_blocks = blocks[n_train + n_val:]

    return train_blocks, val_blocks, test_blocks

def make_windows_from_blocks(X_time, mask_time, y_time, blocks, seq_len):
    X_seq = []
    mask_seq = []
    y_seq = []
    target_indices = []

    for block_start, block_end in blocks:
        # Need seq_len points before first target, all inside the same block
        for target_idx in range(block_start + seq_len - 1, block_end):
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
block_size = 120

train_blocks, val_blocks, test_blocks = make_nonoverlap_block_split(
    num_times=len(X_time),
    seq_len=seq_len,
    block_size=block_size,
    train_frac=0.6,
    val_frac=0.2,
    seed=0,
)

# Raw time indices belonging to train blocks
train_time_indices = np.concatenate([
    np.arange(start, end) for start, end in train_blocks
])

# Fit feature normalization only on real AR rows from train blocks
train_real = X_time[train_time_indices][mask_time[train_time_indices]]

X_mean = train_real.mean(axis=0)
X_std = train_real.std(axis=0)
X_std[X_std == 0] = 1.0

# Scale all real AR rows using train statistics
X_time_scaled = X_time.copy()
X_time_scaled[mask_time] = (X_time_scaled[mask_time] - X_mean) / X_std
X_time_scaled = X_time_scaled.astype(np.float32)

train_time_indices = np.concatenate([
    np.arange(start, end) for start, end in train_blocks
])

# Scale y
y_mean = y_time[train_time_indices].mean()
y_std = y_time[train_time_indices].std()

y_time_scaled = (y_time - y_mean) / y_std

# Now build windows from the already-split blocks
X_train, mask_train, y_train, train_idx = make_windows_from_blocks(
    X_time_scaled, mask_time, y_time_scaled, train_blocks, seq_len
)

X_val, mask_val, y_val, val_idx = make_windows_from_blocks(
    X_time_scaled, mask_time, y_time_scaled, val_blocks, seq_len
)

X_test, mask_test, y_test, test_idx = make_windows_from_blocks(
    X_time_scaled, mask_time, y_time_scaled, test_blocks, seq_len
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

class SHARPLinlinSequenceDataset(Dataset):
    def __init__(self, X, ar_mask, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ar_mask = torch.tensor(ar_mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ar_mask[idx], self.y[idx]

batch_size = 32

train_ds = SHARPLinlinSequenceDataset(X_train, mask_train, y_train)
val_ds = SHARPLinlinSequenceDataset(X_val, mask_val, y_val)
test_ds = SHARPLinlinSequenceDataset(X_test, mask_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)