import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = "npy"
os.makedirs(OUT_DIR, exist_ok=True)

cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL", "LAT_FWT", "LON_FWT"]
rms_cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL"]
mean_cols = ["LAT_FWT", "LON_FWT"]

EVENT_WINDOW = pd.Timedelta(days=10)

SAT_CONFIGS = {
    "psp": {
        "combined_csv": "SHARP_to_PSP_times.csv",
        "catalogue_path": "psp_isois_eventlist_v21.csv",
        "catalogue_type": "csv",
        "start_col": " Start (UTC)     ",
    },
    "solo": {
        "combined_csv": "SHARP_to_SolO_times.csv",
        "catalogue_path": "SolO_catalogue.xlsx",
        "catalogue_type": "xlsx",
        "start_col_index": 28,
        "skip_rows_before": 4,
    },
}


# -----------------------------
# Helpers
# -----------------------------
def parse_time_series(s):
    return pd.to_datetime(
        s.astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )


def load_psp_start_times(path, start_col):
    df = pd.read_csv(path)

    start_times = parse_time_series(df[start_col])
    start_times = start_times.dropna().sort_values().to_list()

    if len(start_times) == 0:
        raise ValueError(f"No valid PSP event start times found in {path}")

    return start_times


def load_solo_start_times(path, start_col_index=28, skip_rows_before=4):
    df = pd.read_excel(path)

    raw = df.iloc[skip_rows_before:, start_col_index]
    start_times = pd.to_datetime(
        raw.astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )
    start_times = start_times.dropna().sort_values().to_list()

    if len(start_times) == 0:
        raise ValueError(f"No valid SolO event start times found in {path}")

    return start_times


def merge_event_windows(start_times, window=EVENT_WINDOW):
    """
    Convert event starts into merged [start, start + window] intervals.
    This handles overlapping SEP event windows correctly.
    """
    windows = [(t, t + window) for t in sorted(start_times)]

    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(s, e) for s, e in merged]


def aggregate_sharp_by_time(combined_csv, satellite_name):
    """
    Converts AR-level SHARP rows into one point per satellite time.

    X_t = [RMS of unsigned/large features, mean of lat/lon features]
    y_t = raw Jlinlin
    """
    df = pd.read_csv(combined_csv)

    df["time"] = parse_time_series(df["time"])
    df = df.dropna(subset=["time", "Jlinlin"])
    df = df.sort_values("time")

    missing = [c for c in cols + ["Jlinlin"] if c not in df.columns]
    if missing:
        raise ValueError(f"{combined_csv} missing columns: {missing}")

    groups = list(df.groupby("time", sort=True))

    X_time = []
    y_time = []
    times = []

    for time, g in tqdm(groups, desc=f"Aggregating SHARP rows for {satellite_name}"):
        rms_features = g[rms_cols].to_numpy(dtype=np.float64)
        mean_features = g[mean_cols].to_numpy(dtype=np.float64)

        # RMS aggregation for non-location features
        X_t_rms = np.sqrt(np.nanmean(rms_features ** 2, axis=0))

        # Mean aggregation for location/sign-sensitive features
        X_t_mean = np.nanmean(mean_features, axis=0)

        X_t = np.concatenate([X_t_rms, X_t_mean])

        # One Jlinlin value per timestamp; verify if needed that it is constant within group
        y_t = float(g["Jlinlin"].iloc[0])

        X_time.append(X_t)
        y_time.append(y_t)
        times.append(time)

    X_time = np.stack(X_time).astype(np.float32)
    y_time = np.array(y_time, dtype=np.float32)
    times = np.array(times, dtype="datetime64[ns]")

    return X_time, y_time, times


def make_blocks_from_event_windows(times, event_windows, max_non_event_window=EVENT_WINDOW):
    """
    Make blocks such that:
      - all points within each merged 10-day SEP event window stay together
      - non-event points are chunked into <=10-day blocks
    """
    times_ts = pd.to_datetime(times)

    event_blocks = []
    non_events = []

    event_ptr = 0
    current_event_block = []

    for i, t in enumerate(times_ts):
        # Advance past event windows that ended before this time
        while event_ptr < len(event_windows) and t > event_windows[event_ptr][1]:
            if len(current_event_block) > 0:
                event_blocks.append(current_event_block)
                current_event_block = []
            event_ptr += 1

        if event_ptr < len(event_windows):
            start, end = event_windows[event_ptr]
            in_event = start <= t <= end
        else:
            in_event = False

        if in_event:
            current_event_block.append(i)
        else:
            if len(current_event_block) > 0:
                event_blocks.append(current_event_block)
                current_event_block = []
            non_events.append(i)

    if len(current_event_block) > 0:
        event_blocks.append(current_event_block)

    # Chunk non-events into <=10-day blocks
    non_event_blocks = []
    block = []

    for idx in non_events:
        if len(block) == 0:
            block = [idx]
            continue

        block_start_time = times_ts[block[0]]
        this_time = times_ts[idx]

        if this_time > block_start_time + max_non_event_window:
            non_event_blocks.append(block)
            block = [idx]
        else:
            block.append(idx)

    if len(block) > 0:
        non_event_blocks.append(block)

    return event_blocks, non_event_blocks


def split_blocks(event_blocks, non_event_blocks, train_frac=0.70, val_frac=0.15):
    """
    Split event and non-event blocks separately so each split gets both types.
    """
    event_blocks = list(event_blocks)
    non_event_blocks = list(non_event_blocks)

    random.shuffle(event_blocks)
    random.shuffle(non_event_blocks)

    def split_one(blocks):
        n = len(blocks)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = blocks[:n_train]
        val = blocks[n_train:n_train + n_val]
        test = blocks[n_train + n_val:]

        return train, val, test

    e_train, e_val, e_test = split_one(event_blocks)
    n_train, n_val, n_test = split_one(non_event_blocks)

    train_blocks = e_train + n_train
    val_blocks = e_val + n_val
    test_blocks = e_test + n_test

    random.shuffle(train_blocks)
    random.shuffle(val_blocks)
    random.shuffle(test_blocks)

    return train_blocks, val_blocks, test_blocks


def points_from_blocks(X_time, y_time, times, blocks, satellite_name):
    blocks = [b for b in blocks if len(b) > 0]

    if len(blocks) == 0:
        return (
            np.empty((0, X_time.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=object),
        )

    indices = np.concatenate([np.array(b, dtype=int) for b in blocks])

    X = X_time[indices]
    y = y_time[indices]
    t = times[indices]
    sat = np.array([satellite_name] * len(indices), dtype=object)

    return X, y, t, sat


def build_satellite_dataset(satellite_name, config):
    if config["catalogue_type"] == "csv":
        start_times = load_psp_start_times(config["catalogue_path"], config["start_col"])
    elif config["catalogue_type"] == "xlsx":
        start_times = load_solo_start_times(
            config["catalogue_path"],
            start_col_index=config["start_col_index"],
            skip_rows_before=config["skip_rows_before"],
        )
    else:
        raise ValueError(f"Unknown catalogue_type: {config['catalogue_type']}")

    event_windows = merge_event_windows(start_times, EVENT_WINDOW)

    X_time, y_time, times = aggregate_sharp_by_time(
        config["combined_csv"],
        satellite_name=satellite_name,
    )

    event_blocks, non_event_blocks = make_blocks_from_event_windows(times, event_windows)

    print(f"\n{satellite_name.upper()} block summary")
    print("event blocks:", len(event_blocks))
    print("non-event blocks:", len(non_event_blocks))
    print("event points:", sum(len(b) for b in event_blocks))
    print("non-event points:", sum(len(b) for b in non_event_blocks))

    train_blocks, val_blocks, test_blocks = split_blocks(event_blocks, non_event_blocks)

    X_train, y_train_raw, t_train, sat_train = points_from_blocks(
        X_time, y_time, times, train_blocks, satellite_name
    )
    X_val, y_val_raw, t_val, sat_val = points_from_blocks(
        X_time, y_time, times, val_blocks, satellite_name
    )
    X_test, y_test_raw, t_test, sat_test = points_from_blocks(
        X_time, y_time, times, test_blocks, satellite_name
    )

    return {
        "X_train": X_train,
        "y_train_raw": y_train_raw,
        "t_train": t_train,
        "sat_train": sat_train,

        "X_val": X_val,
        "y_val_raw": y_val_raw,
        "t_val": t_val,
        "sat_val": sat_val,

        "X_test": X_test,
        "y_test_raw": y_test_raw,
        "t_test": t_test,
        "sat_test": sat_test,
    }


def zscore_y_per_satellite_train_stats(datasets):
    """
    Z-score logy for each satellite using that satellite's train y mean/std.
    Applies the same mean/std to val/test.
    """
    stats = {}

    for sat, data in datasets.items():
        y_train = np.log(data["y_train_raw"].astype(np.float64) + 1e-5)

        mu = np.nanmean(y_train)
        sigma = np.nanstd(y_train)

        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0

        stats[sat] = {"mean": mu, "std": sigma}

        for split in ["train", "val", "test"]:
            raw_key = f"y_{split}_raw"
            z_key = f"y_{split}"

            data[z_key] = (((np.log(data[raw_key].astype(np.float64) + 1e-5)) - mu) / sigma).astype(np.float32)

    return stats


def combine_splits(datasets, split):
    Xs = []
    ys = []
    y_raws = []
    times = []
    sats = []

    for sat, data in datasets.items():
        Xs.append(data[f"X_{split}"])
        ys.append(data[f"y_{split}"])
        y_raws.append(data[f"y_{split}_raw"])
        times.append(data[f"t_{split}"])
        sats.append(data[f"sat_{split}"])

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    y_raw = np.concatenate(y_raws, axis=0).astype(np.float32)
    t = np.concatenate(times, axis=0)
    sat = np.concatenate(sats, axis=0)

    # Shuffle within the split after block-level splitting is already done
    perm = np.random.permutation(len(y))

    return X[perm], y[perm], y_raw[perm], t[perm], sat[perm]


# -----------------------------
# Main
# -----------------------------
datasets = {}

for sat_name, config in SAT_CONFIGS.items():
    datasets[sat_name] = build_satellite_dataset(sat_name, config)

y_stats = zscore_y_per_satellite_train_stats(datasets)

print("\nY normalization stats, train-only:")
for sat, st in y_stats.items():
    print(f"{sat}: mean={st['mean']:.6g}, std={st['std']:.6g}")

X_train, y_train, y_train_raw, t_train, sat_train = combine_splits(datasets, "train")
X_val, y_val, y_val_raw, t_val, sat_val = combine_splits(datasets, "val")
X_test, y_test, y_test_raw, t_test, sat_test = combine_splits(datasets, "test")

print("\nCombined shapes")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

print("\nSatellite counts")
for split_name, sat_arr in [("train", sat_train), ("val", sat_val), ("test", sat_test)]:
    vals, counts = np.unique(sat_arr, return_counts=True)
    print(split_name, dict(zip(vals, counts)))

# Save normalized combined data
np.save(os.path.join(OUT_DIR, "X_t_train_combined.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_t_val_combined.npy"), X_val)
np.save(os.path.join(OUT_DIR, "X_t_test_combined.npy"), X_test)

np.save(os.path.join(OUT_DIR, "y_t_train_combined.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_t_val_combined.npy"), y_val)
np.save(os.path.join(OUT_DIR, "y_t_test_combined.npy"), y_test)

# Also save raw y, times, and satellite labels for debugging/evaluation
np.save(os.path.join(OUT_DIR, "y_t_train_combined_raw.npy"), y_train_raw)
np.save(os.path.join(OUT_DIR, "y_t_val_combined_raw.npy"), y_val_raw)
np.save(os.path.join(OUT_DIR, "y_t_test_combined_raw.npy"), y_test_raw)

np.save(os.path.join(OUT_DIR, "times_t_train_combined.npy"), t_train)
np.save(os.path.join(OUT_DIR, "times_t_val_combined.npy"), t_val)
np.save(os.path.join(OUT_DIR, "times_t_test_combined.npy"), t_test)

np.save(os.path.join(OUT_DIR, "sat_t_train_combined.npy"), sat_train)
np.save(os.path.join(OUT_DIR, "sat_t_val_combined.npy"), sat_val)
np.save(os.path.join(OUT_DIR, "sat_t_test_combined.npy"), sat_test)

pd.DataFrame(y_stats).T.to_csv(os.path.join(OUT_DIR, "y_normalization_stats_by_satellite.csv"))

print(f"\nSaved combined dataset to {OUT_DIR}/")