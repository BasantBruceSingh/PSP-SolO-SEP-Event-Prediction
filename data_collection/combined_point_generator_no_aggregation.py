"""Generate combined pointwise splits that retain individual active regions."""


import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = "npy"
os.makedirs(OUT_DIR, exist_ok=True)

cols = [
    "USFLUXL",
    "R_VALUE",
    "MEANGBL_GMM",
    "USFLUXZ",
    "MEANGBZ",
    "CMASKL",
    "LAT_FWT",
    "LON_FWT",
    "footprint_lon",
    "spacecraft_r",
]

EVENT_WINDOW = pd.Timedelta(days=10)
Y_EPS = 1e-5
SPLITS = ("train", "val", "test")

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


# ============================================================
# TIME / CATALOGUE HELPERS
# ============================================================

def parse_time_series(s):
    """Parse timestamp strings and represent UTC times without a timezone attached."""
    return pd.to_datetime(
        s.astype(str).str.strip(),
        format="mixed",
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)


def load_event_times(config):
    """Read catalogue event onsets for the specified spacecraft."""
    if config["catalogue_type"] == "csv":
        df = pd.read_csv(config["catalogue_path"])
        raw = df[config["start_col"]]

    elif config["catalogue_type"] == "xlsx":
        df = pd.read_excel(config["catalogue_path"])
        raw = df.iloc[
            config["skip_rows_before"]:,
            config["start_col_index"]
        ]

    else:
        raise ValueError("Unknown catalogue type")

    times = parse_time_series(raw)
    times = times.dropna().sort_values().to_list()

    if len(times) == 0:
        raise ValueError("No valid SEP event times")

    return times


def merge_event_windows(start_times, window=EVENT_WINDOW):
    """
    Merge SEP event windows from both PSP and SolO.

    Each event contributes [start, start + 10 days].
    Overlapping windows are merged.
    """
    windows = sorted(
        (t, t + window) for t in start_times
    )

    merged = []

    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


# ============================================================
# BUILD NON-AGGREGATED SHARP TENSORS
# ============================================================

def aggregate_sharp_by_time(combined_csv, satellite_name):
    """
    Convert AR-level SHARP rows into one sample per satellite time.

    No averaging across active regions.

    Returns:
        X_time:    [T, A_sat, F]
        mask_time: [T, A_sat]
        y_time:    [T] raw Jlinlin
        times:     [T]
    """
    df = pd.read_csv(combined_csv)

    required = ["time", "Jlinlin"] + cols
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{combined_csv} missing columns: {missing}"
        )

    df["time"] = parse_time_series(df["time"])
    df = df.dropna(subset=required)
    df = df.sort_values("time", kind="stable").reset_index(drop=True)

    groups = list(df.groupby("time", sort=True))

    if len(groups) == 0:
        raise ValueError(
            f"No valid time groups in {combined_csv}"
        )

    max_num_ARs = max(len(g) for _, g in groups)
    num_features = len(cols)

    X_time = np.zeros(
        (len(groups), max_num_ARs, num_features),
        dtype=np.float32
    )

    mask_time = np.zeros(
        (len(groups), max_num_ARs),
        dtype=bool
    )

    y_time = np.zeros(len(groups), dtype=np.float32)
    times = []

    for i, (time, g) in enumerate(
        tqdm(groups, desc=f"Building AR tensors for {satellite_name}")
    ):
        features = g[cols].to_numpy(dtype=np.float32)
        n_ar = len(features)

        X_time[i, :n_ar, :] = features
        mask_time[i, :n_ar] = True

        targets = g["Jlinlin"].to_numpy(dtype=np.float64)

        # One satellite target per timestamp.
        if not np.allclose(
            targets, targets[0], rtol=1e-5, atol=1e-10
        ):
            raise ValueError(
                f"{satellite_name}: inconsistent Jlinlin at {time}"
            )

        y_time[i] = targets[0]
        times.append(time)

    times = np.array(times, dtype="datetime64[ns]")

    return {
        "X_time": X_time,
        "mask_time": mask_time,
        "y_time": y_time,
        "times": times,
    }


# ============================================================
# BUILD SHARED TEMPORAL BLOCKS
# ============================================================

def build_shared_blocks(all_times, event_windows):
    """
    Construct temporal blocks using the union of PSP and SolO times.

    All observations belonging to the same block receive
    the same train/val/test assignment.

    Event blocks:
        One block per merged SEP event window.

    Non-event blocks:
        Consecutive non-event observations in <=10-day chunks.
    """
    all_times = pd.DatetimeIndex(all_times).sort_values().unique()

    if len(all_times) == 0:
        raise ValueError("No observation timestamps")

    event_membership = np.full(
        len(all_times), -1, dtype=int
    )

    # Assign timestamps to merged event windows.
    for event_id, (start, end) in enumerate(event_windows):
        selected = (all_times >= start) & (all_times <= end)
        event_membership[selected] = event_id

    blocks = []
    block_ids = np.full(len(all_times), -1, dtype=int)

    # Event blocks
    for event_id in range(len(event_windows)):
        indices = np.flatnonzero(event_membership == event_id)

        if len(indices) == 0:
            continue

        block_id = len(blocks)

        blocks.append({
            "id": block_id,
            "type": "event",
            "indices": indices
        })

        block_ids[indices] = block_id

    # Non-event blocks
    non_event_indices = np.flatnonzero(event_membership == -1)

    current = []

    def append_non_event_block(indices):
        """Register a nonempty quiet-time block and assign its ID to its timestamps."""
        if len(indices) == 0:
            return

        indices = np.array(indices, dtype=int)
        block_id = len(blocks)

        blocks.append({
            "id": block_id,
            "type": "non_event",
            "indices": indices
        })

        block_ids[indices] = block_id

    for idx in non_event_indices:
        if len(current) == 0:
            current = [int(idx)]
            continue

        start_time = all_times[current[0]]
        current_time = all_times[idx]

        if current_time > start_time + EVENT_WINDOW:
            append_non_event_block(current)
            current = [int(idx)]
        else:
            current.append(int(idx))

    append_non_event_block(current)

    assert np.all(block_ids >= 0)
    assert sum(len(b["indices"]) for b in blocks) == len(all_times)

    return blocks, block_ids, all_times


# ============================================================
# SPLIT SHARED BLOCKS
# ============================================================

def split_shared_blocks(
    blocks,
    seed=SEED,
    train_frac=0.70,
    val_frac=0.15
):
    """
    Randomly assign shared blocks to train/val/test.

    Event and non-event blocks are split separately.

    Fractions apply to block counts, not individual samples.
    """
    rng = random.Random(seed)
    assignments = {}

    for block_type in ("event", "non_event"):

        selected = [
            b["id"]
            for b in blocks
            if b["type"] == block_type
        ]

        rng.shuffle(selected)

        n = len(selected)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        for bid in selected[:n_train]:
            assignments[bid] = "train"

        for bid in selected[n_train:n_train + n_val]:
            assignments[bid] = "val"

        for bid in selected[n_train + n_val:]:
            assignments[bid] = "test"

    assert len(assignments) == len(blocks)

    return assignments


# ============================================================
# EXTRACT SATELLITE SAMPLES USING SHARED ASSIGNMENTS
# ============================================================

def extract_satellite_splits(
    data,
    global_times,
    global_block_ids,
    assignments
):
    """
    Assign each satellite observation according to its
    globally shared temporal block.
    """
    lookup = {
        pd.Timestamp(t): assignments[int(bid)]
        for t, bid in zip(global_times, global_block_ids)
    }

    sample_splits = np.array([
        lookup[pd.Timestamp(t)]
        for t in data["times"]
    ])

    result = {}

    for split in SPLITS:
        indices = np.flatnonzero(sample_splits == split)

        result[f"X_{split}"] = data["X_time"][indices].copy()
        result[f"mask_{split}"] = data["mask_time"][indices].copy()
        result[f"y_{split}_raw"] = data["y_time"][indices].copy()
        result[f"t_{split}"] = data["times"][indices].copy()

    return result


# ============================================================
# PAD PSP AND SOLO TO COMMON AR DIMENSION
# ============================================================

def pad_point_split(X, mask, global_max_ARs):
    """
    X:    [N, A_sat, F]
    mask: [N, A_sat]

    Returns:
        X:    [N, global_max_ARs, F]
        mask: [N, global_max_ARs]
    """
    N, A, F = X.shape

    if A == global_max_ARs:
        return X, mask

    X_padded = np.zeros(
        (N, global_max_ARs, F),
        dtype=np.float32
    )

    mask_padded = np.zeros(
        (N, global_max_ARs),
        dtype=bool
    )

    X_padded[:, :A, :] = X
    mask_padded[:, :A] = mask

    return X_padded, mask_padded


# ============================================================
# SHARED SHARP FEATURE NORMALIZATION
# ============================================================

def fit_X_normalization_from_train(datasets):
    """
    Fit SHARP feature normalization using PSP + SolO
    training observations only.

    Padded AR rows are excluded.

    This preserves the normalization used in the original code.
    """
    train_real_rows = []

    for sat, data in datasets.items():
        X_train = data["X_train"]
        mask_train = data["mask_train"]

        if len(X_train) == 0:
            continue

        real_rows = X_train[mask_train]

        if len(real_rows) > 0:
            train_real_rows.append(real_rows)

    if len(train_real_rows) == 0:
        raise ValueError("No real training AR rows")

    train_real_rows = np.concatenate(
        train_real_rows, axis=0
    ).astype(np.float64)

    X_mean = np.nanmean(train_real_rows, axis=0)
    X_std = np.nanstd(train_real_rows, axis=0)

    X_mean[~np.isfinite(X_mean)] = 0.0
    X_std[~np.isfinite(X_std) | (X_std == 0)] = 1.0

    return X_mean.astype(np.float32), X_std.astype(np.float32)


def apply_X_normalization_to_splits(datasets, X_mean, X_std):
    """
    Apply shared train-only feature normalization.

    Padded AR rows remain exactly zero.
    """
    for sat, data in datasets.items():
        for split in SPLITS:
            X = data[f"X_{split}"].copy()
            mask = data[f"mask_{split}"]

            if len(X) > 0:
                X[mask] = (X[mask] - X_mean) / X_std

            X[~mask] = 0.0
            data[f"X_{split}"] = X.astype(np.float32)


# ============================================================
# SATELLITE-SPECIFIC TARGET NORMALIZATION
# ============================================================

def zscore_y_per_satellite_train_stats(datasets, eps=Y_EPS):
    """
    Normalize Jlinlin separately for PSP and SolO.

    y_log = log(y_raw + eps)
    y_z = (y_log - satellite_train_mean) / satellite_train_std
    """
    stats = {}

    for sat, data in datasets.items():

        y_train_raw = data["y_train_raw"].astype(np.float64)

        if len(y_train_raw) == 0:
            raise ValueError(f"{sat}: empty training targets")

        if (
            not np.all(np.isfinite(y_train_raw))
            or np.any(y_train_raw <= -eps)
        ):
            raise ValueError(f"{sat}: invalid training targets")

        y_train_log = np.log(y_train_raw + eps)

        y_mean = np.mean(y_train_log)
        y_std = np.std(y_train_log)

        if not np.isfinite(y_std) or y_std == 0:
            y_std = 1.0

        stats[sat] = {
            "log_mean": y_mean,
            "log_std": y_std,
            "raw_train_min": np.min(y_train_raw),
            "raw_train_median": np.median(y_train_raw),
            "raw_train_max": np.max(y_train_raw),
            "log_train_min": np.min(y_train_log),
            "log_train_median": np.median(y_train_log),
            "log_train_max": np.max(y_train_log),
        }

        for split in SPLITS:
            raw = data[f"y_{split}_raw"].astype(np.float64)

            if (
                not np.all(np.isfinite(raw))
                or np.any(raw <= -eps)
            ):
                raise ValueError(
                    f"{sat}: invalid {split} targets"
                )

            log_y = np.log(raw + eps)
            z_y = (log_y - y_mean) / y_std

            data[f"y_{split}_log"] = log_y.astype(np.float32)
            data[f"y_{split}"] = z_y.astype(np.float32)

        print(
            f"{sat.upper()} y normalization: "
            f"mean={y_mean:.6g}, std={y_std:.6g}"
        )

    return stats


# ============================================================
# COMBINE PSP + SOLO SPLITS
# ============================================================

def combine_splits(datasets, split):
    """
    Concatenate PSP and SolO observations within one split.

    All arrays are shuffled using the same permutation.
    """
    Xs = []
    masks = []
    ys = []
    y_logs = []
    y_raws = []
    times = []
    sats = []

    for sat, data in datasets.items():
        Xs.append(data[f"X_{split}"])
        masks.append(data[f"mask_{split}"])
        ys.append(data[f"y_{split}"])
        y_logs.append(data[f"y_{split}_log"])
        y_raws.append(data[f"y_{split}_raw"])
        times.append(data[f"t_{split}"])

        sats.append(
            np.full(
                len(data[f"t_{split}"]),
                sat,
                dtype="<U8"
            )
        )

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    mask = np.concatenate(masks, axis=0).astype(bool)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    y_log = np.concatenate(y_logs, axis=0).astype(np.float32)
    y_raw = np.concatenate(y_raws, axis=0).astype(np.float32)
    t = np.concatenate(times, axis=0)
    sat = np.concatenate(sats, axis=0).astype("<U8")

    perm = np.random.permutation(len(y))

    return (
        X[perm],
        mask[perm],
        y[perm],
        y_log[perm],
        y_raw[perm],
        t[perm],
        sat[perm],
    )


# ============================================================
# VALIDATE TEMPORAL SPLITS
# ============================================================

def validate_global_splits(datasets, blocks, assignments):
    """
    Check that:
    1. No exact timestamp is assigned to multiple splits.
    2. PSP and SolO inherit the same global block assignment.
    3. All blocks have exactly one split.
    """
    split_times = {s: set() for s in SPLITS}

    for sat, data in datasets.items():
        for split in SPLITS:
            times = pd.to_datetime(data[f"t_{split}"])
            split_times[split].update(times)

            X = data[f"X_{split}"]
            mask = data[f"mask_{split}"]
            y = data[f"y_{split}"]

            assert X.shape[:2] == mask.shape
            assert len(X) == len(y)
            assert np.isfinite(X).all()
            assert np.isfinite(y).all()
            assert np.all(X[~mask] == 0.0)

    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            overlap = split_times[s1] & split_times[s2]

            assert len(overlap) == 0, (
                f"Temporal leakage between {s1} and {s2}: "
                f"{len(overlap)} timestamps"
            )

    assert len(assignments) == len(blocks)

    print("\nPASS: No exact timestamp leakage across PSP/SolO splits.")


# ============================================================
# MAIN PIPELINE
# ============================================================

# 1. Load SEP event times for both satellites.
all_event_starts = []

for sat, config in SAT_CONFIGS.items():
    event_times = load_event_times(config)
    all_event_starts.extend(event_times)

    print(f"{sat.upper()}: {len(event_times)} SEP event starts")

# Merge overlapping 10-day windows globally.
event_windows = merge_event_windows(all_event_starts)

print("\nMerged global SEP event windows:", len(event_windows))


# 2. Build point-level SHARP tensors separately.
raw_datasets = {}

for sat, config in SAT_CONFIGS.items():
    raw_datasets[sat] = aggregate_sharp_by_time(
        config["combined_csv"],
        satellite_name=sat
    )

    print(
        f"{sat.upper()} raw tensor:",
        raw_datasets[sat]["X_time"].shape
    )


# 3. Construct union of PSP + SolO timestamps.
all_times = np.unique(
    np.concatenate([
        raw_datasets["psp"]["times"],
        raw_datasets["solo"]["times"]
    ])
)


# 4. Build shared temporal blocks.
blocks, global_block_ids, global_times = build_shared_blocks(
    all_times,
    event_windows
)

print("\nGlobal block summary:")
print("Unique timestamps:", len(global_times))
print("Event blocks:", sum(b["type"] == "event" for b in blocks))
print("Non-event blocks:", sum(b["type"] == "non_event" for b in blocks))


# 5. Split globally, ONCE.
assignments = split_shared_blocks(
    blocks,
    seed=SEED
)

print("\nShared block split counts:")
for split in SPLITS:
    print(
        split,
        sum(v == split for v in assignments.values())
    )


# 6. Extract satellite-specific samples using global splits.
datasets = {}

for sat, raw in raw_datasets.items():
    datasets[sat] = extract_satellite_splits(
        raw,
        global_times,
        global_block_ids,
        assignments
    )


# 7. Pad both satellites to the same AR dimension.
global_max_ARs = max(
    raw["X_time"].shape[1]
    for raw in raw_datasets.values()
)

print("\nGlobal maximum ARs:", global_max_ARs)

for sat, data in datasets.items():
    for split in SPLITS:
        X_padded, mask_padded = pad_point_split(
            data[f"X_{split}"],
            data[f"mask_{split}"],
            global_max_ARs
        )

        data[f"X_{split}"] = X_padded
        data[f"mask_{split}"] = mask_padded


# 8. Fit SHARP normalization from COMBINED training AR rows.
X_mean, X_std = fit_X_normalization_from_train(datasets)

apply_X_normalization_to_splits(
    datasets,
    X_mean,
    X_std
)

pd.DataFrame({
    "feature": cols,
    "mean": X_mean,
    "std": X_std,
}).to_csv(
    os.path.join(
        OUT_DIR,
        "X_point_normalization_stats_combined_na.csv"
    ),
    index=False
)


# 9. Normalize Jlinlin independently per satellite.
y_stats = zscore_y_per_satellite_train_stats(datasets)

pd.DataFrame(y_stats).T.to_csv(
    os.path.join(
        OUT_DIR,
        "y_point_normalization_stats_by_satellite_na.csv"
    )
)


# 10. Validate the shared block splitting.
validate_global_splits(
    datasets,
    blocks,
    assignments
)


# 11. Combine PSP and SolO for each split.
combined = {}

for split in SPLITS:
    combined[split] = combine_splits(datasets, split)


# 12. Save original combined dataset filenames.
for split in SPLITS:
    X, mask, y, y_log, y_raw, t, sat = combined[split]

    np.save(
        os.path.join(OUT_DIR, f"X_t_{split}_combined_na.npy"),
        X
    )

    np.save(
        os.path.join(OUT_DIR, f"mask_t_{split}_combined_na.npy"),
        mask
    )

    np.save(
        os.path.join(OUT_DIR, f"y_t_{split}_combined_na.npy"),
        y
    )

    np.save(
        os.path.join(OUT_DIR, f"y_t_{split}_combined_na_log.npy"),
        y_log
    )

    np.save(
        os.path.join(OUT_DIR, f"y_t_{split}_combined_na_raw.npy"),
        y_raw
    )

    np.save(
        os.path.join(OUT_DIR, f"times_t_{split}_combined_na.npy"),
        t
    )

    np.save(
        os.path.join(OUT_DIR, f"sat_t_{split}_combined_na.npy"),
        sat
    )

    print(
        f"\n{split.upper()} combined shapes:"
        f"\n  X: {X.shape}"
        f"\n  mask: {mask.shape}"
        f"\n  y: {y.shape}"
        f"\n  PSP samples: {np.sum(sat == 'psp')}"
        f"\n  SolO samples: {np.sum(sat == 'solo')}"
    )


# 13. Save global timestamp/block assignment manifest.
block_types = {
    b["id"]: b["type"]
    for b in blocks
}

manifest = pd.DataFrame({
    "time": global_times,
    "block_id": global_block_ids,
    "block_type": [
        block_types[int(bid)]
        for bid in global_block_ids
    ],
    "split": [
        assignments[int(bid)]
        for bid in global_block_ids
    ],
})

manifest.to_csv(
    os.path.join(OUT_DIR, "shared_time_split_combined_na.csv"),
    index=False
)

print("\nAll combined datasets saved successfully.")
print("Output directory:", OUT_DIR)
print("Shared split manifest: shared_time_split_combined_na.csv")
