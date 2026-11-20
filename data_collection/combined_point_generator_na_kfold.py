"""Generate repeated blocked holdouts of pooled PSP and Solar Orbiter active-region sets."""


import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ============================================================
# Configuration
# ============================================================

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
OUT_DIR = "npy"
os.makedirs(OUT_DIR, exist_ok=True)

cols = [
    "USFLUXL", "R_VALUE", "MEANGBL_GMM",
    "USFLUXZ", "MEANGBZ", "CMASKL",
    "LAT_FWT", "LON_FWT",
    "footprint_lon", "spacecraft_r"
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
# Time and catalogue helpers
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
    else:
        df = pd.read_excel(config["catalogue_path"])
        raw = df.iloc[
            config["skip_rows_before"]:,
            config["start_col_index"]
        ]

    times = parse_time_series(raw).dropna().sort_values()

    if len(times) == 0:
        raise ValueError("No valid SEP event times")

    return times.to_list()


def merge_event_windows(start_times):
    """
    Merge overlapping 10-day event windows from BOTH satellites.
    """
    windows = sorted(
        (t, t + EVENT_WINDOW) for t in start_times
    )

    merged = []

    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(a, b) for a, b in merged]


# ============================================================
# Build point-level SHARP tensors (once, before holdouts)
# ============================================================

def aggregate_sharp_by_time(csv_path, satellite_name):
    """
    No AR averaging.

    X_time:    [T, A_sat, 10]
    mask_time: [T, A_sat]
    y_time:    [T]
    times:     [T]
    """
    df = pd.read_csv(csv_path)

    required = ["time", "Jlinlin"] + cols
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{csv_path} missing {missing}")

    df["time"] = parse_time_series(df["time"])
    df = df.dropna(subset=required)
    df = df.sort_values("time", kind="stable")

    groups = list(df.groupby("time", sort=True))

    if not groups:
        raise ValueError(f"No valid time groups in {csv_path}")

    max_ars = max(len(g) for _, g in groups)

    X = np.zeros(
        (len(groups), max_ars, len(cols)),
        dtype=np.float32
    )
    mask = np.zeros((len(groups), max_ars), dtype=bool)
    y = np.zeros(len(groups), dtype=np.float32)
    times = []

    for i, (t, g) in enumerate(
        tqdm(groups, desc=f"Building {satellite_name} tensors")
    ):
        features = g[cols].to_numpy(dtype=np.float32)
        n = len(features)

        X[i, :n] = features
        mask[i, :n] = True

        targets = g["Jlinlin"].to_numpy(dtype=np.float64)

        if not np.allclose(
            targets, targets[0], rtol=1e-5, atol=1e-10
        ):
            raise ValueError(
                f"{satellite_name}: inconsistent Jlinlin at {t}"
            )

        y[i] = targets[0]
        times.append(t)

    return {
        "X": X,
        "mask": mask,
        "y": y,
        "times": np.array(times, dtype="datetime64[ns]")
    }


# ============================================================
# Global temporal blocking
# ============================================================

def build_shared_blocks(all_times, event_windows):
    """
    Create blocks from the UNION of PSP and SolO timestamps.

    Each merged SEP event window forms one event block.
    Non-event timestamps form <=10-day blocks.

    Each unique timestamp belongs to exactly one block.
    """
    all_times = pd.DatetimeIndex(
        np.unique(all_times)
    ).sort_values()

    membership = np.full(len(all_times), -1, dtype=int)

    for event_id, (start, end) in enumerate(event_windows):
        selected = (all_times >= start) & (all_times <= end)
        membership[selected] = event_id

    blocks = []
    block_ids = np.full(len(all_times), -1, dtype=int)

    def append_block(indices, kind):
        """Register a nonempty block and record its ID for every member timestamp."""
        if len(indices) == 0:
            return

        indices = np.asarray(indices, dtype=int)
        bid = len(blocks)

        blocks.append({
            "id": bid,
            "type": kind,
            "indices": indices
        })

        block_ids[indices] = bid

    # Merged event blocks
    for event_id in range(len(event_windows)):
        indices = np.flatnonzero(membership == event_id)
        append_block(indices, "event")

    # Non-event blocks
    non_event_indices = np.flatnonzero(membership == -1)
    current = []

    for idx in non_event_indices:
        if not current:
            current = [int(idx)]
            continue

        if all_times[idx] > all_times[current[0]] + EVENT_WINDOW:
            append_block(current, "non_event")
            current = [int(idx)]
        else:
            current.append(int(idx))

    append_block(current, "non_event")

    assert np.all(block_ids >= 0)
    assert sum(len(b["indices"]) for b in blocks) == len(all_times)

    return blocks, block_ids, all_times


# ============================================================
# Repeated shared block splitting
# ============================================================

def split_shared_blocks(blocks, seed):
    """
    One global block split per seed, shared by both satellites.

    Event and non-event blocks are split separately.
    70% train, 15% val, remainder test (by block count).
    """
    rng = random.Random(seed)
    assignments = {}

    for kind in ("event", "non_event"):
        selected = [
            b["id"] for b in blocks if b["type"] == kind
        ]

        rng.shuffle(selected)

        n = len(selected)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        for bid in selected[:n_train]:
            assignments[bid] = "train"

        for bid in selected[n_train:n_train + n_val]:
            assignments[bid] = "val"

        for bid in selected[n_train + n_val:]:
            assignments[bid] = "test"

    assert len(assignments) == len(blocks)
    return assignments


# ============================================================
# Extract satellite-specific observations from shared split
# ============================================================

def extract_splits(raw, global_times, block_ids, assignments):
    """Use shared block assignments to select one spacecraft’s observations for each split."""
    lookup = {
        pd.Timestamp(t): assignments[int(bid)]
        for t, bid in zip(global_times, block_ids)
    }

    labels = np.array([
        lookup[pd.Timestamp(t)] for t in raw["times"]
    ])

    result = {}

    for split in SPLITS:
        idx = np.flatnonzero(labels == split)

        result[f"X_{split}"] = raw["X"][idx].copy()
        result[f"mask_{split}"] = raw["mask"][idx].copy()
        result[f"y_{split}_raw"] = raw["y"][idx].copy()
        result[f"t_{split}"] = raw["times"][idx].copy()

    return result


# ============================================================
# Shared AR padding
# ============================================================

def pad_splits(datasets, max_ars):
    """Pad every split to the common active-region dimension and update the masks."""
    for data in datasets.values():
        for split in SPLITS:
            X = data[f"X_{split}"]
            mask = data[f"mask_{split}"]

            N, A, F = X.shape

            padded_X = np.zeros(
                (N, max_ars, F), dtype=np.float32
            )
            padded_mask = np.zeros(
                (N, max_ars), dtype=bool
            )

            padded_X[:, :A] = X
            padded_mask[:, :A] = mask

            data[f"X_{split}"] = padded_X
            data[f"mask_{split}"] = padded_mask


# ============================================================
# Joint SHARP normalization (train-only, each holdout)
# ============================================================

def normalize_X(datasets):
    """
    Fit feature statistics using the combined PSP + SolO
    real AR training rows for THIS holdout only.
    """
    rows = []

    for data in datasets.values():
        X = data["X_train"]
        mask = data["mask_train"]

        if mask.any():
            rows.append(X[mask])

    if not rows:
        raise ValueError("No training AR observations")

    rows = np.concatenate(rows, axis=0).astype(np.float64)

    mu = np.nanmean(rows, axis=0)
    std = np.nanstd(rows, axis=0)

    mu[~np.isfinite(mu)] = 0.0
    std[~np.isfinite(std) | (std == 0)] = 1.0

    for data in datasets.values():
        for split in SPLITS:
            X = data[f"X_{split}"].copy()
            mask = data[f"mask_{split}"]

            X[mask] = ((X[mask] - mu) / std).astype(np.float32)
            X[~mask] = 0.0

            data[f"X_{split}"] = X

    return mu, std


# ============================================================
# Satellite-specific y normalization (train-only)
# ============================================================

def normalize_y(datasets):
    """Log-transform flux, standardize it with spacecraft-specific training statistics, and return those statistics."""
    stats = {}

    for sat, data in datasets.items():
        raw_train = data["y_train_raw"].astype(np.float64)

        if len(raw_train) == 0:
            raise ValueError(
                f"{sat}: empty training set for this holdout"
            )

        if (
            not np.all(np.isfinite(raw_train))
            or np.any(raw_train <= -Y_EPS)
        ):
            raise ValueError(f"{sat}: invalid Jlinlin targets")

        log_train = np.log(raw_train + Y_EPS)
        mu = np.mean(log_train)
        std = np.std(log_train)

        if not np.isfinite(std) or std == 0:
            std = 1.0

        stats[sat] = {
            "log_mean": mu,
            "log_std": std,
            "raw_train_min": np.min(raw_train),
            "raw_train_median": np.median(raw_train),
            "raw_train_max": np.max(raw_train),
            "log_train_min": np.min(log_train),
            "log_train_median": np.median(log_train),
            "log_train_max": np.max(log_train),
        }

        for split in SPLITS:
            raw = data[f"y_{split}_raw"].astype(np.float64)

            if (
                not np.all(np.isfinite(raw))
                or np.any(raw <= -Y_EPS)
            ):
                raise ValueError(
                    f"{sat}: invalid {split} Jlinlin values"
                )

            log_y = np.log(raw + Y_EPS)

            data[f"y_{split}_log"] = log_y.astype(np.float32)
            data[f"y_{split}"] = (
                (log_y - mu) / std
            ).astype(np.float32)

    return stats


# ============================================================
# Combine PSP + SolO observations within each split
# ============================================================

def combine_splits(datasets, split, rng):
    """Concatenate the same split from both spacecraft and apply a shared permutation to its arrays."""
    Xs, masks, ys, y_logs, y_raws, times, sats = (
        [], [], [], [], [], [], []
    )

    for sat, data in datasets.items():
        Xs.append(data[f"X_{split}"])
        masks.append(data[f"mask_{split}"])
        ys.append(data[f"y_{split}"])
        y_logs.append(data[f"y_{split}_log"])
        y_raws.append(data[f"y_{split}_raw"])
        times.append(data[f"t_{split}"])

        sats.append(np.full(
            len(data[f"t_{split}"]), sat, dtype="<U8"
        ))

    X = np.concatenate(Xs, axis=0)
    mask = np.concatenate(masks, axis=0)
    y = np.concatenate(ys, axis=0)
    y_log = np.concatenate(y_logs, axis=0)
    y_raw = np.concatenate(y_raws, axis=0)
    t = np.concatenate(times, axis=0)
    sat = np.concatenate(sats, axis=0)

    perm = rng.permutation(len(y))

    return (
        X[perm], mask[perm], y[perm],
        y_log[perm], y_raw[perm],
        t[perm], sat[perm]
    )


# ============================================================
# Leakage and shape checks
# ============================================================

def validate_splits(datasets):
    """Check the split arrays and verify that timestamps do not cross split boundaries."""
    split_times = {s: set() for s in SPLITS}

    for sat, data in datasets.items():
        for split in SPLITS:
            X = data[f"X_{split}"]
            mask = data[f"mask_{split}"]
            y = data[f"y_{split}"]
            t = data[f"t_{split}"]

            assert X.shape[:2] == mask.shape
            assert len(X) == len(y) == len(t)
            assert np.isfinite(X).all()
            assert np.isfinite(y).all()
            assert np.all(X[~mask] == 0)

            split_times[split].update(pd.to_datetime(t))

    for i, split1 in enumerate(SPLITS):
        for split2 in SPLITS[i + 1:]:
            overlap = split_times[split1] & split_times[split2]

            assert not overlap, (
                f"Cross-satellite leakage between "
                f"{split1} and {split2}: "
                f"{len(overlap)} timestamps"
            )


# ============================================================
# Load data ONCE before all repeated holdouts
# ============================================================

all_event_starts = []

for sat, config in SAT_CONFIGS.items():
    event_times = load_event_times(config)
    all_event_starts.extend(event_times)
    print(f"{sat.upper()}: {len(event_times)} SEP events")

event_windows = merge_event_windows(all_event_starts)

raw_datasets = {}

for sat, config in SAT_CONFIGS.items():
    raw_datasets[sat] = aggregate_sharp_by_time(
        config["combined_csv"], sat
    )

all_times = np.unique(np.concatenate([
    raw_datasets["psp"]["times"],
    raw_datasets["solo"]["times"]
]))

blocks, global_block_ids, global_times = build_shared_blocks(
    all_times, event_windows
)

global_max_ars = max(
    raw["X"].shape[1] for raw in raw_datasets.values()
)

print("\nGlobal setup:")
print("Unique timestamps:", len(global_times))
print("Event blocks:", sum(b["type"] == "event" for b in blocks))
print("Non-event blocks:", sum(b["type"] == "non_event" for b in blocks))
print("Maximum ARs:", global_max_ars)


# ============================================================
# MAIN: 10 repeated shared-block holdouts
# ============================================================

for i, SEED in enumerate(SEEDS):
    print("\n" + "=" * 65)
    print(f"HOLDOUT {i} / SEED {SEED}")
    print("=" * 65)

    # Independent randomization for this holdout.
    rng = np.random.default_rng(SEED)

    # 1. ONE global block assignment for PSP + SolO.
    assignments = split_shared_blocks(blocks, SEED)

    # 2. Extract independent satellite observations.
    datasets = {
        sat: extract_splits(
            raw, global_times, global_block_ids, assignments
        )
        for sat, raw in raw_datasets.items()
    }

    # 3. Pad to common AR dimension.
    pad_splits(datasets, global_max_ars)

    # 4. Fit shared X normalization from this holdout's
    #    combined PSP + SolO training observations.
    X_mean, X_std = normalize_X(datasets)

    # 5. Fit separate y normalization for each satellite.
    y_stats = normalize_y(datasets)

    # 6. Verify no exact timestamp crosses splits.
    validate_splits(datasets)

    # 7. Save normalization statistics FOR THIS HOLDOUT.
    pd.DataFrame({
        "feature": cols,
        "mean": X_mean,
        "std": X_std,
    }).to_csv(
        os.path.join(
            OUT_DIR,
            f"X_point_normalization_stats_combined_na_{i}.csv"
        ),
        index=False
    )

    pd.DataFrame(y_stats).T.to_csv(
        os.path.join(
            OUT_DIR,
            f"y_point_normalization_stats_by_satellite_na_{i}.csv"
        )
    )

    # 8. Combine PSP + SolO and save each split.
    for split in SPLITS:
        X, mask, y, y_log, y_raw, t, sat = combine_splits(
            datasets, split, rng
        )

        np.save(
            os.path.join(OUT_DIR, f"X_t_{split}_combined_na_{i}.npy"),
            X
        )

        np.save(
            os.path.join(OUT_DIR, f"mask_t_{split}_combined_na_{i}.npy"),
            mask
        )

        np.save(
            os.path.join(OUT_DIR, f"y_t_{split}_combined_na_{i}.npy"),
            y
        )

        np.save(
            os.path.join(
                OUT_DIR, f"y_t_{split}_combined_na_log_{i}.npy"
            ),
            y_log
        )

        np.save(
            os.path.join(
                OUT_DIR, f"y_t_{split}_combined_na_raw_{i}.npy"
            ),
            y_raw
        )

        np.save(
            os.path.join(
                OUT_DIR, f"times_t_{split}_combined_na_{i}.npy"
            ),
            t
        )

        np.save(
            os.path.join(
                OUT_DIR, f"sat_t_{split}_combined_na_{i}.npy"
            ),
            sat
        )

        print(
            f"{split:5s}: X={X.shape}, "
            f"PSP={np.sum(sat == 'psp')}, "
            f"SolO={np.sum(sat == 'solo')}"
        )

    # 9. Save global block assignments for reproducibility.
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
        os.path.join(
            OUT_DIR, f"shared_time_split_combined_na_{i}.csv"
        ),
        index=False
    )

    print(f"Holdout {i}: passed timestamp leakage checks.")

print("\nAll 10 combined repeated-holdout datasets saved.")
print("Output directory:", OUT_DIR)
