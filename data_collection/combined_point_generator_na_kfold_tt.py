import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in range(10):
    SEED = SEEDS[i]
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
    # Time / catalogue helpers
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
        This keeps overlapping/nearby SEP event windows together.
        """
        windows = [(t, t + window) for t in sorted(start_times)]

        merged = []
        for start, end in windows:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        return [(s, e) for s, e in merged]


    # -----------------------------
    # Build point-level AR tensors
    # -----------------------------
    def aggregate_sharp_by_time(combined_csv, satellite_name):
        """
        Converts AR-level SHARP rows into one point per satellite time.

        Returns:
            X_time:    [T, A_sat, F]
            mask_time: [T, A_sat]
            y_time:    [T] raw Jlinlin
            times:     [T]
        """
        df = pd.read_csv(combined_csv)

        df["time"] = parse_time_series(df["time"])
        df = df.dropna()
        df = df.sort_values("time").reset_index(drop=True)

        missing = [c for c in cols + ["Jlinlin"] if c not in df.columns]
        if missing:
            raise ValueError(f"{combined_csv} missing columns: {missing}")

        groups = list(df.groupby("time", sort=True))

        if len(groups) == 0:
            raise ValueError(f"No valid time groups found in {combined_csv}")

        max_num_ARs = max(len(g) for _, g in groups)
        num_features = len(cols)

        X_time = []
        mask_time = []
        y_time = []
        times = []

        for time, g in tqdm(groups, desc=f"Building AR tensors for {satellite_name}"):
            features = g[cols].to_numpy(dtype=np.float64)
            n_ar = len(features)

            X_t = np.zeros((max_num_ARs, num_features), dtype=np.float64)
            mask_t = np.zeros((max_num_ARs,), dtype=bool)

            X_t[:n_ar, :] = features
            mask_t[:n_ar] = True

            # Raw continuous target. Do not threshold here.
            y_t = float(g["Jlinlin"].iloc[0])

            X_time.append(X_t)
            mask_time.append(mask_t)
            y_time.append(y_t)
            times.append(time)

        X_time = np.stack(X_time).astype(np.float32)
        mask_time = np.stack(mask_time).astype(bool)
        y_time = np.array(y_time, dtype=np.float32)
        times = np.array(times, dtype="datetime64[ns]")

        return X_time, mask_time, y_time, times


    # -----------------------------
    # Blocking / splitting
    # -----------------------------
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
            # Advance past event windows that ended before this time.
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

        # Chunk non-events into <=10-day blocks.
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


    def split_blocks(event_blocks, non_event_blocks, train_frac=0.7):
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

            train = blocks[:n_train]
            test = blocks[n_train:]

            return train, test

        e_train, e_test = split_one(event_blocks)
        n_train, n_test = split_one(non_event_blocks)

        train_blocks = e_train + n_train
        test_blocks = e_test + n_test

        random.shuffle(train_blocks)
        random.shuffle(test_blocks)

        return train_blocks, test_blocks


    # -----------------------------
    # Split extraction / padding
    # -----------------------------
    def points_from_blocks(X_time, mask_time, y_time, times, blocks, satellite_name):
        """
        Extract point-level samples from split blocks.

        Returns:
            X:    [N, A_sat, F]
            mask: [N, A_sat]
            y:    [N]
            t:    [N]
            sat:  [N]
        """
        blocks = [b for b in blocks if len(b) > 0]

        if len(blocks) == 0:
            return (
                np.empty((0, X_time.shape[1], X_time.shape[2]), dtype=np.float32),
                np.empty((0, mask_time.shape[1]), dtype=bool),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype="<U8"),
            )

        indices = np.concatenate([np.array(b, dtype=int) for b in blocks])

        X = X_time[indices]
        mask = mask_time[indices]
        y = y_time[indices]
        t = times[indices]
        sat = np.array([satellite_name] * len(indices), dtype="<U8")

        return X, mask, y, t, sat


    def pad_point_split(X, mask, global_max_ARs):
        """
        Pads point-level X/mask:
            X:    [N, A, F]
            mask: [N, A]
        """
        N, A, F = X.shape

        if A == global_max_ARs:
            return X, mask

        X_padded = np.zeros((N, global_max_ARs, F), dtype=X.dtype)
        mask_padded = np.zeros((N, global_max_ARs), dtype=bool)

        X_padded[:, :A, :] = X
        mask_padded[:, :A] = mask

        return X_padded, mask_padded


    # -----------------------------
    # Build per-satellite data
    # -----------------------------
    def build_satellite_dataset(satellite_name, config):
        if config["catalogue_type"] == "csv":
            start_times = load_psp_start_times(
                config["catalogue_path"],
                config["start_col"],
            )
        elif config["catalogue_type"] == "xlsx":
            start_times = load_solo_start_times(
                config["catalogue_path"],
                start_col_index=config["start_col_index"],
                skip_rows_before=config["skip_rows_before"],
            )
        else:
            raise ValueError(f"Unknown catalogue_type: {config['catalogue_type']}")

        event_windows = merge_event_windows(start_times, EVENT_WINDOW)

        X_time, mask_time, y_time, times = aggregate_sharp_by_time(
            config["combined_csv"],
            satellite_name=satellite_name,
        )

        event_blocks, non_event_blocks = make_blocks_from_event_windows(
            times,
            event_windows,
        )

        print(f"\n{satellite_name.upper()} block summary")
        print("time points:", len(times))
        print("max ARs:", X_time.shape[1])
        print("num features:", X_time.shape[2])
        print("event blocks:", len(event_blocks))
        print("non-event blocks:", len(non_event_blocks))
        print("event points:", sum(len(b) for b in event_blocks))
        print("non-event points:", sum(len(b) for b in non_event_blocks))

        train_blocks, test_blocks = split_blocks(
            event_blocks,
            non_event_blocks,
        )

        X_train, mask_train, y_train_raw, t_train, sat_train = points_from_blocks(
            X_time,
            mask_time,
            y_time,
            times,
            train_blocks,
            satellite_name,
        )

        X_test, mask_test, y_test_raw, t_test, sat_test = points_from_blocks(
            X_time,
            mask_time,
            y_time,
            times,
            test_blocks,
            satellite_name,
        )

        return {
            "X_time": X_time,
            "mask_time": mask_time,
            "times": times,

            "X_train": X_train,
            "mask_train": mask_train,
            "y_train_raw": y_train_raw,
            "t_train": t_train,
            "sat_train": sat_train,

            "X_test": X_test,
            "mask_test": mask_test,
            "y_test_raw": y_test_raw,
            "t_test": t_test,
            "sat_test": sat_test,
        }


    # -----------------------------
    # Feature normalization
    # -----------------------------
    def fit_X_normalization_from_train(datasets):
        """
        Fit X normalization from real AR rows in train splits only.
        """
        train_real_rows = []

        for _, data in datasets.items():
            X_train = data["X_train"]
            mask_train = data["mask_train"]

            if len(X_train) == 0:
                continue

            train_real_rows.append(X_train[mask_train])

        if len(train_real_rows) == 0:
            raise ValueError("No real train AR rows found for X normalization.")

        train_real_rows = np.concatenate(train_real_rows, axis=0)

        X_mean = np.nanmean(train_real_rows, axis=0)
        X_std = np.nanstd(train_real_rows, axis=0)

        X_mean[~np.isfinite(X_mean)] = 0.0
        X_std[~np.isfinite(X_std)] = 1.0
        X_std[X_std == 0] = 1.0

        return X_mean.astype(np.float32), X_std.astype(np.float32)

    # -----------------------------
    # Feature normalization
    # -----------------------------
    def fit_X_normalization_from_train(datasets):
        """
        Fit X z-score normalization using real AR rows from train splits only.

        This combines PSP + SolO train rows to get one shared X mean/std:
            X_z = (X - X_mean) / X_std

        Padded AR rows are ignored via mask.
        """
        train_real_rows = []

        for sat, data in datasets.items():
            X_train = data["X_train"]
            mask_train = data["mask_train"]

            if len(X_train) == 0:
                continue

            # X_train[mask_train] has shape [num_real_AR_rows, num_features]
            real_rows = X_train[mask_train]

            if len(real_rows) > 0:
                train_real_rows.append(real_rows)

        if len(train_real_rows) == 0:
            raise ValueError("No real train AR rows found for X normalization.")

        train_real_rows = np.concatenate(train_real_rows, axis=0).astype(np.float64)

        X_mean = np.nanmean(train_real_rows, axis=0)
        X_std = np.nanstd(train_real_rows, axis=0)

        X_mean[~np.isfinite(X_mean)] = 0.0
        X_std[~np.isfinite(X_std)] = 1.0
        X_std[X_std == 0] = 1.0

        return X_mean.astype(np.float32), X_std.astype(np.float32)


    def apply_X_normalization_to_splits(datasets, X_mean, X_std):
        """
        Apply train-only X z-score normalization to real AR rows in train/val/test.

        Padded AR rows remain zero.
        """
        for sat, data in datasets.items():
            for split in ["train", "test"]:
                X_key = f"X_{split}"
                mask_key = f"mask_{split}"

                X = data[X_key].copy().astype(np.float32)
                mask = data[mask_key]

                if len(X) > 0:
                    X[mask] = (X[mask] - X_mean) / X_std

                # Keep padded rows exactly zero.
                X[~mask] = 0.0

                data[X_key] = X.astype(np.float32)


    # -----------------------------
    # Y normalization
    # -----------------------------
    def zscore_y_per_satellite_train_stats(datasets, eps=Y_EPS):
        """
        For each satellite separately:
        1. log-transform raw Jlinlin: log(y + eps)
        2. fit mean/std using that satellite's train split only
        3. apply those train stats to train/val/test

        This prevents PSP and SolO from being normalized against each other's target scale.
        """
        stats = {}

        for sat, data in datasets.items():
            y_train_raw = data["y_train_raw"].astype(np.float64)

            if len(y_train_raw) == 0:
                raise ValueError(f"{sat}: empty y_train_raw")

            if np.nanmin(y_train_raw) <= -eps:
                raise ValueError(
                    f"{sat}: y contains values <= -eps, so log(y + eps) is invalid. "
                    f"min y = {np.nanmin(y_train_raw)}"
                )

            y_train_log = np.log(y_train_raw + eps)

            y_mean = np.nanmean(y_train_log)
            y_std = np.nanstd(y_train_log)

            if not np.isfinite(y_std) or y_std == 0:
                y_std = 1.0

            stats[sat] = {
                "log_mean": y_mean,
                "log_std": y_std,
                "raw_train_min": np.nanmin(y_train_raw),
                "raw_train_median": np.nanmedian(y_train_raw),
                "raw_train_max": np.nanmax(y_train_raw),
                "log_train_min": np.nanmin(y_train_log),
                "log_train_median": np.nanmedian(y_train_log),
                "log_train_max": np.nanmax(y_train_log),
            }

            for split in ["train", "test"]:
                raw_key = f"y_{split}_raw"
                log_key = f"y_{split}_log"
                z_key = f"y_{split}"

                y_raw = data[raw_key].astype(np.float64)

                if len(y_raw) == 0:
                    data[log_key] = np.empty((0,), dtype=np.float32)
                    data[z_key] = np.empty((0,), dtype=np.float32)
                    continue

                if np.nanmin(y_raw) <= -eps:
                    raise ValueError(
                        f"{sat} {split}: y contains values <= -eps, so log(y + eps) is invalid. "
                        f"min y = {np.nanmin(y_raw)}"
                    )

                y_log = np.log(y_raw + eps)
                y_z = (y_log - y_mean) / y_std

                data[log_key] = y_log.astype(np.float32)
                data[z_key] = y_z.astype(np.float32)

            z_train = data["y_train"]

            print(f"\n{sat} y normalization check")
            print("train z min/mean/max:", np.nanmin(z_train), np.nanmean(z_train), np.nanmax(z_train))
            print("train z < 0:", np.sum(z_train < 0))
            print("train z > 0:", np.sum(z_train > 0))

        return stats


    # -----------------------------
    # Combine splits
    # -----------------------------
    def combine_splits(datasets, split):
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
            sats.append(data[f"sat_{split}"])

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


    # -----------------------------
    # Main
    # -----------------------------
    datasets = {}

    for sat_name, config in SAT_CONFIGS.items():
        datasets[sat_name] = build_satellite_dataset(sat_name, config)

    # Pad PSP/SolO to the same AR dimension.
    global_max_ARs = max(data["X_time"].shape[1] for data in datasets.values())
    print("\nGlobal max ARs:", global_max_ARs)

    for sat_name, data in datasets.items():
        for split in ["train", "test"]:
            X_key = f"X_{split}"
            mask_key = f"mask_{split}"

            X_padded, mask_padded = pad_point_split(
                data[X_key],
                data[mask_key],
                global_max_ARs,
            )

            data[X_key] = X_padded.astype(np.float32)
            data[mask_key] = mask_padded.astype(bool)

    # Normalize X using combined train real AR rows only.
    X_mean, X_std = fit_X_normalization_from_train(datasets)
    apply_X_normalization_to_splits(datasets, X_mean, X_std)

    pd.DataFrame({
        "feature": cols,
        "mean": X_mean,
        "std": X_std,
    }).to_csv(
        os.path.join(OUT_DIR, "X_point_normalization_stats_combined_na.csv"),
        index=False,
    )

    # Normalize y per satellite using train-only stats.
    y_stats = zscore_y_per_satellite_train_stats(datasets)

    pd.DataFrame({
        "feature": cols,
        "mean": X_mean,
        "std": X_std,
    }).to_csv(
        os.path.join(OUT_DIR, "X_point_normalization_stats_combined_na.csv"),
        index=False,
    )

    print("\nY normalization stats, train-only:")
    for sat, st in y_stats.items():
        print(f"{sat}: log_mean={st['log_mean']:.6g}, log_std={st['log_std']:.6g}")

    # Combine PSP + SolO splits.
    X_train, mask_train, y_train, y_train_log, y_train_raw, t_train, sat_train = combine_splits(datasets, "train")
    X_test, mask_test, y_test, y_test_log, y_test_raw, t_test, sat_test = combine_splits(datasets, "test")

    print("\nCombined shapes")
    print("X_train:", X_train.shape, "mask_train:", mask_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "mask_test:", mask_test.shape, "y_test:", y_test.shape)

    print("\nSatellite counts")
    for split_name, sat_arr in [("train", sat_train), ("test", sat_test)]:
        vals, counts = np.unique(sat_arr, return_counts=True)
        print(split_name, dict(zip(vals, counts)))

    # -----------------------------
    # Save outputs
    # -----------------------------
    np.save(os.path.join(OUT_DIR, f"X_t_train_combined_na_kfold_tt_{i}.npy"), X_train)
    np.save(os.path.join(OUT_DIR, f"X_t_test_combined_na_kfold_tt_{i}.npy"), X_test)

    np.save(os.path.join(OUT_DIR, f"mask_t_train_combined_na_kfold_tt_{i}.npy"), mask_train)
    np.save(os.path.join(OUT_DIR, f"mask_t_test_combined_na_kfold_tt_{i}.npy"), mask_test)

    np.save(os.path.join(OUT_DIR, f"y_t_train_combined_na_kfold_tt_{i}.npy"), y_train)
    np.save(os.path.join(OUT_DIR, f"y_t_test_combined_na_kfold_tt_{i}.npy"), y_test)

    np.save(os.path.join(OUT_DIR, f"sat_t_train_combined_na_kfold_tt_{i}.npy"), sat_train)
    np.save(os.path.join(OUT_DIR, f"sat_t_test_combined_na_kfold_tt_{i}.npy"), sat_test)

    pd.DataFrame(y_stats).T.to_csv(
        os.path.join(OUT_DIR, f"y_point_normalization_stats_by_satellite_na_kfold_tt_{i}.csv")
    )

    print(f"\nSaved combined non-aggregated point dataset to {OUT_DIR}/")