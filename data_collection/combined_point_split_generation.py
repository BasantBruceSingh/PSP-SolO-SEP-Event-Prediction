"""Build independently saved PSP/SolO aggregated-SHARP datasets with shared time splits.

Run from the directory containing SHARP_to_PSP_times.csv,
SHARP_to_SolO_times.csv, psp_isois_eventlist_v21.csv, SolO_catalogue.xlsx.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 0
OUT_DIR = Path('npy')
EVENT_WINDOW = pd.Timedelta(days=10)
Y_EPS = 1e-5
COLS = ['USFLUXL', 'R_VALUE', 'MEANGBL_GMM', 'USFLUXZ', 'MEANGBZ', 'CMASKL', 'LAT_FWT', 'LON_FWT']
RMS_COLS = COLS[:6]
MEAN_COLS = COLS[6:]
SAT_CONFIGS = {
    'psp': {'combined_csv': 'SHARP_to_PSP_times.csv',
            'catalogue_path': 'psp_isois_eventlist_v21.csv',
            'catalogue_type': 'csv', 'start_col': ' Start (UTC)     '},
    'solo': {'combined_csv': 'SHARP_to_SolO_times.csv',
             'catalogue_path': 'SolO_catalogue.xlsx',
             'catalogue_type': 'xlsx', 'start_col_index': 28,
             'skip_rows_before': 4},
}
SPLITS = ('train', 'val', 'test')


def parse_times(s):
    # Parse all times into one consistent UTC-naive representation.
    """Parse mixed-format timestamps as UTC, coercing invalid values to NaT and dropping timezone metadata."""
    return pd.to_datetime(s.astype(str).str.strip(), format='mixed', errors='coerce', utc=True).dt.tz_localize(None)


def load_event_starts(cfg):
    """Read the configured event catalogue and return parsed SEP onset timestamps."""
    if cfg['catalogue_type'] == 'csv':
        raw = pd.read_csv(cfg['catalogue_path'])[cfg['start_col']]
    else:
        df = pd.read_excel(cfg['catalogue_path'])
        raw = df.iloc[cfg['skip_rows_before']:, cfg['start_col_index']]
    times = parse_times(raw).dropna().sort_values()
    if times.empty:
        raise ValueError(f"No event start times in {cfg['catalogue_path']}")
    return times.tolist()


def aggregate_sharp_by_time(csv, satellite):
    """Aggregate the SHARP active-region features for each observation timestamp."""
    df = pd.read_csv(csv)
    required = ['time', 'Jlinlin'] + COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{csv} missing columns: {missing}')
    df['time'] = parse_times(df['time'])
    df = df.dropna(subset=['time', 'Jlinlin']).sort_values('time')
    if df.empty:
        raise ValueError(f'No usable samples in {csv}')
    X, y, times = [], [], []
    for t, g in tqdm(df.groupby('time', sort=True), desc=f'Aggregating {satellite}'):
        rms = np.sqrt(np.nanmean(g[RMS_COLS].to_numpy(dtype=np.float64) ** 2, axis=0))
        mean = np.nanmean(g[MEAN_COLS].to_numpy(dtype=np.float64), axis=0)
        features = np.concatenate((rms, mean))
        if not np.isfinite(features).all():
            raise ValueError(f'{satellite} has nonfinite aggregate features at {t}')
        targets = g['Jlinlin'].to_numpy(dtype=np.float64)
        if not np.allclose(targets, targets[0], rtol=1e-5, atol=1e-8):
            raise ValueError(f'{satellite}: inconsistent Jlinlin at {t}')
        X.append(features)
        y.append(targets[0])
        times.append(t)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(times, dtype='datetime64[ns]'))


def merge_event_windows(event_starts):
    """Merge overlapping SEP windows into contiguous event intervals."""
    merged = []
    for t in sorted(event_starts):
        end = t + EVENT_WINDOW
        if not merged or t > merged[-1][1]:
            merged.append([t, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def shared_blocks(all_times, event_windows):
    """Every unique time appears once; each block has a single split assignment."""
    times = pd.to_datetime(all_times)
    event_blocks = []
    non_event_blocks = []
    j = 0
    active_event = None
    non_event = []
    for idx, t in enumerate(times):
        while j < len(event_windows) and t > event_windows[j][1]:
            j += 1
        in_event = j < len(event_windows) and event_windows[j][0] <= t <= event_windows[j][1]
        if in_event:
            if non_event:
                non_event_blocks.append(non_event)
                non_event = []
            if active_event != j:
                event_blocks.append([])
                active_event = j
            event_blocks[-1].append(idx)
        else:
            active_event = None
            if non_event and t > times[non_event[0]] + EVENT_WINDOW:
                non_event_blocks.append(non_event)
                non_event = []
            non_event.append(idx)
    if non_event:
        non_event_blocks.append(non_event)
    return event_blocks, non_event_blocks


def split_blocks(event_blocks, non_event_blocks, seed=SEED):
    """Shuffle event and non-event blocks separately and assign each block to a split."""
    rng = random.Random(seed)
    assignments = {}
    for category, blocks in (('event', event_blocks), ('non_event', non_event_blocks)):
        ids = list(range(len(blocks)))
        rng.shuffle(ids)
        n_train = int(len(ids) * .70)
        n_val = int(len(ids) * .15)
        for pos, block_id in enumerate(ids):
            split = 'train' if pos < n_train else ('val' if pos < n_train+n_val else 'test')
            assignments[(category, block_id)] = split
    return assignments


def global_split_table(all_times, event_blocks, non_event_blocks, assignments):
    """Build the sorted timestamp-to-split manifest and verify its timestamps are unique."""
    rows = []
    for cat, blocks in (('event', event_blocks), ('non_event', non_event_blocks)):
        for i, block in enumerate(blocks):
            for idx in block:
                rows.append((pd.Timestamp(all_times[idx]), f'{cat}_{i}', cat, assignments[(cat, i)]))
    result = pd.DataFrame(rows, columns=['time', 'block_id', 'block_type', 'split']).sort_values('time')
    assert len(result) == len(all_times)
    assert result['time'].is_unique
    return result


def build_satellite_outputs(sat, X, y_raw, times, assignments):
    """Apply the timestamp assignments, normalize from training data, and save one spacecraft dataset."""
    split_by_time = assignments.set_index('time')['split']
    sat_splits = split_by_time.loc[pd.to_datetime(times)].to_numpy()
    train = sat_splits == 'train'
    if not train.any():
        raise ValueError(f'{sat}: empty train split; adjust split size/seed')
    y_train = y_raw[train].astype(np.float64)
    if not np.isfinite(y_raw).all() or np.any(y_raw <= -Y_EPS):
        raise ValueError(f'{sat}: invalid values for log(Jlinlin + eps)')
    log_train = np.log(y_train + Y_EPS)
    mu, std = float(log_train.mean()), float(log_train.std())
    if std == 0 or not np.isfinite(std):
        std = 1.0
    for split in SPLITS:
        select = sat_splits == split
        sx, sy, st = X[select], y_raw[select], times[select]
        sy_log = np.log(sy.astype(np.float64) + Y_EPS).astype(np.float32)
        sy_z = ((sy_log.astype(np.float64) - mu) / std).astype(np.float32)
        # Independent satellite files; retain naming convention without "combined".
        for name, arr in (
            (f'X_t_{split}_{sat}.npy', sx),
            (f'y_t_{split}_{sat}.npy', sy_z),
            (f'y_t_{split}_{sat}_raw.npy', sy),
            (f'y_t_{split}_{sat}_log.npy', sy_log),
            (f'times_t_{split}_{sat}.npy', st),
            (f'sat_t_{split}_{sat}.npy', np.full(len(st), sat, dtype='<U8')),
        ):
            np.save(OUT_DIR / name, arr)
        print(f'{sat.upper():4} {split:5}: {len(sy):6} observations; X={sx.shape}')
    pd.DataFrame([{'satellite': sat, 'log_mean': mu, 'log_std': std, 'eps': Y_EPS}]).to_csv(
        OUT_DIR / f'y_normalization_stats_{sat}.csv', index=False)
    return sat_splits


def main():
    """Build the shared split manifest and save aggregated pointwise arrays for both spacecraft."""
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    data = {}
    starts = []
    for sat, cfg in SAT_CONFIGS.items():
        data[sat] = aggregate_sharp_by_time(cfg['combined_csv'], sat)
        starts.extend(load_event_starts(cfg))
    all_times = np.unique(np.concatenate([v[2] for v in data.values()]))
    windows = merge_event_windows(starts)
    event_blocks, non_event_blocks = shared_blocks(all_times, windows)
    assignments = split_blocks(event_blocks, non_event_blocks)
    split_table = global_split_table(all_times, event_blocks, non_event_blocks, assignments)
    split_table.to_csv(OUT_DIR / 'shared_time_split_aggregated.csv', index=False)
    print(f'Shared blocks: {len(event_blocks)} event, {len(non_event_blocks)} non-event')
    for sat, (X, y, times) in data.items():
        build_satellite_outputs(sat, X, y, times, split_table)
    # Explicitly check global cross-satellite split consistency.
    merged = []
    for sat, (_, _, times) in data.items():
        lookup = split_table.set_index('time')['split']
        merged.append(pd.DataFrame({'time': pd.to_datetime(times), 'sat': sat,
                                    'split': lookup.loc[pd.to_datetime(times)].to_numpy()}))
    check = pd.concat(merged, ignore_index=True)
    assert (check.groupby('time')['split'].nunique() == 1).all(), 'Cross-satellite time leakage!'
    print('Cross-satellite exact-timestamp leakage check: PASSED')
    print(f'Wrote separate PSP/SolO aggregated datasets to {OUT_DIR}/')


if __name__ == '__main__':
    main()
