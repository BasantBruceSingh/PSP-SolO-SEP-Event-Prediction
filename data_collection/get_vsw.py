from pathlib import Path
import numpy as np
import pandas as pd
import cdflib
from tqdm import tqdm

input_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/Jlinlin_SolO.csv"
output_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/Jlinlin_SolO_with_VSW.csv"

def cdf_time_to_datetime(epoch_values):
    """
    Convert CDF epoch values to pandas UTC datetimes.
    """
    times = cdflib.cdfepoch.to_datetime(epoch_values)
    return pd.to_datetime(times, utc=True)

def read_cdf_times_and_velocity(cdf_path):
    """
    Read time and velocity from one SWA-PAS CDF file.
    Returns a dataframe with time, velocity components, and scalar VSW.
    """
    cdf = cdflib.CDF(str(cdf_path))

    epoch = cdf.varget('Epoch')
    time = cdf_time_to_datetime(epoch)

    velocity = cdf.varget('V_RTN')

    velocity = np.asarray(velocity)

    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(
            f"Expected velocity shape (N, 3), got {velocity.shape} for V_RTN"
        )

    df = pd.DataFrame({
        "time": time,
        "V_R": velocity[:, 0],
        "V_T": velocity[:, 1],
        "V_N": velocity[:, 2],
    })

    df["VSW"] = np.sqrt(df["V_R"]**2 + df["V_T"]**2 + df["V_N"]**2)
    df["source_file"] = str(cdf_path)

    return df


def find_candidate_cdf_files(root_dir, target_time, day_buffer=2):
    """
    Find candidate CDF files using dates embedded in filenames.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing year subdirectories.
    target_time : str or pd.Timestamp
        Desired timestamp.
    day_buffer : int
        Include files from target date +/- this many days.

    Returns
    -------
    list[Path]
        Candidate CDF files.
    """
    root_dir = Path(root_dir)
    target_time = pd.Timestamp(target_time)

    target_dates = [
        (target_time + pd.Timedelta(days=d)).strftime("%Y%m%d")
        for d in range(-day_buffer, day_buffer + 1)
    ]

    files = []

    for yyyymmdd in target_dates:
        year = yyyymmdd[:4]
        year_dir = root_dir / year

        if not year_dir.exists():
            continue

        # Fast filename pattern search
        files.extend(sorted(year_dir.glob(f"*{yyyymmdd}*.cdf")))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No CDF files found for dates {target_dates} under {root_dir}"
        )

    return files


def find_closest_swa_pas_reading(
    root_dir,
    target_time,
):
    """
    Find closest SWA-PAS reading to target_time.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing year subdirectories of CDF files.
    target_time : str or pd.Timestamp
        Desired timestamp.

    Returns
    -------
    closest_row : pandas Series
        Closest measurement with time, V_R, V_T, V_N, VSW, source_file, dt.
    """
    target_time = pd.Timestamp(target_time, tz="UTC")

    files = find_candidate_cdf_files(root_dir, target_time)

    all_chunks = []

    for f in files:
    
        try:
            df = read_cdf_times_and_velocity(f)

            # Optional speed-up: keep only rows somewhat near the target.
            # This avoids concatenating huge amounts if files are large.
            df["dt"] = (df["time"] - target_time).abs()
            all_chunks.append(df)

        except Exception as e:
            print(f"Skipping {f}: {e}")

    if len(all_chunks) == 0:
        raise RuntimeError("No usable CDF files were read.")

    all_data = pd.concat(all_chunks, ignore_index=True)

    idx = all_data["dt"].idxmin()
    closest = all_data.loc[idx].copy()

    return closest

df = pd.read_csv(input_csv, parse_dates=["time"])

for i, row in tqdm(df.iterrows(), total=len(df)):
    target_time = row["time"]
    try:
        closest_reading = find_closest_swa_pas_reading(
            root_dir="/Users/basant/Desktop/code/stellar_local/level2",
            target_time=target_time,
        )
        df.at[i, "VSW"] = closest_reading["VSW"]

    except Exception as e:
        print(f"Could not find reading for {target_time}: {e}")
        df.at[i, "VSW"] = np.nan

df.to_csv(output_csv, index=False)