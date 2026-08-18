import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

cadence_hours = 1 # cadence of final measurements

output_csv = "SHARP_to_PSP_times.csv"

# Load psp data
psp_csv = "processed_psp.csv"
df_psp = pd.read_csv(psp_csv, parse_dates=["time","offset_time"])

# Load and preprocess SHARP data
SHARP_csv_file = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP.csv"
SHARP_columns = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL", "LAT_FWT", "LON_FWT"]

df_SHARP = pd.read_csv(SHARP_csv_file, parse_dates=["T_OBS"])
df_SHARP = df_SHARP.dropna(subset=["T_OBS"]+SHARP_columns)
df_SHARP = df_SHARP.drop_duplicates()
df_SHARP = df_SHARP.sort_values(by=["T_OBS", "ARPNUM"])

SHARP_times = df_SHARP["T_OBS"].values

def find_SHARP_for_psp_time(target_time, SHARP_times, window_hours=1):
    '''
    Find the indices of SHARP measurements within a window around a given psp time.
    If none exist, return the indices of the closest SHARP measurement before and after the target time.
    '''
    closest_SHARP_index = 0
    min_time_diff = float('inf')

    # Binary search to find the range of SHARP times within the window around target_time
    if SHARP_times[0] - pd.Timedelta(hours=window_hours) > target_time:
        return 0, 0
    if SHARP_times[-1] + pd.Timedelta(hours=window_hours) < target_time:
        return len(SHARP_times) - 1, len(SHARP_times) - 1

    # Find bottom of the window
    bottom_SHARP_index = None
    low, high = 0, len(SHARP_times) - 1
    while low < high:
        mid = (low + high) // 2
        window_start = target_time - pd.Timedelta(hours=window_hours)
        # If the mid sharp time is below the window, move the low pointer up
        if SHARP_times[mid] < window_start:
            low = mid + 1
        # If the element at mid is the first element in the window, we found the bottom index
        elif SHARP_times[mid] > window_start and mid > 0 and SHARP_times[mid - 1] < window_start:
            bottom_SHARP_index = mid
            break
        # If the element at mid is the first element in the window, and is also the first element in the SHARP times, we found the bottom index
        elif mid == 0 and SHARP_times[mid] >= window_start:
            bottom_SHARP_index = mid
            break
        # If the element at mid is within the window, and is not the first element in the window, move the high pointer down
        else:
            high = mid - 1
    
    if bottom_SHARP_index is None:
        bottom_SHARP_index = low

    # Find top of the window
    low, high = 0, len(SHARP_times) - 1
    top_SHARP_index = None
    while low < high:
        mid = (low + high) // 2
        window_end = target_time + pd.Timedelta(hours=window_hours)
        # If the mid sharp time is above the window, move the high pointer down
        if SHARP_times[mid] > window_end:
            high = mid - 1
        # If the element at mid is the last element in the window, we found the top index
        elif SHARP_times[mid] < window_end and mid < len(SHARP_times) - 1 and SHARP_times[mid + 1] > window_end:
            top_SHARP_index = mid
            break
        # If the element at mid is the last element in the window, and is also the last element in the SHARP times, we found the top index
        elif mid == len(SHARP_times) - 1 and SHARP_times[mid] <= window_end:
            top_SHARP_index = mid
            break
        # If the element at mid is within the window, and is not the last element in the window, move the low pointer up
        else:
            low = mid + 1

    if top_SHARP_index is None:
        top_SHARP_index = high

    return bottom_SHARP_index, top_SHARP_index

def test_find_SHARP_for_psp_time():
    '''
    Test the find_SHARP_for_psp_time function with various cases.
    '''
    SHARP_times = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 02:00:00",
        "2024-01-01 03:00:00",
        "2024-01-01 03:00:00",
        "2024-01-01 04:00:00",
    ])
    target_time = pd.to_datetime("2024-01-01 02:30:00")
    window_hours = 1

    bottom_index, top_index = find_SHARP_for_psp_time(target_time, SHARP_times, window_hours)
    assert bottom_index == 2, f"Expected bottom index 2, got {bottom_index}"
    assert top_index == 4, f"Expected top index 4, got {top_index}"

    SHARP_times = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 02:00:00",
        "2024-01-01 03:00:00",
        "2024-01-01 04:00:00",
    ])
    target_time = pd.to_datetime("2024-01-01 01:00:01")
    window_hours = 2
    bottom_index, top_index = find_SHARP_for_psp_time(target_time, SHARP_times, window_hours)
    assert bottom_index == 0, f"Expected bottom index 0, got {bottom_index}"
    assert top_index == 3, f"Expected top index 3, got {top_index}"

    SHARP_times = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 02:00:00",  
        "2024-01-01 03:00:00",
        "2024-01-01 04:00:00",
    ])
    target_time = pd.to_datetime("2024-01-01 07:30:00")
    window_hours = 1
    bottom_index, top_index = find_SHARP_for_psp_time(target_time, SHARP_times, window_hours)
    assert bottom_index == 4, f"Expected bottom index 4, got {bottom_index}"
    assert top_index == 4, f"Expected top index 4, got {top_index}"

    print("All tests passed for find_SHARP_for_psp_time!")

test_find_SHARP_for_psp_time()

def compute_mean_flux(df, time, window_hrs=12):
    mask = (pd.to_datetime(df["time"], format="mixed") >= pd.to_datetime(time)) & (pd.to_datetime(df["time"], format="mixed") <= pd.to_datetime(time) + pd.Timedelta(hours=window_hrs))
    if not mask.any():
        return np.nan
    return np.nanmean(df.loc[mask, "psp_epilo_jlinlin_flux"])

def compute_mean_flux(times, linlin, time, window_hrs=12):
    start = np.datetime64(pd.to_datetime(time), "ns")
    end = np.datetime64(pd.to_datetime(time) + pd.Timedelta(hours=window_hrs), "ns")

    left = np.searchsorted(times, start, side="left")
    right = np.searchsorted(times, end, side="right")  # inclusive end

    if left == right:
        return np.nan

    return np.nanmean(linlin[left:right])
    
def compute_mean_flux_cadence(times, smoothed_linlin, time, cadence_hours):
    start = np.datetime64(pd.to_datetime(time) - pd.Timedelta(hours=cadence_hours/2), "ns")
    end = np.datetime64(pd.to_datetime(time) + pd.Timedelta(hours=cadence_hours/2), "ns")

    left = np.searchsorted(times, start, side="left")
    right = np.searchsorted(times, end, side="left")  # exclusive end

    if left == right:
        return np.nan

    return np.nanmean(smoothed_linlin[left:right])

# Find the mean SHARP measurement for each psp time and store the results in a csv file
results = []
times = df_psp["time"].to_numpy(dtype="datetime64[ns]")
linlin = df_psp["psp_epilo_jlinlin_flux"].to_numpy(dtype=np.float64)
smoothed_linlin = [compute_mean_flux(times, linlin, time) for time in times]
count = 0

for offset_psp_time, psp_time, Jlinlin, footprint_valid, lon, r, footprint_lon in tqdm(df_psp[["offset_time", "time", "psp_epilo_jlinlin_flux", "footprint_valid", "psp_ephem_features_HGS_Lon", "psp_ephem_features_HCI_R", "psp_footpoint_hgs_lon"]].values, desc="Processing psp times"):
    if count % cadence_hours != 0:
        count += 1
        continue

    count += 1

    # Skip if the angular separation between the psp footpoint and Earth is too large, since the SHARP measurements may not be relevant for this psp time
    if not footprint_valid:
        continue

    bottom_index, top_index = find_SHARP_for_psp_time(offset_psp_time, SHARP_times, cadence_hours/2)
    bottom_sharp_time = df_SHARP.iloc[bottom_index]["T_OBS"]
    top_sharp_time = df_SHARP.iloc[top_index]["T_OBS"]

    # Calculate the mean SHARP measurements for the window (for each active region)
    sharp_values = df_SHARP.iloc[bottom_index:top_index+1][SHARP_columns + ["ARPNUM"]]

    # Maps ARPNUM to the SHARP measurements for that active region in the window around the psp time. If there are no SHARP measurements for an active region in the window, it will not be included in the dictionary.
    AR_dict = {}
    for i, row in sharp_values.iterrows():
        ARPNUM = row["ARPNUM"]

        if ARPNUM not in AR_dict:
            AR_dict[ARPNUM] = {col: [] for col in SHARP_columns}

        for col in SHARP_columns:
            AR_dict[ARPNUM][col].append(row[col])

        if row.isnull().any():
            print(f"Warning: NaN values found in SHARP measurements for time {row['T_OBS']}. Skipping this row.")
            continue
        
    # Produce row of new df
    for ARPNUM, measurements in AR_dict.items():
        row = {"ARPNUM": ARPNUM}
        for col in SHARP_columns:
            row[col] = np.mean(measurements[col])
        result_dict = {"time": psp_time, "Jlinlin": compute_mean_flux_cadence(times, smoothed_linlin, psp_time, cadence_hours), "Jlinlin_raw":Jlinlin, "ARPNUM": ARPNUM, "hgs_lon": lon, "spacecraft_r": r, "footprint_lon": footprint_lon}
        for col in SHARP_columns:
            result_dict[col] = row[col]
        results.append(result_dict)

final_df = pd.DataFrame(results).set_index("time").sort_values(by=["ARPNUM", "time"])

final_df.to_csv(output_csv)