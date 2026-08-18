import pandas as pd

# Load and preprocess SHtime data
SHARP_csv_file = "SHARP.csv"

df_SHARP = pd.read_csv(SHARP_csv_file, parse_dates=["T_OBS"])
df_SHARP = df_SHARP.sort_values(by=["T_OBS", "ARPNUM"])

prev_time = None
largest_time = -1
largest_val = -1
count = 1

for i, row in df_SHARP.iterrows():
    time = pd.to_datetime(row["T_OBS"])
    if time != prev_time:
        if count > largest_val:
            largest_val = count
            largest_time = time
        count = 1
    else :
        count += 1
    prev_time = time

if count > largest_val:
    largest_val = count
    largest_time = time

print(f'Largest time: {largest_time}')
print(f'Number of SHARP regions: {largest_val}')

df_SHARP.to_csv("SHARP_sorted.csv")