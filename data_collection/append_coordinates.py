import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# Load SolO data
SolO_csv = "processed_SolO.csv"
df_solO = pd.read_csv(SolO_csv, parse_dates=["time","offset_time"])

# Load
raw_csv = "Jlinlin_SolO.csv"
df_raw = pd.read_csv(raw_csv, parse_dates=["time"])

df_solO["hgs_lon"] = df_raw["hgs_lon"]
df_solO["spacecraft_r"] = df_raw["spacecraft_r"]

df_solO.to_csv("processed_SolO.csv")