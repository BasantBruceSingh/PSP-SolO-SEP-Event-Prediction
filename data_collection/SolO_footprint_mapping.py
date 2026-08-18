import pandas as pd
import numpy as np
from tqdm import tqdm
from solarmach import SolarMACH

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import frames

SolO_csv = "Jlinlin_SolO_with_VSW.csv"
output_csv = "processed_SolO.csv"

AU_TO_KM = 1.496e8

df_solO = pd.read_csv(SolO_csv, parse_dates=["time"])
df_solO = df_solO.dropna(subset=["time", "linlin"])
df_solO = df_solO.drop_duplicates()
df_solO = df_solO.sort_values("time").reset_index(drop=True)


def get_offset(r_km, speed_km_s=400.0):
    """
    Estimate travel-time offset using Parker spiral path length.

    Note: speed_km_s=400 means 400 km/s.
    If you intentionally want 4000 km/s, pass speed_km_s=4000 explicitly.
    """
    PARKER_A = 4.9e-17
    PARKER_B = 7e-9
    PARKER_C = 7.14e7

    S_R = (
        0.5 * np.sqrt(PARKER_A * r_km**2 + 1) * r_km
        + PARKER_C * np.arcsinh(PARKER_B * r_km)
    )

    return S_R / speed_km_s


def carrington_to_stonyhurst_lon(carr_lon_deg, time, lat_deg=0.0):
    """
    Convert Carrington longitude to Heliographic Stonyhurst longitude.
    """
    obstime = Time(pd.Timestamp(time).to_pydatetime())

    hgc = SkyCoord(
        lon=carr_lon_deg * u.deg,
        lat=lat_deg * u.deg,
        radius=1.0 * u.R_sun,
        frame=frames.HeliographicCarrington(
            obstime=obstime,
            observer="earth",
        ),
    )

    hgs = hgc.transform_to(
        frames.HeliographicStonyhurst(obstime=obstime)
    )

    return hgs.lon.to_value(u.deg)


# Pre-create output columns
df_solO["offset_time"] = pd.NaT
df_solO["solo_r_au"] = np.nan
df_solO["solo_footpoint_carrington_lon"] = np.nan
df_solO["solo_footpoint_hgs_lon"] = np.nan
df_solO["footprint_valid"] = False


for i, row in tqdm(df_solO.iterrows(), total=len(df_solO), desc="Computing SolO footprints"):
    time = row["time"]

    vsw = row.get("VSW", np.nan)
    if pd.isna(vsw):
        vsw = 400.0

    try:
        sm = SolarMACH(
            body_list=["Solar Orbiter"],
            date=time,
            vsw_list=[float(vsw)],
        )

        coord_table = sm.coord_table
        solo_row = coord_table.iloc[0]

        r_au = solo_row["Heliocentric distance (AU)"]
        r_km = r_au * AU_TO_KM

        offset_seconds = get_offset(r_km, speed_km_s=float(vsw))
        offset_time = time - pd.Timedelta(seconds=offset_seconds)

        footpoint_carr_lon = solo_row["Magnetic footpoint longitude (Carrington)"]

        footpoint_hgs_lon = carrington_to_stonyhurst_lon(
            carr_lon_deg=footpoint_carr_lon,
            time=time,
            lat_deg=0.0,
        )

        footprint_valid = abs(footpoint_hgs_lon) <= 80.0

        df_solO.at[i, "offset_time"] = offset_time
        df_solO.at[i, "solo_r_au"] = r_au
        df_solO.at[i, "solo_footpoint_carrington_lon"] = footpoint_carr_lon
        df_solO.at[i, "solo_footpoint_hgs_lon"] = footpoint_hgs_lon
        df_solO.at[i, "footprint_valid"] = footprint_valid

        print(
            f"Time: {time}, "
            f"Carrington FP: {footpoint_carr_lon:.2f}, "
            f"HGS FP: {footpoint_hgs_lon:.2f}, "
            f"valid: {footprint_valid}"
        )

    except Exception as e:
        print(f"Failed at time {time}: {e}")
        continue


df_solO.to_csv(output_csv, index=False)
print("Footprint mapping and offset time calculation completed and saved to:", output_csv)