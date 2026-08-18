import pandas as pd
import glob
import cdflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.lines import Line2D
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import astropy.units as u
from astropy.time import Time
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, sys
from tqdm import tqdm

# Get SolO and PSP positions
psp_dir = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/processed_psp.csv"
solo_dir = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/processed_SolO.csv"

df_psp = pd.read_csv(psp_dir, parse_dates=["time"])
df_solo = pd.read_csv(solo_dir, parse_dates=["time"])

#print("PSP CSV headers:", df_psp.columns.tolist())
#print("SolO CSV headers:", df_solo.columns.tolist())
#sys.exit()

all_times_psp = pd.to_datetime(df_psp["time"], errors="coerce")
all_times_solo = pd.to_datetime(df_solo["time"], errors="coerce")
all_times_psp_num = mdates.date2num(all_times_psp)
all_times_solo_num = mdates.date2num(all_times_solo)

# Get X and Y positions in AU
r_psp = df_psp["psp_r_au"].to_numpy()
lon_psp = df_psp["psp_ephem_features_HGS_Lon"].to_numpy()
r_solo = df_solo["solo_r_au"].to_numpy()
lon_solo = df_solo["hgs_lon"].to_numpy()

lon_psp_rad = np.deg2rad(lon_psp)
lon_solo_rad = np.deg2rad(lon_solo)
x_psp = -r_psp * np.sin(lon_psp_rad)
y_psp =  r_psp * np.cos(lon_psp_rad)
x_solo = -r_solo * np.sin(lon_solo_rad)
y_solo =  r_solo * np.cos(lon_solo_rad)

psp_mask = df_psp["footprint_valid"].to_numpy() == 1
solo_mask = df_solo["footprint_valid"].to_numpy() == 1

# Intensities for right panel
solo_intensity = df_solo["linlin"].to_numpy()
psp_intensity = df_psp["psp_epilo_jlinlin_flux"].to_numpy()
#print(f"SolO intensity min/max: {np.nanmin(solo_intensity)} / {np.nanmax(solo_intensity)}")
#print(f"PSP intensity min/max: {np.nanmin(psp_intensity)} / {np.nanmax(psp_intensity)}")

def apply_zscore_normalization(arr, mean, std):
    arr = arr.astype(float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr)
    out[valid] = (arr[valid] - mean) / std
    return out

def log_positive(arr):
    arr = arr.astype(float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    out[valid] = np.log(arr[valid])
    return out

solo_intensity_log = log_positive(solo_intensity)
psp_intensity_log = log_positive(psp_intensity)

# Mean and std stats from training set (for z-score normalization)
solo_mean = 3.7579716919768713
solo_std = 3.4871714751297733
psp_mean = -3.33214516064573
psp_std = 1.8158943207532017

solo_intensity_norm = apply_zscore_normalization(solo_intensity_log, solo_mean, solo_std)
psp_intensity_norm = apply_zscore_normalization(psp_intensity_log, psp_mean, psp_std)

#print(f"SolO intensity norm min/max: {np.nanmin(solo_intensity_norm)} / {np.nanmax(solo_intensity_norm)}")
#print(f"PSP intensity norm min/max: {np.nanmin(psp_intensity_norm)} / {np.nanmax(psp_intensity_norm)}")
#sys.exit()
solo_int_mask = solo_mask & np.isfinite(solo_intensity_norm)
psp_int_mask = psp_mask & np.isfinite(psp_intensity_norm)
solo_int_time_mask = solo_int_mask & np.isfinite(all_times_solo_num)
psp_int_time_mask = psp_int_mask & np.isfinite(all_times_psp_num)
int_norm = mcolors.Normalize(vmin=-5.0, vmax=5.0)
int_cmap = "plasma"
axis_label_fs = 16
tick_label_fs = 13
top_label_fs = 14

# Parker spiral
v_sw = 400          # km/s
omega_sun = 2 * np.pi / (27.27 * 86400)  # rad/s
theta_spiral = np.linspace(0, 4 * np.pi, 1000)
r_spiral = (v_sw / omega_sun) * theta_spiral / 1.496e8  # AU

time_num_psp = mdates.date2num(df_psp["time"])
time_num_solo = mdates.date2num(df_solo["time"])
time_min = min(time_num_psp.min(), time_num_solo.min())
time_max = max(time_num_psp.max(), time_num_solo.max())
time_norm = mcolors.Normalize(vmin=time_min, vmax=time_max)
time_cmap = "viridis"

fig2, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)
left_ax, middle_ax, right_ax = axes

for ax in (left_ax, middle_ax, right_ax):
    for angle in [-80, -60, -30, 0, 30, 60, 80]:
        phi0 = np.deg2rad(angle)
        x_sp = r_spiral * np.sin(theta_spiral + phi0)
        y_sp = r_spiral * np.cos(theta_spiral + phi0)
        ax.plot(x_sp, y_sp, color="gray", linewidth=1.5, alpha=0.5)

    ax.scatter(0, 0, s=350, color="yellow", zorder=5)
    ax.scatter(0, 1, s=100, color="blue", zorder=5)
    ax.scatter(0, 0.99, s=100, color="green", marker=".", zorder=6)
    ax.set_xlabel("X (AU)", fontsize=axis_label_fs)
    ax.set_ylabel("Y (AU)", fontsize=axis_label_fs)
    ax.set_xlim(-0.6, 1)
    ax.set_ylim(-0.2, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=tick_label_fs)
    ax.grid(True, linestyle="--", alpha=0.5)

middle_ax.set_ylabel("")
right_ax.set_ylabel("")

# Middle panel: color both spacecraft points by time
middle_ax.scatter(
    x_solo[solo_mask],
    y_solo[solo_mask],
    c=time_num_solo[solo_mask],
    cmap=time_cmap,
    norm=time_norm,
    marker=".",
)
middle_ax.scatter(
    x_psp[psp_mask],
    y_psp[psp_mask],
    c=time_num_psp[psp_mask],
    cmap=time_cmap,
    norm=time_norm,
    marker=".",
)

# Outer panels: mission colors
left_ax.scatter(x_solo[solo_mask], y_solo[solo_mask], marker=".", color="red", label="Solar Orbiter")
left_ax.scatter(x_psp[psp_mask], y_psp[psp_mask], marker=".", color="blue", label="Parker Solar Probe")

# Right panel: color both spacecraft points by intensity
right_ax.scatter(
    x_solo[solo_int_mask],
    y_solo[solo_int_mask],
    c=solo_intensity_norm[solo_int_mask],
    cmap=int_cmap,
    norm=int_norm,
    marker=".",
)
right_ax.scatter(
    x_psp[psp_int_mask],
    y_psp[psp_int_mask],
    c=psp_intensity_norm[psp_int_mask],
    cmap=int_cmap,
    norm=int_norm,
    marker=".",
)

# Top legend area for the left panel (same size as colorbar areas)
divider_left = make_axes_locatable(left_ax)
cax_left = divider_left.append_axes("top", size="6%", pad=0.35)
cax_left.axis("off")
legend_handles = [
    Line2D([0], [0], color="red", lw=2),
    Line2D([0], [0], color="blue", lw=2),
]
legend_labels = ["Solar Orbiter", "Parker Solar Probe"]
cax_left.legend(
    legend_handles,
    legend_labels,
    frameon=False,
    fontsize=16,
    loc="center",
    ncol=2,
    handlelength=2.0,
)

# Top colorbar for the middle panel
sm = cm.ScalarMappable(norm=time_norm, cmap=time_cmap)
sm.set_array([])
divider = make_axes_locatable(middle_ax)
cax = divider.append_axes("top", size="6%", pad=0.35)
cbar = fig2.colorbar(sm, cax=cax, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_label("Time", fontsize=top_label_fs)
cbar.ax.xaxis.set_major_locator(mdates.YearLocator())
cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
cbar.ax.tick_params(labelsize=tick_label_fs)

# Top colorbar for the right panel
sm_int = cm.ScalarMappable(norm=int_norm, cmap=int_cmap)
sm_int.set_array([])
divider_right = make_axes_locatable(right_ax)
cax_right = divider_right.append_axes("top", size="6%", pad=0.35)
cbar_right = fig2.colorbar(sm_int, cax=cax_right, orientation="horizontal")
cbar_right.ax.xaxis.set_ticks_position("top")
cbar_right.ax.xaxis.set_label_position("top")
cbar_right.set_label("Normalized log(Intensity)", fontsize=top_label_fs)
cbar_right.ax.tick_params(labelsize=tick_label_fs)

fig2.subplots_adjust(left=0.05, right=0.99, bottom=0.09, top=0.91, wspace=0.06)

out_path = "SolO_PSP_orbit_heliocentric.png"
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")