import glob
import cdflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import frames
from tqdm import tqdm

LEVEL3_DIR = "/scratch/gpfs/sk6617/SolO/Level3/"

cdf_files = sorted(glob.glob(LEVEL3_DIR + "solo_l3_epd-ept-1hour_*.cdf"))
print(f"Found {len(cdf_files)} CDF files.")

all_times = []
all_flux = []
all_hci_r = []
all_hci_lat = []
all_hci_lon = []

for f in cdf_files:
    cdf = cdflib.CDF(f)
    epoch = cdf.varget("EPOCH")
    times = cdflib.cdfepoch.to_datetime(epoch)
    flux_S = cdf.varget("Ion_Flux_S")
    flux_A = cdf.varget("Ion_Flux_A")
    flux_N = cdf.varget("Ion_Flux_N")
    flux_D = cdf.varget("Ion_Flux_D")
    flux = flux_S + flux_A + flux_N + flux_D
    all_times.append(times)
    all_flux.append(flux)
    all_hci_r.append(cdf.varget("HCI_R"))
    all_hci_lat.append(cdf.varget("HCI_Lat"))
    all_hci_lon.append(cdf.varget("HCI_Lon"))

all_times = pd.to_datetime(np.concatenate(all_times))
all_flux = np.concatenate(all_flux, axis=0).astype(float)
all_flux[all_flux < 0] = np.nan
all_hci_r   = np.concatenate(all_hci_r).astype(float)
all_hci_lat = np.concatenate(all_hci_lat).astype(float)
all_hci_lon = np.concatenate(all_hci_lon).astype(float)

# Get energy axis from the last file
energy = cdf.varget("Ion_Energy")

print("CDF variables:")
for var in cdf.cdf_info().zVariables:
    print(f"  {var}")

energy_indices = range(5, 21)

energy_delta_plus = cdf.varget("Ion_Energy_Delta_Plus")
energy_delta_minus = cdf.varget("Ion_Energy_Delta_Minus")

print("Ion_Energy / Ion_Energy_Delta_Minus / Ion_Energy_Delta_Plus:")
for i, (e, dm, dp) in enumerate(zip(energy, energy_delta_minus, energy_delta_plus)):
    print(f"  [{i}] {e:.4f} MeV  -{dm:.4f}  +{dp:.4f}")

# Convert HCI longitude to Stonyhurst (HGS): subtract Earth's HCI longitude
times_ap = Time(all_times)

hci_coord = SkyCoord(
    lon=all_hci_lon * u.deg,
    lat=all_hci_lat * u.deg,
    distance=all_hci_r * u.au,
    frame=frames.HeliocentricInertial(obstime=times_ap),
    representation_type="spherical",
)

hgs_coord = hci_coord.transform_to(
    frames.HeliographicStonyhurst(obstime=times_ap)
)

all_hgs_lon = hgs_coord.lon.to_value(u.deg)
all_hgs_lat = hgs_coord.lat.to_value(u.deg)
all_hgs_r = hgs_coord.radius.to_value(u.au)

print(f"Total time steps: {len(all_times)}, flux shape: {all_flux.shape}")

# Compute linlin: energy-width-weighted average over 0.1033–1.0358 MeV (indices 5–20)
energy_width = energy_delta_plus - energy_delta_minus          # shape: (n_energies,)
energy_mask = (energy >= 0.1032) & (energy <= 1.0359)         # shape: (n_energies,)
linlin = (np.nansum(all_flux * energy_width * energy_mask, axis=1) /
          np.nansum(energy_width * energy_mask))               # shape: (n_times,)

fig, (ax_top, ax_bot, ax_third) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

psp_csv = "../../data_collection/all_psp_data.csv"

df_psp = pd.read_csv(psp_csv, parse_dates=["time"])

psp_times = pd.to_datetime(df_psp["time"], format="mixed", errors="coerce")
psp_flux = df_psp["psp_epilo_jlinlin_flux"].to_numpy(dtype=float) * 1000

r = df_psp["psp_ephem_features_HCI_R"].to_numpy(dtype=float)
lon_deg = df_psp["psp_ephem_features_HGS_Lon"].to_numpy(dtype=float)

valid_psp_mask = (
    ~pd.isna(psp_times) &
    np.isfinite(psp_flux) &
    np.isfinite(r) &
    np.isfinite(lon_deg)
)

psp_times = psp_times[valid_psp_mask]
psp_flux = psp_flux[valid_psp_mask]
r = r[valid_psp_mask]
lon_deg = lon_deg[valid_psp_mask]

lon_rad = np.deg2rad(lon_deg)

x_psp = -r * np.sin(lon_rad)
y_psp =  r * np.cos(lon_rad)

# Filter out Nan and zero values from psp data for plotting
# valid_psp_mask = ~np.isnan(psp_flux) & (psp_flux != 0)
valid_psp_mask = ~np.isnan(psp_flux)
psp_times = psp_times[valid_psp_mask]
psp_flux = psp_flux[valid_psp_mask]

# Top panel: z-score normalized linlin flux

# z-score normalize linlin and psp_flux
linlin_nzeromask = ~np.isnan(linlin) & (linlin != 0)
linlin_zeromask = linlin == 0
linlin_nanmask = ~np.isnan(linlin)
linlin_mean = np.mean(np.log(linlin[linlin_nzeromask]))
linlin_std = np.std(np.log(linlin[linlin_nzeromask]))
linlin_norm = (np.log(linlin + np.exp(-5)) - linlin_mean) / linlin_std

psp_flux_nzeromask = ~np.isnan(psp_flux) & (psp_flux != 0)
psp_flux_zeromask = psp_flux == 0
psp_flux_nanmask = ~np.isnan(psp_flux)
psp_flux_mean = np.mean(np.log(psp_flux[psp_flux_nzeromask]))
psp_flux_std = np.std(np.log(psp_flux[psp_flux_nzeromask]))
psp_flux_norm = (np.log(psp_flux + np.exp(-5)) - psp_flux_mean) / psp_flux_std

print(linlin[:10], psp_flux[:10])
print(linlin_norm[:10], psp_flux_norm[:10])

ax_top.plot(all_times[linlin_nanmask], linlin_norm[linlin_nanmask], lw=0.8, color="tab:blue", label="SolO EPT Linlin")
ax_top.plot(psp_times[psp_flux_nanmask], psp_flux_norm[psp_flux_nanmask], lw=0.8, color="tab:orange", label="PSP EPI-Lo Linlin")
ax_top.legend(loc="upper left", fontsize=8)
ax_top.set_ylabel("log Linlin Flux [z-score normalized]")
ax_top.set_title("SolO and PSP — Linlin (S+A+N+D)")
ax_top.set_ylim(-5, 5)
ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# Middle panel: linlin flux

ax_bot.plot(all_times[linlin_nanmask], np.log(linlin + np.exp(-5))[linlin_nanmask], lw=0.8, color="tab:blue", label="SolO EPT Linlin")
ax_bot.plot(psp_times[psp_flux_nanmask], np.log(psp_flux + np.exp(-5))[psp_flux_nanmask], lw=0.8, color="tab:orange", label="PSP EPI-Lo Linlin")
ax_bot.legend(loc="upper left", fontsize=8)
ax_bot.set_ylabel("log Linlin Flux")
ax_bot.set_title("SolO and PSP — Linlin (S+A+N+D)")
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

fig.autofmt_xdate()
plt.tight_layout()

# Bottom panel: Eventplot

SHARP_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP.csv"
SolO_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/processed_SolO.csv"
output_img = "active_region_durations_packed.png"

df = pd.read_csv(SHARP_csv, parse_dates=["T_OBS"])
df = df[df["T_OBS"] > pd.Timestamp("2020-06-01")]
df = df.dropna(subset=["T_OBS", "ARPNUM"])
df = df.sort_values(["ARPNUM", "T_OBS"])

max_gap = pd.Timedelta(hours=4)

intervals = []

for ar, g in tqdm(df.groupby("ARPNUM"), desc="Processing active regions"):
    times = g["T_OBS"].sort_values().drop_duplicates()

    if len(times) == 0:
        continue

    start = times.iloc[0]
    prev = times.iloc[0]

    for t in times.iloc[1:]:
        if t - prev > max_gap:
            intervals.append({"ARPNUM": ar, "start": start, "end": prev})
            start = t
        prev = t

    intervals.append({"ARPNUM": ar, "start": start, "end": prev})

interval_df = pd.DataFrame(intervals)
interval_df["duration"] = interval_df["end"] - interval_df["start"]
interval_df = interval_df[interval_df["duration"] > pd.Timedelta(0)]
interval_df = interval_df.sort_values("start").reset_index(drop=True)

# Greedy lane packing
lane_end_times = []
lanes = []

for _, row in tqdm(interval_df.iterrows(), desc="Packing intervals"):
    placed = False

    for lane_idx, lane_end in enumerate(lane_end_times):
        if row["start"] > lane_end:
            lanes.append(lane_idx)
            lane_end_times[lane_idx] = row["end"]
            placed = True
            break

    if not placed:
        lanes.append(len(lane_end_times))
        lane_end_times.append(row["end"])

interval_df["lane"] = lanes

for _, row in tqdm(interval_df.iterrows(), desc="Plotting intervals"):
    start_num = plt.matplotlib.dates.date2num(row["start"])
    width_days = row["duration"].total_seconds() / 86400
    y = row["lane"]

    ax_third.broken_barh(
        [(start_num, width_days)],
        (y - 0.4, 0.8),
    )

ax_third.set_xlabel("Time")
ax_third.set_ylabel("Packed active-region row")
ax_third.set_title("Packed SHARP Active Region Patch Durations")

ax_third.grid(True, axis="x")

df_solO = pd.read_csv(SolO_csv, parse_dates=["time"])
df_solO = df_solO.sort_values("time")

mask = df_solO["footprint_valid"].astype(bool)

J_plot = df_solO["linlin"].where(mask, np.nan)

out_path = "Ion_Flux_SAND_linlin_all.png"
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")

# --- Heliocentric orbit plot ---
lon_rad = np.deg2rad(all_hgs_lon)
x_solo = -all_hci_r * np.sin(lon_rad)
y_solo =  all_hci_r * np.cos(lon_rad)

# Parker spiral
v_sw = 400          # km/s
omega_sun = 2 * np.pi / (27.27 * 86400)  # rad/s
theta_spiral = np.linspace(0, 4 * np.pi, 1000)
r_spiral = (v_sw / omega_sun) * theta_spiral / 1.496e8  # AU

# Map times to numeric values for colormap
times_num = mdates.date2num(all_times.astype("M8[ms]").astype("O"))
norm = mcolors.Normalize(vmin=times_num.min(), vmax=times_num.max())
cmap = cm.plasma

fig2, ax2 = plt.subplots(figsize=(9, 8))
for angle in [-80, -60, -30, 0, 30, 60, 80]:
    phi0 = np.deg2rad(angle)
    x_sp =  r_spiral * np.sin(theta_spiral + phi0)
    y_sp =  r_spiral * np.cos(theta_spiral + phi0)
    ax2.plot(x_sp, y_sp, color="gray", linewidth=1.5, alpha=0.5)

times_num_psp = mdates.date2num(psp_times.astype("M8[ms]").astype("O"))
norm_psp = mcolors.Normalize(vmin=times_num_psp.min(), vmax=times_num_psp.max())
cmap_psp = cm.plasma

sc = ax2.scatter(x_solo, y_solo, marker=".", c=times_num, cmap=cmap, norm=norm,
                 alpha=0.7, s=4, label="SolO")
sc_psp = ax2.scatter(x_psp, y_psp, marker="x", c=times_num_psp, cmap=cmap_psp, norm=norm_psp,
                     alpha=0.7, s=20, label="PSP")
ax2.scatter(0, 0,    s=350, color="yellow", label="Sun",   zorder=5)
ax2.scatter(0, 1,    s=100, color="blue",   label="Earth", zorder=5)
ax2.scatter(0, 0.99, s=100, color="green",  marker="*", label="L1", zorder=6)

cbar = fig2.colorbar(sc, ax=ax2, pad=0.02)
cbar.set_label("Time")
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

ax2.set_xlabel("X (AU)")
ax2.set_ylabel("Y (AU)")
ax2.set_xlim(-0.6, 1)
ax2.set_ylim(-0.2, 1.1)
ax2.set_aspect("equal", adjustable="box")
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

#out_path2 = "/scratch/gpfs/bb4178/initial_plots/flux_t/SolO_orbit_heliocentric.png"
out_path2 = "SolO_orbit_heliocentric.png"
plt.savefig(out_path2, dpi=150)
print(f"Plot saved to {out_path2}")