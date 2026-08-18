import glob
import cdflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

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

all_times = np.concatenate(all_times)
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
earth_bc = get_body_barycentric('earth', times_ap)
sun_bc   = get_body_barycentric('sun',   times_ap)
earth_helio = earth_bc - sun_bc
earth_lon = np.degrees(np.arctan2(-earth_helio.x.value, earth_helio.y.value)) % 360.0
all_hgs_lon = (all_hci_lon - earth_lon) % 360.0
all_hgs_lon = np.where(all_hgs_lon > 180, all_hgs_lon - 360, all_hgs_lon)

print(f"Total time steps: {len(all_times)}, flux shape: {all_flux.shape}")

# Compute linlin: energy-width-weighted average over 0.1033–1.0358 MeV (indices 5–20)
energy_width = energy_delta_plus - energy_delta_minus          # shape: (n_energies,)
energy_mask = (energy >= 0.1032) & (energy <= 1.0359)         # shape: (n_energies,)
linlin = (np.nansum(all_flux * energy_width * energy_mask, axis=1) /
          np.nansum(energy_width * energy_mask))               # shape: (n_times,)

df_linlin = pd.DataFrame({
    "time": all_times,
    "linlin": linlin, # Before averaging
    "hgs_lon": all_hgs_lon,
    "spacecraft_r": all_hci_r
})

output_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/Jlinlin_SolO.csv"
df_linlin.to_csv(output_csv, index=False)
print(f"Saved linlin data to {output_csv}")