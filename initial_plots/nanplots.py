# This script parses the Jlinlin values from the h5 files
# and plots where NaN values occur in each file.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def load_epi_flux_data(h5_path, key="/data", is_epilo=False, fields=None):
    """
    Load EPI flux data from HDF5, optionally selecting specific fields.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    df = pd.read_hdf(h5_path, key=key)

    # For EPI-Lo files, time may be stored as the index
    if is_epilo:
        df = df.reset_index()

        if "index" in df.columns and "time" not in df.columns:
            df = df.rename(columns={"index": "time"})

    if "time" not in df.columns:
        raise ValueError(f"'time' column not found in {h5_path}. Columns: {df.columns.tolist()}")

    if "flux" not in df.columns:
        raise ValueError(f"'flux' column not found in {h5_path}. Columns: {df.columns.tolist()}")

    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

    if fields is not None:
        missing = [f for f in fields if f not in df.columns]
        if missing:
            raise ValueError(f"Fields not found in dataset: {missing}")
        df = df[fields]

    df = df.sort_values("time").reset_index(drop=True)

    return df


def inspect_flux(df, name):
    flux = pd.to_numeric(df["flux"], errors="coerce")

    print(f"\n{name}")
    print("rows:", len(df))
    print("NaN flux:", flux.isna().sum())
    print("NaN share:", flux.isna().mean())
    print("zero flux:", (flux == 0).sum())
    print("negative flux:", (flux < 0).sum())
    print("positive flux:", (flux > 0).sum())
    print("min:", flux.min())
    print("median:", flux.median())
    print("max:", flux.max())
    print(df.head(20))


def plot_nan_locations(dfs, names, output_path="epilo_nan_locations.png"):
    """
    Plot where NaN flux values occur for each dataframe.

    Each subplot has:
        y = 1 for NaN flux
        y = 0 for finite flux
    """
    n = len(dfs)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(14, 3.5 * n),
        sharex=True,
    )

    if n == 1:
        axes = [axes]

    for ax, df, name in zip(axes, dfs, names):
        flux = pd.to_numeric(df["flux"], errors="coerce")
        nan_mask = flux.isna()

        ax.plot(
            df["time"],
            nan_mask.astype(int),
            marker=".",
            linestyle="None",
            alpha=0.7,
            markersize=2,
            label="NaN location",
        )

        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["finite", "NaN"])
        ax.set_ylabel("Flux status")
        ax.set_title(f"{name}: NaN locations in flux")
        ax.grid(True)
        ax.legend()

        print(f"{name}: {nan_mask.sum()} NaNs out of {len(df)} rows ({nan_mask.mean():.4%})")

    axes[-1].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved NaN-location plot to: {output_path}")

    plt.show()


def plot_overlayed_jlinlin_lines(
    dfs,
    names,
    output_path="epilo_jlinlin_overlayed_lines.png",
    eps=1e-5,
    raw_ylim_percentile=99.5,
):
    """
    Overlay new and old Jlinlin/flux values on the same axes.

    Produces two panels:
      1. Raw finite flux line plot
      2. log(flux + eps) line plot for finite positive flux

    Notes:
      - Raw plot uses finite values, including zeros.
      - Log plot uses finite positive values only.
      - Optional raw y-limit clips extreme outliers visually but does not alter data.
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16, 9),
        sharex=True,
    )

    all_raw_finite = []

    for df, name in zip(dfs, names):
        df_plot = df.copy()
        df_plot["time"] = pd.to_datetime(df_plot["time"], format="mixed", errors="coerce")
        df_plot["flux"] = pd.to_numeric(df_plot["flux"], errors="coerce")
        df_plot = df_plot.dropna(subset=["time"]).sort_values("time")

        flux = df_plot["flux"].to_numpy(dtype=float)
        time = df_plot["time"]

        finite_mask = np.isfinite(flux)
        positive_mask = np.isfinite(flux) & (flux > 0)

        all_raw_finite.append(flux[finite_mask])

        # Raw finite flux line plot
        axes[0].plot(
            time[finite_mask],
            flux[finite_mask],
            linewidth=0.8,
            alpha=0.75,
            label=f"{name} finite flux",
        )

        # Log positive flux line plot
        axes[1].plot(
            time[positive_mask],
            np.log(flux[positive_mask] + eps),
            linewidth=0.8,
            alpha=0.75,
            label=f"{name} log positive flux",
        )

        print(f"\n{name} line plot inputs")
        print("rows:", len(df_plot))
        print("finite:", finite_mask.sum())
        print("positive:", positive_mask.sum())
        print("zero:", np.sum(np.isfinite(flux) & (flux == 0)))
        print("negative:", np.sum(np.isfinite(flux) & (flux < 0)))
        print("nan:", np.isnan(flux).sum())

    axes[0].set_ylabel("Jlinlin flux")
    axes[0].set_title("New vs Old EPI-Lo Jlinlin flux — overlayed raw line plot")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylabel(f"log(flux + {eps})")
    axes[1].set_title("New vs Old EPI-Lo Jlinlin flux — overlayed log line plot")
    axes[1].grid(True)
    axes[1].legend()

    axes[1].set_xlabel("Time")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    print(f"\nSaved overlayed Jlinlin line plot to: {output_path}")
    plt.show()


# -----------------------------
# File paths
# -----------------------------
epilo_file_new = "/scratch/gpfs/sk6617/ISOIS_data_Tate/spp-isois.sr.unh.edu/data_public/EPILo/epilo_Jlinlin_flux_full_mission_2026_04.h5"
epilo_file_old = "/scratch/gpfs/sk6617/ISOIS_data_Tate/spp-isois.sr.unh.edu/data_public/EPILo/epilo_Jlinlin_flux_full_mission.h5"

print("New file:")
with pd.HDFStore(epilo_file_new, mode="r") as store:
    print(store.keys())

print("Old file:")
with pd.HDFStore(epilo_file_old, mode="r") as store:
    print(store.keys())

fields = ["time", "flux"]

# -----------------------------
# Load datasets
# -----------------------------
df_new = load_epi_flux_data(
    epilo_file_new,
    key="/data",
    is_epilo=True,
    fields=fields,
)

df_old = load_epi_flux_data(
    epilo_file_old,
    key="/epilo_data",
    is_epilo=True,
    fields=fields,
)

def plot_flux_histograms(dfs, names, output_path="epilo_flux_histograms.png", eps=1e-5):
    """
    Overlay histograms comparing the distribution of finite Jlinlin/flux values.

    Produces two panels:
      1. Raw finite flux distribution
      2. Log finite positive flux distribution

    Notes:
      - Raw panel uses only finite values.
      - Log panel uses finite positive values only.
      - density=True makes the histograms comparable even if one file has many more rows.
    """
    finite_fluxes = []
    positive_fluxes = []

    for df, name in zip(dfs, names):
        df_filtered = df[df["time"] <= pd.Timestamp("2025-01-01")]
        flux = pd.to_numeric(df_filtered["flux"], errors="coerce").to_numpy(dtype=float)

        finite = flux[np.isfinite(flux)]
        positive = flux[np.isfinite(flux) & (flux > 0)]

        finite_fluxes.append(finite)
        positive_fluxes.append(positive)

        print(f"\n{name} histogram inputs")
        print("finite count:", len(finite))
        print("positive count:", len(positive))
        print("zero count:", np.sum(finite == 0))
        print("negative count:", np.sum(finite < 0))

        if len(positive) > 0:
            log_positive = np.log(positive + eps)
            print("raw positive min/median/mean/max:",
                  np.min(positive),
                  np.median(positive),
                  np.mean(positive),
                  np.max(positive))
            print("log positive min/p1/p10/median/mean/p90/p99/max:",
                  np.min(log_positive),
                  np.percentile(log_positive, 1),
                  np.percentile(log_positive, 10),
                  np.median(log_positive),
                  np.mean(log_positive),
                  np.percentile(log_positive, 90),
                  np.percentile(log_positive, 99),
                  np.max(log_positive))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 9),
    )

    # -----------------------------
    # Raw finite flux histogram
    # -----------------------------
    all_finite = np.concatenate([x for x in finite_fluxes if len(x) > 0])

    if len(all_finite) > 0:
        raw_min = np.nanmin(all_finite)
        raw_max = np.nanmax(all_finite)

        # Avoid pathological binning if there are huge outliers
        raw_p99 = np.nanpercentile(all_finite, 99.5)

        raw_bins = np.linspace(raw_min, raw_p99, 150)

        for finite, name in zip(finite_fluxes, names):
            finite_plot = finite[finite <= raw_p99]

            axes[0].hist(
                finite_plot,
                bins=raw_bins,
                alpha=0.5,
                density=True,
                label=f"{name} finite flux",
            )

        axes[0].set_xlabel("Finite flux")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Distribution of finite Jlinlin flux values")
        axes[0].grid(True)
        axes[0].legend()

    # -----------------------------
    # Log positive flux histogram
    # -----------------------------
    log_fluxes = []

    for positive in positive_fluxes:
        log_fluxes.append(np.log(positive + eps))

    all_log = np.concatenate([x for x in log_fluxes if len(x) > 0])

    if len(all_log) > 0:
        log_min = np.nanpercentile(all_log, 0.5)
        log_max = np.nanpercentile(all_log, 99.5)
        log_bins = np.linspace(log_min, log_max, 150)

        for log_flux, name in zip(log_fluxes, names):
            log_plot = log_flux[
                np.isfinite(log_flux) &
                (log_flux >= log_min) &
                (log_flux <= log_max)
            ]

            axes[1].hist(
                log_plot,
                bins=log_bins,
                alpha=0.5,
                density=True,
                label=f"{name} log positive flux",
            )

        axes[1].set_xlabel(f"log(flux + {eps})")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Distribution of log positive Jlinlin flux values")
        axes[1].grid(True)
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved flux histogram comparison to: {output_path}")

    plt.show()

# -----------------------------
# Inspect
# -----------------------------
inspect_flux(df_new, "New EPI-Lo")
inspect_flux(df_old, "Old EPI-Lo")

print("\nNew EPI-Lo sample:")
print(df_new.head())

print("\nOld EPI-Lo sample:")
print(df_old.head())

# -----------------------------
# Plot NaN locations
# -----------------------------
plot_nan_locations(
    dfs=[df_new, df_old],
    names=["New EPI-Lo", "Old EPI-Lo"],
    output_path="epilo_nan_locations.png",
)

plot_overlayed_jlinlin_lines(
    dfs=[df_new, df_old],
    names=["New EPI-Lo", "Old EPI-Lo"],
    output_path="epilo_jlinlin_overlayed_lines.png",
    eps=1e-5,
)

plot_flux_histograms(
    dfs=[df_new, df_old],
    names=["New EPI-Lo", "Old EPI-Lo"],
    output_path="epilo_flux_histograms.png",
    eps=1e-5,
)