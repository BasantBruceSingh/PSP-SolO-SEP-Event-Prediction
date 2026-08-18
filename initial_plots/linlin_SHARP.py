from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from tqdm import tqdm
import numpy as np

solo_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP_to_SolO_times.csv"
psp_csv = "/scratch/gpfs/bb4178/SEP_prediction/data_collection/SHARP_to_PSP_times.csv"

feature_cols = [
    "USFLUXL",
    "R_VALUE",
    "MEANGBL_GMM",
    "USFLUXZ",
    "MEANGBZ",
    "CMASKL",
    "LAT_FWT",
    "LON_FWT",
]

EPS = 1e-12


def aggregate_by_time(df, feature_cols, spacecraft_name):
    """
    For each time, average each SHARP feature across ARs,
    and keep the corresponding Jlinlin value for that time.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "Jlinlin"])
    df = df.sort_values(by=["time", "ARPNUM"])

    for col in feature_cols + ["Jlinlin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Jlinlin"] + feature_cols)

    rows = []

    for time, group in tqdm(df.groupby("time"), desc=f"Aggregating {spacecraft_name} rows"):
        row = {
            "spacecraft": spacecraft_name,
            "time": time,
            "Jlinlin": group["Jlinlin"].iloc[0],
        }

        for col in feature_cols:
            row[col] = group[col].mean()

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Load and aggregate data
# -----------------------------
df_solo = pd.read_csv(solo_csv, parse_dates=["time"])
df_psp = pd.read_csv(psp_csv, parse_dates=["time"])

df_solo_agg = aggregate_by_time(df_solo, feature_cols, spacecraft_name="SolO")
df_psp_agg = aggregate_by_time(df_psp, feature_cols, spacecraft_name="PSP")

df = pd.concat([df_solo_agg, df_psp_agg], ignore_index=True)

# -----------------------------
# Per-spacecraft z-score Jlinlin
# -----------------------------
# Use log10(Jlinlin) before z-scoring, since Jlinlin is highly skewed.
# This matches the usual SEP-intensity normalization logic better than raw Jlinlin.
df = df[np.isfinite(df["Jlinlin"]) & (df["Jlinlin"] > 0)].copy()
df["log_Jlinlin"] = np.log10(df["Jlinlin"] + EPS)

df["Jlinlin_z"] = np.nan

for spacecraft in df["spacecraft"].unique():
    mask_sc = df["spacecraft"] == spacecraft

    mu = df.loc[mask_sc, "log_Jlinlin"].mean()
    sigma = df.loc[mask_sc, "log_Jlinlin"].std(ddof=0)

    if sigma == 0 or not np.isfinite(sigma):
        sigma = 1.0

    df.loc[mask_sc, "Jlinlin_z"] = (
        df.loc[mask_sc, "log_Jlinlin"] - mu
    ) / sigma

    print(f"{spacecraft}: log10(Jlinlin) mean = {mu:.4f}, std = {sigma:.4f}")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))

fig.subplots_adjust(
    top=0.96,
    bottom=0.08,
    left=0.07,
    right=0.98,
    hspace=0.18,
    wspace=0.12,
)

for i in range(2):
    for j in range(4):
        index = i * 4 + j
        col = feature_cols[index]

        x = df[col].to_numpy(dtype=float)
        y = df["Jlinlin_z"].to_numpy(dtype=float)

        # Since y is z-scored, it can be negative.
        # Only filter x > 0 for non-coordinate SHARP features.
        if col in {"LAT_FWT", "LON_FWT"}:
            mask = np.isfinite(x) & np.isfinite(y)
        else:
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0)

        x = x[mask]
        y = y[mask]

        if len(x) == 0 or len(y) == 0:
            ax[i][j].set_title(col)
            ax[i][j].set_xlabel(col, fontsize=12)
            ax[i][j].set_ylabel("z-scored log10(Jlinlin)", fontsize=12)
            continue

        # Full x-range for latitude / longitude
        if col == "LAT_FWT":
            x_bins = np.linspace(-90, 90, 50)
            ax[i][j].set_xlim(-90, 90)
        elif col == "LON_FWT":
            x_bins = np.linspace(-180, 180, 50)
            ax[i][j].set_xlim(-180, 180)
        else:
            x_bins = np.linspace(x.min(), x.max(), 50)

        # z-scored y is linear, not log-scaled
        y_min = y.min()
        y_max = y.max()

        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

        y_bins = np.linspace(y_min, y_max, 50)

        hist, x_edges, y_edges, image = ax[i][j].hist2d(
            x,
            y,
            bins=(x_bins, y_bins),
            cmin=1,
        )

        # -----------------------------
        # Weighted fit lines from binned counts
        # -----------------------------
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        Xg, Yg = np.meshgrid(x_centers, y_centers, indexing="ij")

        counts = hist.ravel()
        positive_mask = counts > 0

        if positive_mask.sum() >= 3:
            Xv = Xg.ravel()[positive_mask]
            Yv = Yg.ravel()[positive_mask]
            Wv = counts[positive_mask]

            weights = np.sqrt(Wv)

            # Normalize x before fitting for numerical stability.
            x_mean = Xv.mean()
            x_std = Xv.std()

            if x_std == 0 or not np.isfinite(x_std):
                x_std = 1.0

            Xv_scaled = (Xv - x_mean) / x_std

            x_fit = np.linspace(x_bins[0], x_bins[-1], 200)
            x_fit_scaled = (x_fit - x_mean) / x_std

            # Linear weighted fit: y_z = a*x_scaled + b
            A = np.column_stack([Xv_scaled, np.ones_like(Xv_scaled)])
            coeffs, _, _, _ = np.linalg.lstsq(
                A * weights[:, None],
                Yv * weights,
                rcond=None,
            )

            y_fit = coeffs[0] * x_fit_scaled + coeffs[1]

            line1, = ax[i][j].plot(
                x_fit,
                y_fit,
                color="red",
                linewidth=2.0,
                linestyle="-",
                zorder=20,
            )

            # Quadratic weighted fit: y_z = a*x_scaled^2 + b*x_scaled + c
            A_poly = np.column_stack(
                [Xv_scaled**2, Xv_scaled, np.ones_like(Xv_scaled)]
            )

            coeffs_poly, _, _, _ = np.linalg.lstsq(
                A_poly * weights[:, None],
                Yv * weights,
                rcond=None,
            )

            y_poly_fit = (
                coeffs_poly[0] * x_fit_scaled**2
                + coeffs_poly[1] * x_fit_scaled
                + coeffs_poly[2]
            )

            line2, = ax[i][j].plot(
                x_fit,
                y_poly_fit,
                color="orange",
                linewidth=2.0,
                linestyle="-",
                zorder=21,
            )

            if col == "USFLUXL":
                ax[i][j].legend(
                    [line1, line2],
                    ["Linear Fit", "Quadratic Fit"],
                    loc="lower right",
                    frameon=True,
                    framealpha=0.9,
                    fontsize=8,
                    handlelength=2.0,
                )

        ax[i][j].set_xlabel(col, fontsize=12)
        ax[i][j].set_ylabel("z-scored log10(Jlinlin)", fontsize=12)
        ax[i][j].set_box_aspect(1)

        if col in {"R_VALUE", "CMASKL"}:
            ax[i][j].ticklabel_format(
                axis="x",
                style="sci",
                scilimits=(0, 0),
                useMathText=False,
            )

        if j != 0:
            ax[i][j].set_ylabel("")
            ax[i][j].tick_params(axis="y", which="both", labelleft=False)

        # Inset colorbar
        if col == "R_VALUE":
            cax = inset_axes(
                ax[i][j],
                width="5%",
                height="30%",
                loc="lower right",
                bbox_to_anchor=(-0.02, 0.02, 1, 1),
                bbox_transform=ax[i][j].transAxes,
            )
        else:
            cax = inset_axes(
                ax[i][j],
                width="5%",
                height="30%",
                loc="upper right",
                bbox_to_anchor=(-0.02, -0.02, 1, 1),
                bbox_transform=ax[i][j].transAxes,
            )

        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")

plt.savefig("linlin_SHARP_zscored_by_spacecraft.png", dpi=200, bbox_inches="tight")
plt.show()