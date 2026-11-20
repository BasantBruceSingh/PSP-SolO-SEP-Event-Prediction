"""Summarize the temporal blocks in saved split manifests."""

import numpy as np
import pandas as pd

base = "npy"

manifest = pd.read_csv(
    f"{base}/shared_time_split_na.csv",
    parse_dates=["time"]
)

psp_times = set()
solo_times = set()

for split in ["train", "val", "test"]:
    psp_times.update(
        pd.to_datetime(
            np.load(f"{base}/times_t_{split}_psp_na.npy")
        )
    )

    solo_times.update(
        pd.to_datetime(
            np.load(f"{base}/times_t_{split}_solo_na.npy")
        )
    )

manifest["psp"] = manifest["time"].isin(psp_times)
manifest["solo"] = manifest["time"].isin(solo_times)

block_summary = manifest.groupby(
    ["block_id", "block_type", "split"]
).agg(
    psp=("psp", "any"),
    solo=("solo", "any"),
    psp_observations=("psp", "sum"),
    solo_observations=("solo", "sum")
).reset_index()

block_summary["satellite_type"] = np.select(
    [
        block_summary["psp"] & block_summary["solo"],
        block_summary["psp"],
        block_summary["solo"]
    ],
    ["Both", "PSP only", "SolO only"],
    default="Neither"
)

print("\nBlocks by satellite:")
print(block_summary["satellite_type"].value_counts())

print("\nBlocks by event type and satellite:")
print(pd.crosstab(
    block_summary["block_type"],
    block_summary["satellite_type"]
))

print("\nBlocks by split and satellite:")
print(pd.crosstab(
    block_summary["split"],
    block_summary["satellite_type"]
))

print("\nEvent blocks by split and satellite:")
print(pd.crosstab(
    block_summary.loc[
        block_summary["block_type"] == "event", "split"
    ],
    block_summary.loc[
        block_summary["block_type"] == "event", "satellite_type"
    ]
))