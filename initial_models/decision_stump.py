import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

cols = ["USFLUXL", "R_VALUE", "MEANGBL_GMM", "USFLUXZ", "MEANGBZ", "CMASKL", "LAT_FWT", "LON_FWT"]

class DecisionStump:
    def __init__(self, feature_index=None, threshold=None, polarity=1):
        self.threshold = threshold
        self.polarity = polarity

    def fit(self, x, y):
        n_samples = x.shape[0]

        max_score = -float('inf')

        for threshold in np.concat([np.unique(x), [np.min(x) - 1, np.max(x) + 1]]):
            for polarity in [1, -1]:
                predictions = np.ones(n_samples)
                if polarity == 1:
                    predictions[x < threshold] = 0
                else:
                    predictions[x > threshold] = 0

                score = self.compute_metrics(y, predictions)["TSS"]

                if score > max_score:
                    max_score = score
                    self.threshold = threshold
                    self.polarity = polarity

    def predict(self, x):
        n_samples = x.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[x < self.threshold] = 0
        else:
            predictions[x > self.threshold] = 0

        return predictions
    
    def compute_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        TN, FP, FN, TP = cm.ravel()

        total = TP + TN + FP + FN

        acc = (TP + TN) / max(total, 1)

        precision = TP / max((TP + FP), 1)
        recall    = TP / max((TP + FN), 1) # probability of detection (POD)

        FAR = FP / max((TP + FP), 1) # false alarm ratio
        FPR = FP / max((FP + TN), 1) # false positive rate (for TSS)

        # skill scores
        TSS = recall - FPR

        denom = (TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)
        HSS = (2*(TP*TN - FP*FN))/denom if denom != 0 else 0

        # F1 score
        F1 = (2 * precision * recall / max((precision + recall), 1e-12))

        # Weighted accuracy
        w_acc = (2.04 * TP + TN) / max(2.04 * TP + 2.04 * FN + TN + FP, 1)

        return {
            "cm": cm,
            "acc": acc,
            "w_acc":w_acc,
            "precision": precision,
            "recall/POD": recall,
            "FAR": FAR,
            "TSS": TSS,
            "HSS": HSS,
            "F1": F1,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }

    def prediction_metrics(self, x, y):
        predictions = self.predict(x)
        return self.compute_metrics(y, predictions)

# SolO only
X_train = np.load("../data_collection/npy/X_t_train_solo.npy")
X_val = np.load("../data_collection/npy/X_t_val_solo.npy")
X_test = np.load("../data_collection/npy/X_t_test_solo.npy")
y_train = np.load("../data_collection/npy/y_t_train_solo.npy")
y_val = np.load("../data_collection/npy/y_t_val_solo.npy")
y_test = np.load("../data_collection/npy/y_t_test_solo.npy")

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))

with open("logs/decision_stump_results_solo.txt", "w") as f:
    for i, col in enumerate(cols):
        ds = DecisionStump()

        ds.fit(X_train[:, i].reshape(-1), y_train > 100.0)
        metrics = ds.prediction_metrics(X_val[:, i].reshape(-1), y_val > 0.5)
        for metric_name, metric_value in metrics.items():
            f.write(f"Feature: {col}, Metric: {metric_name}, Value: {metric_value} \n")
        
        prediction_mask = ds.predict(X_val[:, i].reshape(-1)).astype(bool)
        ax[i//4, i%4].scatter(X_val[:, i][prediction_mask], y_val[prediction_mask], label="Predicted above", color="blue")
        ax[i//4, i%4].scatter(X_val[:, i][np.invert(prediction_mask)], y_val[np.invert(prediction_mask)], label="Predicted below", color="red")
        ax[i//4, i%4].axvline(ds.threshold, color="red", label="Threshold")
        ax[i//4, i%4].axhline(100.0, color="green", label="Jlinlin threshold")
        ax[i//4, i%4].set_xlabel(col)
        ax[i//4, i%4].set_ylabel("Jlinlin")
        ax[i//4, i%4].set_yscale("log")
        ax[i//4, i%4].legend()
        ax[i//4, i%4].set_title(col)

plt.savefig("decisionstumpsolo.png", dpi=200)

# PSP only
X_train = np.load("../data_collection/npy/X_t_train_psp.npy")
X_val = np.load("../data_collection/npy/X_t_val_psp.npy")
X_test = np.load("../data_collection/npy/X_t_test_psp.npy")
y_train = np.load("../data_collection/npy/y_t_train_psp.npy")
y_val = np.load("../data_collection/npy/y_t_val_psp.npy")
y_test = np.load("../data_collection/npy/y_t_test_psp.npy")

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))

with open("logs/decision_stump_results_psp.txt", "w") as f:
    for i, col in enumerate(cols):
        ds = DecisionStump()

        ds.fit(X_train[:, i].reshape(-1), y_train > 100.0)
        metrics = ds.prediction_metrics(X_val[:, i].reshape(-1), y_val > 0.5)
        for metric_name, metric_value in metrics.items():
            f.write(f"Feature: {col}, Metric: {metric_name}, Value: {metric_value} \n")
        
        prediction_mask = ds.predict(X_val[:, i].reshape(-1)).astype(bool)
        ax[i//4, i%4].scatter(X_val[:, i][prediction_mask], y_val[prediction_mask], label="Predicted above", color="blue")
        ax[i//4, i%4].scatter(X_val[:, i][np.invert(prediction_mask)], y_val[np.invert(prediction_mask)], label="Predicted below", color="red")
        ax[i//4, i%4].axvline(ds.threshold, color="red", label="Threshold")
        ax[i//4, i%4].axhline(100.0, color="green", label="Jlinlin threshold")
        ax[i//4, i%4].set_xlabel(col)
        ax[i//4, i%4].set_ylabel("Jlinlin")
        ax[i//4, i%4].set_yscale("log")
        ax[i//4, i%4].legend()
        ax[i//4, i%4].set_title(col)

plt.savefig("decisionstumppsp.png", dpi=200)

# Combined dataset

X_train = np.load("../data_collection/npy/X_t_train_combined.npy")
X_val = np.load("../data_collection/npy/X_t_val_combined.npy")
X_test = np.load("../data_collection/npy/X_t_test_combined.npy")
y_train = np.load("../data_collection/npy/y_t_train_combined.npy")
y_val = np.load("../data_collection/npy/y_t_val_combined.npy")
y_test = np.load("../data_collection/npy/y_t_test_combined.npy")
sat_train = np.load("../data_collection/npy/sat_t_train_combined.npy", allow_pickle=True)
sat_val = np.load("../data_collection/npy/sat_t_val_combined.npy", allow_pickle=True)
sat_test = np.load("../data_collection/npy/sat_t_test_combined.npy", allow_pickle=True)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))

with open("logs/decision_stump_results_combined.txt", "w") as f:
    for k, feature_name in enumerate(cols):
        r = k // 4
        c = k % 4
        this_ax = ax[r, c]

        ds = DecisionStump()

        x_train = X_train[:, k].reshape(-1)
        x_val = X_val[:, k].reshape(-1)

        # y_val is z-scored log Jlinlin
        y_train_label = y_train > 1.0
        y_val_label = y_val > 1.0

        finite_train = np.isfinite(x_train) & np.isfinite(y_train)
        finite_val = np.isfinite(x_val) & np.isfinite(y_val)

        ds.fit(x_train[finite_train], y_train_label[finite_train])
        metrics = ds.prediction_metrics(x_val[finite_val], y_val_label[finite_val])

        f.write(f"\nFeature: {feature_name}\n")
        f.write(f"Threshold: {ds.threshold}, Polarity: {ds.polarity}\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"Metric: {metric_name}, Value: {metric_value}\n")

        prediction_mask = ds.predict(x_val).astype(bool)

        sat_val_str = sat_val.astype(str)
        solo_mask = sat_val_str == "solo"
        psp_mask = sat_val_str == "psp"
        prediction_mask = ds.predict(X_val[:, k].reshape(-1)).astype(bool)

        # Shared finite data range for both satellites
        x_plot_all = x_val[finite_val]
        y_plot_all = y_val[finite_val]

        if len(x_plot_all) == 0:
            this_ax.set_title(f"{feature_name}: no valid values")
            continue

        x_bins = np.linspace(np.nanmin(x_plot_all), np.nanmax(x_plot_all), 51)
        y_bins = np.linspace(np.nanmin(y_plot_all), np.nanmax(y_plot_all), 51)

        solo_plot_mask_0 = finite_val & solo_mask & np.invert(prediction_mask)
        solo_plot_mask_1 = finite_val & solo_mask & prediction_mask
        psp_plot_mask_0 = finite_val & psp_mask & np.invert(prediction_mask)
        psp_plot_mask_1 = finite_val & psp_mask & prediction_mask

        # Draw PSP and SolO on same axes, same bins
        this_ax.hist2d(
            x_val[psp_plot_mask_0],
            y_val[psp_plot_mask_0],
            bins=(x_bins, y_bins),
            cmin=1,
            cmap="Blues",
            alpha=0.65,
            norm=LogNorm(),
        )

        this_ax.hist2d(
            x_val[psp_plot_mask_1],
            y_val[psp_plot_mask_1],
            bins=(x_bins, y_bins),
            cmin=1,
            cmap="Greens",
            alpha=0.65,
            norm=LogNorm(),
        )

        this_ax.hist2d(
            x_val[solo_plot_mask_0],
            y_val[solo_plot_mask_0],
            bins=(x_bins, y_bins),
            cmin=1,
            cmap="Oranges",
            alpha=0.65,
            norm=LogNorm(),
        )

        this_ax.hist2d(
            x_val[solo_plot_mask_1],
            y_val[solo_plot_mask_1],
            bins=(x_bins, y_bins),
            cmin=1,
            cmap="Reds",
            alpha=0.65,
            norm=LogNorm(),
        )

        this_ax.axvline(ds.threshold, color="red", label="Stump threshold")
        this_ax.axhline(1.0, color="green", label="log-Jlinlin threshold")

        this_ax.set_xlabel(feature_name)
        this_ax.set_ylabel("z-scored log Jlinlin")
        this_ax.set_title(feature_name)

        # Fake legend handles for the two heatmaps
        this_ax.plot([], [], color="tab:blue", linewidth=6, alpha=0.65, label="PSP density (predicted below)")
        this_ax.plot([], [], color="tab:green", linewidth=6, alpha=0.65, label="PSP density (predicted above)")
        this_ax.plot([], [], color="tab:orange", linewidth=6, alpha=0.65, label="SolO density (predicted below)")
        this_ax.plot([], [], color="tab:red", linewidth=6, alpha=0.65, label="SolO density (predicted above)")
        this_ax.legend(fontsize=7)

plt.savefig("decisionstumpcombined.png", dpi=200)