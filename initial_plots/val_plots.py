import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load stats
# -----------------------------
deepsets_stats = []
attention_stats = []

for i in range(10):
    deepsets_stats.append(
        np.load(f"../initial_models/Deepsets MLP Classifier kfold_6 {i}_stats.npy")
    )

for i in range(10):
    attention_stats.append(
        np.load(f"../initial_models/Attention MLP Classifier kfold_6 {i}_stats.npy")
    )

deepsets_stats = np.array(deepsets_stats)
attention_stats = np.array(attention_stats)

print("DeepSets stats shape:", deepsets_stats.shape)
print("Attention stats shape:", attention_stats.shape)

# Shape: [num_runs, num_epochs, num_stats]
num_runs, num_epochs, num_stats = deepsets_stats.shape
epochs = np.arange(1, num_epochs + 1)

# Logged stats:
# 0: train loss
# 1: train accuracy
# 2: val accuracy
# 3: precision
# 4: recall/POD
# 5: FAR
# 6: TSS
# 7: HSS
# 8: F1

stat_indices = [2, 3, 4, 5, 6, 7, 8]
cols = ["Val ACC", "Precision", "POD", "FAR", "TSS", "HSS", "F1"]


def plot_model_runs(stats, model_name, output_path):
    fig, ax = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    fig.suptitle(f"Validation Metrics over Training Time for {model_name}", fontsize=16)

    for row in range(2):
        for col_idx in range(4):
            plot_idx = row * 4 + col_idx

            if plot_idx >= len(cols):
                ax[row, col_idx].axis("off")
                continue

            stat_idx = stat_indices[plot_idx]
            metric_name = cols[plot_idx]

            for k in range(num_runs):
                ax[row, col_idx].plot(
                    epochs,
                    stats[k, :, stat_idx],
                    linewidth=1.2,
                    alpha=0.75,
                    label=f"Run {k}" if plot_idx == 0 else None,
                )

            ax[row, col_idx].set_title(metric_name)
            ax[row, col_idx].set_xlabel("Epoch")
            ax[row, col_idx].set_ylabel(metric_name)
            ax[row, col_idx].grid(True, alpha=0.3)

            if plot_idx == 0:
                ax[row, col_idx].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

plot_model_runs(
    deepsets_stats,
    model_name="DeepSets-MLP",
    output_path="validation_metrics_deepsets_runs.png",
)

plot_model_runs(
    attention_stats,
    model_name="Attention-MLP",
    output_path="validation_metrics_attention_runs.png",
)