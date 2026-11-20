"""Evaluate saved DeepSets checkpoints."""

import torch
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
from set_classifiers import DeepSetMLPClassification


def sigmoid_np(x):
    """Convert logits to probabilities with the logistic sigmoid."""
    x = np.asarray(x)
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def compute_metrics(y_true, y_pred):
    """Compute binary classification scores from true and predicted labels."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    total = TP + TN + FP + FN

    acc = (TP + TN) / max(total, 1)

    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)  # POD

    FAR = FP / max((TP + FP), 1)
    FPR = FP / max((FP + TN), 1)

    TSS = recall - FPR

    denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    HSS = (2 * (TP * TN - FP * FN)) / denom if denom != 0 else 0

    F1 = 2 * precision * recall / max((precision + recall), 1e-12)

    return {
        "cm": cm,
        "acc": acc,
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


def get_logits_and_labels(loader, model, device):
    """Collect model logits and corresponding labels from the loader as one-dimensional arrays."""
    y_logits = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for Xb, auxb, yb in loader:
            Xb = Xb.to(device)
            auxb = auxb.to(device)

            logits = model(Xb, auxb)

            y_logits.append(logits.detach().cpu().numpy().reshape(-1))
            y_true.append(yb.detach().cpu().numpy().reshape(-1))

    y_logits = np.concatenate(y_logits)
    y_true = np.concatenate(y_true).astype(int)

    return y_logits, y_true


def get_metrics_from_loader(loader, model, device, threshold):
    """Run inference and score the loader at the given probability threshold."""
    y_logits, y_true = get_logits_and_labels(loader, model, device)

    probs = sigmoid_np(y_logits)
    y_pred = (probs > threshold).astype(int)

    return compute_metrics(y_true, y_pred)


def print_metrics(name, metrics):
    """Print the confusion matrix and scores for the named split."""
    print(f"\n{name} metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name == "cm":
            print(f"{metric_name}:\n{metric_value}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")


def main():
    """Evaluate saved DeepSets checkpoints."""
    parser = argparse.ArgumentParser(
        description="Evaluate Deepsets SEP classifiers for Combined, PSP, and SolO"
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for converting probabilities to binary predictions")

    args = parser.parse_args()

    from load_data_pointwise import load_data_pointwise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for sat, dataset_name in (("combined", "Combined"), ("psp", "PSP"), ("solo", "SolO")):
        checkpoint = f"model_weights/Deepsets MLP Classifier {sat}_end.pth"
        print(f"\n=== {dataset_name} ===")
        print(f"Checkpoint: {checkpoint}")

        model = DeepSetMLPClassification(
            SHARP_length=10,
            mlp_hidden=16,
            dropout=0.4,
            individual_embedding_dim=32,
            set_embedding_dim=16,
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )

        loaders = load_data_pointwise(sat=sat)
        for split, loader in zip(("Train", "Validation", "Test"), loaders):
            metrics = get_metrics_from_loader(loader, model, device, args.threshold)
            print_metrics(split, metrics)


if __name__ == "__main__":
    main()
