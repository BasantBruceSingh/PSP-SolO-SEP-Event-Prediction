"""Evaluate saved DeepSets checkpoints across repeated block holdouts."""

import torch
import argparse
import numpy as np
from set_classifiers import DeepSetEncoder, DeepSetMLPClassification


def main():
    """Evaluate saved DeepSets checkpoints across repeated block holdouts."""
    parser = argparse.ArgumentParser(
        description="Evaluation script for DeepSetLSTM regression model on SEP prediction task"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for classification (default: 0.5)",
    )

    args = parser.parse_args()

    import load_data_pointwise_kfold as load_data
    from train_classification import get_metrics_from_model

    sats = ["combined", "psp", "solo"]

    results = [{
        "acc": [],
        "precision": [],
        "recall/POD": [],
        "FAR": [],
        "TSS": [],
        "HSS": [],
        "F1": [],
    } for _ in range(3)]

    test_results = [{
        "acc": [],
        "precision": [],
        "recall/POD": [],
        "FAR": [],
        "TSS": [],
        "HSS": [],
        "F1": [],
    } for _ in range(3)]

    for i in range(10):
        for j, sat in enumerate(sats):
            train_loader, val_loader, test_loader = load_data.load_data_pointwise_kfold(i, sat)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = DeepSetMLPClassification(
                SHARP_length=10,
                mlp_hidden=16,
                dropout=0.45,
                individual_embedding_dim=32,
                set_embedding_dim=16
            ).to(device)

            model_weight_path = f"model_weights/Deepsets MLP Classifier kfold {i} {sat}_end.pth"
            model.load_state_dict(torch.load(model_weight_path, map_location=device))

            metrics = get_metrics_from_model(val_loader, model, args.threshold)
            test_metrics = get_metrics_from_model(test_loader, model, args.threshold)

            for col in results[j].keys():
                results[j][col].append(metrics[col])

            for col in test_results[j].keys():
                test_results[j][col].append(test_metrics[col])

    for j in range(3):
        print(f"Val results for {sats[j]}:")
        for col in results[j].keys():
            results[j][col] = np.array(results[j][col])
            mean = results[j][col].mean()
            std = results[j][col].std()
            print(f"{col}: {mean:.4f} ± {std:.4f}")

        print(f"Test Results for {sats[j]}:")
        for col in test_results[j].keys():
            test_results[j][col] = np.array(test_results[j][col])
            mean = test_results[j][col].mean()
            std = test_results[j][col].std()
            print(f"{col}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
