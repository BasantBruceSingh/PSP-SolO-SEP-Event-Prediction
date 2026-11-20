"""Evaluate saved Attention checkpoints across repeated block holdouts."""

import os
import argparse
import numpy as np
import torch
from set_classifiers import ARSetAttentionEncoder, AttentionMLPClassification


def main():
    """Evaluate saved Attention checkpoints across repeated block holdouts."""
    parser = argparse.ArgumentParser(
        description="Evaluate Attention-MLP SEP classifier across 10 holdouts and satellites"
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--study_name", type=str, default="sep_attention_tss_v2")
    parser.add_argument("--study_db", type=str, default="sqlite:///optuna_sep_attention_tss_v2.db")

    args = parser.parse_args()

    import optuna
    import load_data_pointwise_kfold as load_data
    from train_classification import get_metrics_from_model

    SATS = ["combined", "psp", "solo"]
    METRICS = ["acc", "precision", "recall/POD", "FAR", "TSS", "HSS", "F1"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # Load Optuna hyperparameters once
    # ============================================================

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.study_db,
    )

    best = study.best_trial.params

    print("\nBest Optuna hyperparameters:")
    for key, value in best.items():
        print(f"  {key}: {value}")


    # ============================================================
    # Evaluation
    # ============================================================

    results_val = {
        sat: {metric: [] for metric in METRICS}
        for sat in SATS
    }

    results_test = {
        sat: {metric: [] for metric in METRICS}
        for sat in SATS
    }


    for i in range(10):
        print(f"\n{'=' * 60}")
        print(f"HOLDOUT {i}")
        print("=" * 60)

        for sat in SATS:
            print(f"\nEvaluating {sat.upper()}")

            train_loader, val_loader, test_loader = (
                load_data.load_data_pointwise_kfold(i, sat=sat)
            )

            model = AttentionMLPClassification(
                SHARP_length=10,
                d_model=int(best["d_model"]),
                n_heads=int(best["n_heads"]),
                num_layers=int(best.get("num_layers", 1)),
                mlp_hidden=int(best["mlp_hidden"]),
                dropout=float(best["dropout"]),
            ).to(DEVICE)

            # Match this path to your training script's model_name.
            model_weight_path = (
                f"model_weights/"
                f"Attention MLP Classifier kfold {sat} {i}_end.pth"
            )

            if not os.path.exists(model_weight_path):
                raise FileNotFoundError(
                    f"Missing checkpoint: {model_weight_path}\n"
                    "Check the model_name used when training each satellite."
                )

            model.load_state_dict(
                torch.load(
                    model_weight_path,
                    map_location=DEVICE,
                    weights_only=True,
                )
            )
            model.eval()

            metrics_val = get_metrics_from_model(
                val_loader, model, args.threshold
            )
            metrics_test = get_metrics_from_model(
                test_loader, model, args.threshold
            )

            for metric in METRICS:
                results_val[sat][metric].append(metrics_val[metric])
                results_test[sat][metric].append(metrics_test[metric])

            print(
                f"Val:  TSS={metrics_val['TSS']:.4f}, "
                f"F1={metrics_val['F1']:.4f}, "
                f"POD={metrics_val['recall/POD']:.4f}, "
                f"FAR={metrics_val['FAR']:.4f}"
            )
            print(
                f"Test: TSS={metrics_test['TSS']:.4f}, "
                f"F1={metrics_test['F1']:.4f}, "
                f"POD={metrics_test['recall/POD']:.4f}, "
                f"FAR={metrics_test['FAR']:.4f}"
            )


    # ============================================================
    # Print aggregate statistics
    # ============================================================

    for sat in SATS:
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {sat.upper()}")
        print("=" * 60)

        for name, results in [
            ("VALIDATION", results_val),
            ("TEST", results_test),
        ]:
            print(f"\n{name} (10 holdouts)")

            for metric in METRICS:
                values = np.array(results[sat][metric], dtype=float)

                print(
                    f"{metric:12s}: "
                    f"{values.mean():.4f} ± {values.std():.4f}"
                )


if __name__ == "__main__":
    main()
