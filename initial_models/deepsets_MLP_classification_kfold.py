"""Train the DeepSets classifier across repeated block holdouts."""

import torch
import argparse
from set_classifiers import DeepSetEncoder, DeepSetMLPClassification


def main():
    """Train the DeepSets classifier across repeated block holdouts."""
    parser = argparse.ArgumentParser(
        description="Repeated holdout validation script for the Deepsets-MLP classification model on SEP prediction task"
    )

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for converting probabilities to binary predictions")

    parser.add_argument("--psp_only", action="store_true",
                        help="Use only PSP points (no SolO points)")

    parser.add_argument("--solo_only", action="store_true",
                        help="Use only SolO points (no PSP points)")

    parser.add_argument("--model_name", type=str, default="deepsets_MLP_classification_kfold",
                        help="Name of the model for saving and logging purposes")

    args = parser.parse_args()

    import load_data_pointwise_kfold as load_data
    from train_classification import train, get_weight_from_loader

    for i in range(10):
        sat = "psp" if args.psp_only else ("solo" if args.solo_only else "combined")
        train_loader, val_loader, test_loader = load_data.load_data_pointwise_kfold(i, sat=sat)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = DeepSetMLPClassification(
            SHARP_length=10,
            mlp_hidden=16,
            dropout=0.4,
            individual_embedding_dim=32,
            set_embedding_dim=16
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.0
        )

        train(
            f"Deepsets MLP Classifier kfold {i} {args.model_name}",
            model,
            optimizer,
            train_loader,
            val_loader,
            threshold=args.threshold,
            epochs=50,
            weight=get_weight_from_loader(train_loader),
            use_wandb=True,
        )


if __name__ == "__main__":
    main()
