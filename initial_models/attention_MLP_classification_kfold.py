"""Train the Attention classifier across repeated block holdouts."""

import torch
import argparse
from set_classifiers import ARSetAttentionEncoder, AttentionMLPClassification


def main():
    """Train the Attention classifier across repeated block holdouts."""
    parser = argparse.ArgumentParser(
        description="Repeated holdout validation for the attention-MLP classification model on SEP prediction task"
    )

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for converting probabilities to binary predictions")
    parser.add_argument("--psp_only", action="store_true",
                        help="Use only PSP points (no SolO points)")
    parser.add_argument("--solo_only", action="store_true",
                        help="Use only SolO points (no PSP points)")
    parser.add_argument("--model_name", type=str, default="attention_MLP_classification_kfold",
                        help="Name of the model for saving and logging purposes")

    args = parser.parse_args()

    import load_data_pointwise_kfold as load_data
    from train_classification import train, get_weight_from_loader
    import optuna

    for i in range(10):
        sat = "psp" if args.psp_only else ("solo" if args.solo_only else "combined")
        train_loader, val_loader, test_loader = load_data.load_data_pointwise_kfold(i, sat=sat)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        study = optuna.load_study(
            study_name="sep_attention_tss_v2",
            storage="sqlite:///optuna_sep_attention_tss_v2.db",
        )

        best = study.best_trial.params
        print("Best hyperparameters found by Optuna:")
        for key, value in best.items():
            print(f"{key}: {value}")

        model = AttentionMLPClassification(
            SHARP_length=10,
            d_model=best["d_model"],
            n_heads=best["n_heads"],
            num_layers=1,
            mlp_hidden=best["mlp_hidden"],
            dropout=best["dropout"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=best["weight_decay"]
        )

        train(
            f"Attention MLP Classifier kfold {sat} {i}",
            model,
            optimizer,
            train_loader,
            val_loader,
            threshold=args.threshold,
            epochs=50,
            weight=get_weight_from_loader(train_loader),
            use_wandb=True
        )


if __name__ == "__main__":
    main()
