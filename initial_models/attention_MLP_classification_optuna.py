"""Tune Attention hyperparameters with Optuna."""

import torch
import argparse
from set_classifiers import ARSetAttentionEncoder, AttentionMLPClassification


def main():
    """Tune Attention hyperparameters with Optuna."""
    parser = argparse.ArgumentParser(
        description="Optuna tuning script for the attention-MLP classification model on SEP prediction task"
    )

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for converting probabilities to binary predictions")


    args = parser.parse_args()

    import load_data_pointwise as load_data
    from train_classification import train
    import optuna

    train_loader, val_loader, test_loader = load_data.load_data_pointwise()

    def objective(trial):
        """Train and evaluate one hyperparameter configuration for the Optuna study."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        d_model = 16
        n_heads = 4

        # d_model must be divisible by n_heads
        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()

        num_layers = 1
        mlp_hidden = trial.suggest_categorical("mlp_hidden", [4, 8, 16])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            3e-3,
            log=True,
        )

        weight_decay = 5e-6

        model = AttentionMLPClassification(
            SHARP_length=10,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        max_epochs = 100

        _, _, val_tss = train(
            "Attention MLP Classifier Optuna v3",
            model,
            optimizer,
            train_loader,
            val_loader,
            threshold=args.threshold,
            epochs=max_epochs,
            trial=trial,
            use_wandb=False
        )

        return val_tss[-1]


    study = optuna.create_study(
        direction="maximize",
        study_name="sep_attention_tss_v3",
        storage="sqlite:///optuna_sep_attention_tss_v3.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    print("\nBest trial:")
    print("value:", study.best_trial.value)
    print("params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
