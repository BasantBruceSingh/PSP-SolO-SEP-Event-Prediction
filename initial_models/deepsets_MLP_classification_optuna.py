import torch
import torch.nn as nn
import load_data_pointwise as load_data
from train_classification import train, compute_metrics, get_metrics_from_model
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import numpy as np
import optuna

parser = argparse.ArgumentParser(
    description="Optuna tuning script for DeepSet MLP classification model on SEP prediction task"
)

parser.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold for converting probabilities to binary predictions")

args = parser.parse_args()

def sigmoid_np(x):
    # Avoid overflow
    if x < -10:
        return 0
    if x > 10:
        return 1
    return 1 / (1 + np.exp(-x))

class DeepSetEncoder(nn.Module):
    def __init__(self, SHARP_length, individual_embedding_dim, set_embedding_dim, dropout):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(SHARP_length, individual_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(individual_embedding_dim // 2, individual_embedding_dim),
        )

        self.rho = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(individual_embedding_dim, set_embedding_dim),
            nn.ReLU(),
            nn.Linear(set_embedding_dim, set_embedding_dim),
        )

    def forward(self, x, mask):
        h = self.phi(x)

        mask = mask.unsqueeze(-1)

        h = h * mask

        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = h.sum(dim=1) / counts

        z = self.rho(pooled)

        return z

class DeepSetMLPClassification(nn.Module):
    def __init__(
        self,
        SHARP_length,
        d_model,
        n_heads,
        num_layers,
        mlp_hidden,
        dropout,
        individual_embedding_dim,
        set_embedding_dim
    ):
        super().__init__()

        self.set_encoder = DeepSetEncoder(
            SHARP_length=SHARP_length,
            individual_embedding_dim=individual_embedding_dim,
            set_embedding_dim=set_embedding_dim,
            dropout=dropout
        )

        self.body = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(set_embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x, ar_mask):
        """
        x:       [B, A, F]
        ar_mask: [B, A]
        """

        z = self.set_encoder(x, ar_mask)

        pred = self.body(z)

        return pred.squeeze(-1)

train_loader = load_data.train_loader
val_loader = load_data.val_loader
test_loader = load_data.test_loader

def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_model = 16
    n_heads = 4

    # d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        raise optuna.exceptions.TrialPruned()

    num_layers = trial.suggest_int("num_layers", 1, 2)
    mlp_hidden = trial.suggest_categorical("mlp_hidden", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.1, 0.6)

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        3e-3,
        log=True,
    )

    weight_decay = 5e-6

    individual_embedding_dim = 8

    set_embedding_dim = 8

    model = DeepSetMLPClassification(
        SHARP_length=10,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        individual_embedding_dim=individual_embedding_dim,
        set_embedding_dim=set_embedding_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader = load_data.train_loader
    val_loader = load_data.val_loader

    max_epochs = 100

    train(
        "DeepSet MLP Classifier Optuna v3",
        model,
        optimizer,
        train_loader,
        val_loader,
        threshold=args.threshold,
        epochs=max_epochs,
        trial=trial,
        use_wandb=False
    )

    val_metrics = get_metrics_from_model(
        val_loader,
        model,
        threshold=args.threshold,
    )

    return val_metrics["TSS"]

study = optuna.create_study(
        direction="maximize",
        study_name="sep_deepsets_tss_v3",
        storage="sqlite:///optuna_sep_deepsets_tss_v3.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
    )

study.optimize(
    objective,
    n_trials=100,
    timeout=None,
)

print("\nBest trial:")
print("value:", study.best_trial.value)
print("params:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")