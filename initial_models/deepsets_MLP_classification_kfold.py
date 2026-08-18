import torch
import torch.nn as nn
import load_data_pointwise_kfold as load_data
from train_classification import train, compute_metrics, get_metrics_from_model
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import numpy as np
import optuna

parser = argparse.ArgumentParser(
    description="Repeated holdout validation script for the Deepsets-MLP classification model on SEP prediction task"
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

for i in range(10):
    train_loader, val_loader, test_loader = load_data.load_data_pointwise_kfold(i)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepSetMLPClassification(
        SHARP_length=10,
        d_model=16,
        n_heads=2,
        num_layers=2,
        mlp_hidden=16,
        dropout=0.4,
        individual_embedding_dim=32,
        set_embedding_dim=16
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00001,
        weight_decay=3.3e-06
    )

    train(
        f"Deepsets MLP Classifier kfold {i}",
        model,
        optimizer,
        train_loader,
        val_loader,
        threshold=args.threshold,
        epochs=50,
        use_wandb=True
    )