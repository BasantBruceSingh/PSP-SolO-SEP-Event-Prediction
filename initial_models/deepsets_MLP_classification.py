import torch
import torch.nn as nn
import load_data_pointwise as load_data
from train_classification import train, compute_metrics, get_metrics_from_model
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import numpy as np
import optuna

parser = argparse.ArgumentParser(
    description="Training script for DeepSet MLP clasification model on SEP prediction task"
)

parser.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=6e-4,
                    help="Learning rate")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold for converting probabilities to binary predictions")
parser.add_argument("--mlp_hidden", type=float, default=16,
                    help="Number of neurons in the hidden layer of the MLP")
parser.add_argument("--dropout", type=float, default=0.4,
                    help="Dropout rate for the MLP layers")
parser.add_argument("--individual_embedding_dim", type=float, default=32,
                    help="Dimension of the individual embeddings in the DeepSet encoder")
parser.add_argument("--set_embedding_dim", type=float, default=16,
                    help="Dimension of the set embeddings in the DeepSet encoder")

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

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepSetMLPClassification(
    SHARP_length=10,
    mlp_hidden=args.mlp_hidden,
    dropout=args.dropout,
    individual_embedding_dim=args.individual_embedding_dim,
    set_embedding_dim=args.set_embedding_dim
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

train(
    "Deepsets MLP Classifier",
    model,
    optimizer,
    train_loader,
    val_loader,
    threshold=args.threshold,
    epochs=args.epochs,
    use_wandb=True
)