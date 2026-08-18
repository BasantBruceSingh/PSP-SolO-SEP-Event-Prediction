import torch
import torch.nn as nn
import load_data_pointwise as load_data
from train_classification import train, compute_metrics, get_metrics_from_model
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import numpy as np
import optuna

parser = argparse.ArgumentParser(
    description="Optuna tuning script for the attention-MLP classification model on SEP prediction task"
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

class ARSetAttentionEncoder(nn.Module):
    def __init__(
        self,
        SHARP_length=10,
        d_model=8,
        n_heads=4,
        num_layers=1,
        dropout=0.4,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(SHARP_length, d_model),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2*d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x, ar_mask):
        """
        x:       [B, A, F]
        ar_mask: [B, A], True for real ARs, False for padded ARs
        """

        h = self.input_proj(x)  # [B, A, d_model]

        # PyTorch Transformer expects True for positions to ignore.
        key_padding_mask = ~ar_mask.bool()  # [B, A]

        h = self.encoder(
            h,
            src_key_padding_mask=key_padding_mask,
        )  # [B, A, d_model]

        # Masked mean pooling over real ARs only.
        mask_float = ar_mask.unsqueeze(-1).float()  # [B, A, 1]
        h = h * mask_float

        counts = mask_float.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = h.sum(dim=1) / counts                # [B, d_model]

        return pooled

class AttentionMLPClassification(nn.Module):
    def __init__(
        self,
        SHARP_length,
        d_model,
        n_heads,
        num_layers,
        mlp_hidden,
        dropout,
    ):
        super().__init__()

        self.set_encoder = ARSetAttentionEncoder(
            SHARP_length=SHARP_length,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.body = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, mlp_hidden),
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

    train_loader = load_data.train_loader
    val_loader = load_data.val_loader

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


print("\nBest trial:")
print("value:", study.best_trial.value)
print("params:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")