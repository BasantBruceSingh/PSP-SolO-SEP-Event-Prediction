import torch
import torch.nn as nn
import load_data_pointwise as load_data
from train_classification import train, compute_metrics, get_metrics_from_model
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import numpy as np
import optuna

parser = argparse.ArgumentParser(
    description="Training script for Attention-MLP regression model on SEP prediction task"
)

parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-5,
                    help="Learning rate")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold for converting probabilities to binary predictions")
parser.add_argument("--d_model", type=float, default=32,
                    help="Dimension of the tokens in the Attention layer")
parser.add_argument("--n_heads", type=float, default=4,
                    help="Number of heads in the Attention layer")
parser.add_argument("--num_layers", type=float, default=1,
                    help="Number of layers in the Attention layer")
parser.add_argument("--mlp_hidden", type=float, default=16,
                    help="Number of neurons in the hidden layer of the MLP")
parser.add_argument("--dropout", type=float, default=0.54,
                    help="Dropout rate for the Attention and MLP layers")
parser.add_argument("--weight_decay", type=float, default=0.02,
                    help="Weight decay for the optimizer")


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

model = AttentionMLPClassification(
    SHARP_length=10,
    d_model=args.d_model,
    n_heads=args.n_heads,
    num_layers=args.num_layers,
    mlp_hidden=args.mlp_hidden,
    dropout=args.dropout,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)

train(
    "Attention MLP Classifier",
    model,
    optimizer,
    train_loader,
    val_loader,
    threshold=args.threshold,
    epochs=args.epochs,
    use_wandb=True
)