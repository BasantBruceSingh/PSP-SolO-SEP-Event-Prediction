"""Train the Attention classifier."""

import torch
import argparse
from set_classifiers import ARSetAttentionEncoder, AttentionMLPClassification


def main():
    """Train the Attention classifier."""
    parser = argparse.ArgumentParser(
        description="Training script for Attention-MLP regression model on SEP prediction task"
    )

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for converting probabilities to binary predictions")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Dimension of the tokens in the Attention layer")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of heads in the Attention layer")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers in the Attention layer")
    parser.add_argument("--mlp_hidden", type=int, default=16,
                        help="Number of neurons in the hidden layer of the MLP")
    parser.add_argument("--dropout", type=float, default=0.54,
                        help="Dropout rate for the Attention and MLP layers")
    parser.add_argument("--weight_decay", type=float, default=0.02,
                        help="Weight decay for the optimizer")
    parser.add_argument(
        "--psp_only",
        action="store_true",
        help="Train using only PSP data"
    )

    parser.add_argument(
        "--solo_only",
        action="store_true",
        help="Train using only SolO data"
    )

    args = parser.parse_args()

    from load_data_pointwise import load_data_pointwise
    from train_classification import train, get_weight_from_loader, evaluate_thresholds

    sat = "psp" if args.psp_only else ("solo" if args.solo_only else "combined")

    train_loader, val_loader, test_loader= load_data_pointwise(
        sat = sat
    )

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
        f"Attention MLP Classifier {sat}",
        model,
        optimizer,
        train_loader,
        val_loader,
        threshold=args.threshold,
        epochs=args.epochs,
        use_wandb=True,
        weight=get_weight_from_loader(train_loader)
    )

    print(evaluate_thresholds(model, val_loader))


if __name__ == "__main__":
    main()
