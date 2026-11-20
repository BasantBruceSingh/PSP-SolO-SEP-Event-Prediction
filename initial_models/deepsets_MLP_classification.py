"""Train the DeepSets classifier."""

import torch
import argparse
from set_classifiers import DeepSetEncoder, DeepSetMLPClassification


def main():
    """Train the DeepSets classifier."""
    parser = argparse.ArgumentParser(
        description="Training script for DeepSet MLP clasification model on SEP prediction task"
    )

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for converting probabilities to binary predictions")
    parser.add_argument("--mlp_hidden", type=int, default=16,
                        help="Number of neurons in the hidden layer of the MLP")
    parser.add_argument("--dropout", type=int, default=0.4,
                        help="Dropout rate for the MLP layers")
    parser.add_argument("--individual_embedding_dim", type=int, default=32,
                        help="Dimension of the individual embeddings in the DeepSet encoder")
    parser.add_argument("--set_embedding_dim", type=int, default=16,
                        help="Dimension of the set embeddings in the DeepSet encoder")
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
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for the optimizer")

    args = parser.parse_args()

    from load_data_pointwise import load_data_pointwise
    from train_classification import train, get_weight_from_loader, evaluate_thresholds

    sat="psp" if args.psp_only else "solo" if args.solo_only else "combined"

    train_loader, val_loader, test_loader = load_data_pointwise(
        sat = sat
    )

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
        f"Deepsets MLP Classifier {sat}",
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
