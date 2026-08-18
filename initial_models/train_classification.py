import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import optuna


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def check_accuracy(loader, model, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    total_correct = 0
    num_samples = 0

    model = model.to(device=device)
    model.eval()

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device=device, dtype=dtype)
            mask = mask.to(device=device, dtype=dtype)
            # y = y.to(device=device, dtype=dtype)

            output = model(x, mask)
            output = sigmoid_np(output.cpu().numpy())

            predicted_labels = (output >= threshold).astype(int)

            total_correct += (predicted_labels == y).sum().item()
            num_samples += y.shape[0]

    return total_correct / num_samples


def check_per_class_accuracy(loader, model, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    dtype = torch.float32

    total_correct_0 = 0
    total_correct_1 = 0
    num_samples_0 = 0
    num_samples_1 = 0

    model = model.to(device=device)
    model.eval()

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device=device, dtype=dtype)
            mask = mask.to(device=device, dtype=dtype)
            # y = y.to(device=device, dtype=dtype)

            output = model(x, mask)
            output = sigmoid_np(output.cpu().numpy())

            predicted_labels = (output >= threshold).astype(int)

            total_correct_0 += ((predicted_labels == y) & (y == 0)).sum()
            total_correct_1 += ((predicted_labels == y) & (y == 1)).sum()

            num_samples_0 += (y == 0).sum()
            num_samples_1 += (y == 1).sum()

    acc_0 = total_correct_0 / num_samples_0 if num_samples_0 > 0 else 0
    acc_1 = total_correct_1 / num_samples_1 if num_samples_1 > 0 else 0

    return acc_0, acc_1


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    total = TP + TN + FP + FN

    acc = (TP + TN) / max(total, 1)

    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)  # probability of detection (POD)

    FAR = FP / max((TP + FP), 1)  # false alarm ratio
    FPR = FP / max((FP + TN), 1)  # false positive rate (for TSS)

    # skill scores
    TSS = recall - FPR

    denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    HSS = (2 * (TP * TN - FP * FN)) / denom if denom != 0 else 0

    # F1 score
    F1 = 2 * precision * recall / max((precision + recall), 1e-12)

    return {
        "cm": cm,
        "acc": acc,
        "precision": precision,
        "recall/POD": recall,
        "FAR": FAR,
        "TSS": TSS,
        "HSS": HSS,
        "F1": F1,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }


def get_metrics_from_model(loader, model, threshold):
    # containers
    y_pred, y_true = [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model.eval()

    with torch.no_grad():
        for Xb, auxb, yb in loader:
            preds = model(Xb.to(device), auxb.to(device))

            y_pred.append(preds.cpu().numpy())
            y_true.append(yb.numpy())

    model.eval()

    with torch.no_grad():
        y_pred_raw = []

        for Xb, auxb, yb in loader:
            preds = model(Xb.to(device), auxb.to(device))
            y_pred_raw.append(preds.cpu().numpy())

    # stack into a single array
    y_pred_raw = np.concatenate([yp.reshape(-1) for yp in y_pred_raw])

    # stack into arrays
    y_true = np.concatenate([yb.reshape(-1) for yb in y_true]).astype(int)

    metrics_val = compute_metrics(
        y_true,
        (
            np.vectorize(sigmoid_np)(np.concatenate(y_pred).reshape(-1))
            > threshold
        ).astype(int),
    )

    return metrics_val


def train(
    model_name,
    model,
    optimizer,
    loader_train,
    loader_val,
    threshold,
    epochs=1,
    weight=6.56,
    print_every=10,
    trial=None,
    use_wandb=True,
):
    """
    Train a model using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - loader_train: A dataloader containing the train dataset
    - loader_val: A dataloader containing the validation dataset
    - epochs: (Optional) An integer giving the number of epochs to train for
    - print_every: (Optional) An integer specifying how often to print the loss.

    Returns:
    Nothing, but prints model losses and accuracies during training.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    if use_wandb:
        run = wandb.init(
            project="sep-prediction",
            name=model_name + " test",
            mode="offline",
            config={
                "learning_rate": optimizer.param_groups[-1]["lr"],
                "epochs": epochs,
                "model": model_name,
                "batch_size": 32,
            },
        )

    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.5,
        total_iters=30,
    )

    model = model.to(device=device)

    train_acc = []
    val_acc = []
    val_tss = []

    loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight], device=device)
    )

    best_tss = -1

    for e in range(epochs):
        train_losses = []

        for t, (x, mask, y) in enumerate(loader_train):
            model.train()

            x = x.to(device=device, dtype=dtype)
            mask = mask.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            output = model(x, mask)
            batch_loss = loss(output, y)

            train_losses.append(batch_loss.item())

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if t % print_every == 0:
                print("Epoch {}, iteration {}, loss = {}".format(e, t, batch_loss.item()))

        scheduler.step()

        print("Epoch {} done".format(e))

        train_acc_this = check_accuracy(loader_train, model, threshold)
        train_acc.append(train_acc_this)
        print("train accuracy", train_acc_this)

        val_acc_this = check_accuracy(loader_val, model, threshold)
        val_acc.append(val_acc_this)
        print("val accuracy: ", val_acc_this)

        metrics = get_metrics_from_model(loader_val, model, threshold)
        val_tss.append(metrics["TSS"])

        if metrics["TSS"] > best_tss:
            print(f"best epoch: {e}")
            torch.save(model.state_dict(), "model_weights/" + model_name + "_best.pth")
            best_tss = metrics["TSS"]

        train_losses = np.array(train_losses)

        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_losses.sum() / len(train_losses),
                    "train_accuracy": train_acc_this,
                    "val_accuracy": val_acc_this,
                    "precision": metrics["precision"],
                    "recall/POD": metrics["recall/POD"],
                    "FAR": metrics["FAR"],
                    "TSS": metrics["TSS"],
                    "HSS": metrics["HSS"],
                    "F1": metrics["F1"],
                }
            )

        if trial is not None:
            trial.report(metrics["TSS"], e)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        acc_0, acc_1 = check_per_class_accuracy(loader_train, model, threshold)
        acc_0_val, acc_1_val = check_per_class_accuracy(loader_val, model, threshold)

        print(
            f"Train false class accuracy: {acc_0:.4f}, "
            f"Train true class accuracy: {acc_1:.4f}"
        )
        print(
            f"Val false class accuracy: {acc_0_val:.4f}, "
            f"Val true class accuracy: {acc_1_val:.4f}"
        )

    if use_wandb:
        wandb.finish()

    torch.save(model.state_dict(), "model_weights/" + model_name + "_end.pth")

    return train_acc, val_acc, val_tss