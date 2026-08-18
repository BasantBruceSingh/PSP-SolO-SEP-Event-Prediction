from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np
import random

X_train = np.load("../data_collection/npy/X_t_train_combined_na.npy")
X_val = np.load("../data_collection/npy/X_t_val_combined_na.npy")
X_test = np.load("../data_collection/npy/X_t_test_combined_na.npy")
y_train = np.load("../data_collection/npy/y_t_train_combined_na.npy")
y_val = np.load("../data_collection/npy/y_t_val_combined_na.npy")
y_test = np.load("../data_collection/npy/y_t_test_combined_na.npy")
mask_train = np.load("../data_collection/npy/mask_t_train_combined_na.npy")
mask_val = np.load("../data_collection/npy/mask_t_val_combined_na.npy")
mask_test = np.load("../data_collection/npy/mask_t_test_combined_na.npy")

print("X_train:", X_train.shape)
print("mask_train:", mask_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("mask_val:", mask_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("mask_test:", mask_test.shape)
print("y_test:", y_test.shape)

class SHARPLinlinSequenceDataset(Dataset):
    def __init__(self, X, ar_mask, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ar_mask = torch.tensor(ar_mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ar_mask[idx], self.y[idx]

batch_size = 16

print("Share of positive samples in train:", (y_train > 1.0).mean())

train_ds = SHARPLinlinSequenceDataset(X_train, mask_train, (y_train > 1.0).astype(int))
val_ds = SHARPLinlinSequenceDataset(X_val, mask_val, (y_val > 1.0).astype(int))
test_ds = SHARPLinlinSequenceDataset(X_test, mask_test, (y_test > 1.0).astype(int))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)