"""Shared pointwise classifiers for active-region sets."""


import torch.nn as nn


class ARSetAttentionEncoder(nn.Module):
    def __init__(
        self,
        SHARP_length=10,
        d_model=8,
        n_heads=4,
        num_layers=1,
        dropout=0.4,
        *,
        projection_relu=False,
        feedforward_multiplier=2,
    ):
        """Configure the input projection and Transformer encoder for active-region sets."""
        super().__init__()

        projection = [nn.Linear(SHARP_length, d_model)]
        if projection_relu:
            projection.append(nn.ReLU())
        self.input_proj = nn.Sequential(*projection, nn.Dropout(dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_multiplier * d_model,
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

        ar_mask = ar_mask.bool()
        padding = ~ar_mask.unsqueeze(-1)
        h = self.input_proj(x.masked_fill(padding, 0))
        if h.size(1) == 0:
            return h.sum(dim=1)

        key_padding_mask = ~ar_mask
        # Give empty sets one dummy key so attention always has a valid key.
        key_padding_mask[:, 0] &= ar_mask.any(dim=1)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        # Use the original mask so the dummy key contributes nothing.
        h = h.masked_fill(padding, 0)
        counts = ar_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return h.sum(dim=1) / counts



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
        """Build an Attention set encoder and a binary classification head."""
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


class DeepSetEncoder(nn.Module):
    def __init__(self, SHARP_length, individual_embedding_dim, set_embedding_dim, dropout):
        """Build the per-region and pooled-set embedding networks."""
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
        """Encode a batch of active-region sets by averaging the embeddings of valid regions."""
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
        """Build a DeepSets encoder and a binary classification head."""
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
