"""Módulo de clasificación tabular con mecanismo de auto-atención (Multi-Head Self-Attention)."""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """Focal Loss para mitigar el desbalance extremo de clases en clasificación binaria."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TabularAttentionNet(nn.Module):
    """Arquitectura neuronal con proyección de variables y bloque de Multi-Head Self-Attention."""

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 16,
        num_heads: int = 4,
        ffn_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Proyección independiente de cada variable a un token de dimensión embed_dim
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(num_features)]
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Clasificador final sobre las representaciones proyectadas y atendidas
        classifier_input_dim = num_features * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, num_features]
        batch_size = x.size(0)

        # Tokenización de variables: lista de [B, embed_dim] -> stack a [B, num_features, embed_dim]
        tokens = [
            self.feature_projections[i](x[:, i : i + 1])
            for i in range(self.num_features)
        ]
        token_tensor = torch.stack(tokens, dim=1)

        # Bloque de Auto-Atención con residual
        attn_out, attn_weights = self.mha(token_tensor, token_tensor, token_tensor)
        x_norm = self.norm1(token_tensor + attn_out)

        # Bloque FeedForward con residual
        ffn_out = self.ffn(x_norm)
        encoded = self.norm2(x_norm + ffn_out)

        # Aplanar representaciones para clasificación
        flattened = encoded.reshape(batch_size, -1)
        logits = self.classifier(flattened).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits


class TabularAttentionClassifier:
    """Estimador con interfaz Scikit-learn para entrenamiento y evaluación con early stopping."""

    def __init__(
        self,
        embed_dim: int = 16,
        num_heads: int = 4,
        ffn_dim: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 15,
        alpha_focal: float = 0.25,
        gamma_focal: float = 2.0,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.model: TabularAttentionNet | None = None
        self.num_features_: int = 0
        self.history_: list[dict[str, float]] = []

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TabularAttentionClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_tr_arr = np.asarray(x_train, dtype=np.float32)
        y_tr_arr = np.asarray(y_train, dtype=np.float32)
        self.num_features_ = x_tr_arr.shape[1]

        self.model = TabularAttentionNet(
            num_features=self.num_features_,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
        ).to(self.device)

        criterion = FocalLoss(alpha=self.alpha_focal, gamma=self.gamma_focal)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        dataset = TensorDataset(torch.from_numpy(x_tr_arr), torch.from_numpy(y_tr_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_pr_auc = -1.0
        best_state: dict[str, Any] | None = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)

            train_loss /= len(x_tr_arr)
            epoch_metrics = {"epoch": epoch, "train_loss": train_loss}

            if x_val is not None and y_val is not None:
                val_proba = self.predict_proba(x_val)[:, 1]
                val_pr_auc = float(average_precision_score(y_val, val_proba))
                epoch_metrics["val_pr_auc"] = val_pr_auc

                if val_pr_auc > best_pr_auc:
                    best_pr_auc = val_pr_auc
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

            self.history_.append(epoch_metrics)

        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")

        self.model.eval()
        x_arr = np.asarray(x, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(x_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        probas = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                logits = self.model(bx)
                p1 = torch.sigmoid(logits).cpu().numpy()
                probas.append(p1)

        p1_all = np.concatenate(probas, axis=0)
        p0_all = 1.0 - p1_all
        return np.column_stack([p0_all, p1_all])

    def get_attention_weights(self, x: np.ndarray) -> np.ndarray:
        """Extrae la matriz media de atención inter-variable para una muestra dada."""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de extraer atención.")

        self.model.eval()
        x_arr = np.asarray(x, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(x_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        attns = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                _, weights = self.model(bx, return_attention=True)
                attns.append(weights.cpu().numpy())

        return np.concatenate(attns, axis=0)
