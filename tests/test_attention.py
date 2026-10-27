"""Pruebas unitarias para la red tabular con mecanismo de auto-atención y Focal Loss."""

import numpy as np
import torch

from src.models.attention import (
    FocalLoss,
    TabularAttentionClassifier,
    TabularAttentionNet,
)


def test_focal_loss_computation() -> None:
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([2.0, -2.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_tabular_attention_net_forward() -> None:
    num_features = 10
    batch_size = 16
    net = TabularAttentionNet(
        num_features=num_features,
        embed_dim=8,
        num_heads=2,
        ffn_dim=16,
    )
    x = torch.randn(batch_size, num_features)

    logits = net(x)
    assert logits.shape == (batch_size,)

    logits_with_attn, attn_weights = net(x, return_attention=True)
    assert logits_with_attn.shape == (batch_size,)
    assert attn_weights.shape == (batch_size, num_features, num_features)


def test_tabular_attention_classifier_fit_predict() -> None:
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 8
    x_tr = rng.randn(n_samples, n_features).astype(np.float32)
    y_tr = (rng.rand(n_samples) > 0.8).astype(int)

    clf = TabularAttentionClassifier(
        embed_dim=8,
        num_heads=2,
        ffn_dim=16,
        epochs=2,
        batch_size=32,
        random_state=42,
    )
    clf.fit(x_tr, y_tr)

    probas = clf.predict_proba(x_tr)
    assert probas.shape == (n_samples, 2)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    attns = clf.get_attention_weights(x_tr[:10])
    assert attns.shape == (10, n_features, n_features)
