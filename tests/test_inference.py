"""Tests for post-training inference helpers: load_ensemble and ensemble_predict."""

import torch
import torch.nn as nn
from pathlib import Path

from torch_distributed_ensemble import (
    ensemble_predict,
    load_ensemble,
)


class _TinyNet(nn.Module):
    def __init__(self, bias_val=0.0):
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, bias_val)

    def forward(self, x):
        return self.fc(x)


def _write_checkpoints(dirpath: Path, n: int):
    """Write n dummy checkpoints to dirpath using PerRankBestCheckpoint format."""
    dirpath.mkdir(parents=True, exist_ok=True)
    state_dicts = []
    for rank in range(n):
        model = _TinyNet(bias_val=float(rank))
        sd = model.state_dict()
        state_dicts.append(sd)
        ckpt = {"epoch": rank, "score": float(rank), "state_dict": sd}
        torch.save(ckpt, dirpath / f"best-rank={rank}.pt")
    return state_dicts


# ---------------------------------------------------------------------------
# load_ensemble
# ---------------------------------------------------------------------------


def test_load_ensemble_returns_correct_count(tmp_path):
    n = 3
    _write_checkpoints(tmp_path, n)
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet())
    assert len(models) == n
    for m in models:
        assert not m.training  # eval mode


def test_load_ensemble_loads_correct_weights(tmp_path):
    n = 2
    original_sds = _write_checkpoints(tmp_path, n)
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet())
    for model, orig_sd in zip(models, original_sds, strict=True):
        for key in orig_sd:
            assert torch.equal(model.state_dict()[key], orig_sd[key])


def test_load_ensemble_sorted_by_rank(tmp_path):
    """Models are returned in ascending rank order regardless of filesystem order."""
    n = 4
    _write_checkpoints(tmp_path, n)
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet())
    # Model i should have bias = i (matches the bias_val set during write)
    for i, model in enumerate(models):
        bias = model.fc.bias.data
        assert torch.allclose(bias, torch.full((2,), float(i))), f"Wrong model at index {i}"


def test_load_ensemble_not_found_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_ensemble(tmp_path, model_fn=lambda: _TinyNet())


def test_load_ensemble_custom_pattern(tmp_path):
    """Works with custom filename patterns."""
    model = _TinyNet()
    torch.save(model.state_dict(), tmp_path / "final-rank=0.pt")
    torch.save(model.state_dict(), tmp_path / "final-rank=1.pt")
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet(), pattern="final-rank={rank}.pt")
    assert len(models) == 2


# ---------------------------------------------------------------------------
# ensemble_predict
# ---------------------------------------------------------------------------


def test_ensemble_predict_mean(tmp_path):
    n = 3
    _write_checkpoints(tmp_path, n)
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet())
    x = torch.ones(5, 3)
    mean, std = ensemble_predict(models, x)
    assert mean.shape == (5, 2)
    assert std.shape == (5, 2)
    # mean should equal average of member biases [0, 1, 2] → 1.0
    assert torch.allclose(mean, torch.full((5, 2), 1.0), atol=1e-5)


def test_ensemble_predict_none(tmp_path):
    n = 2
    _write_checkpoints(tmp_path, n)
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet())
    x = torch.ones(4, 3)
    raw = ensemble_predict(models, x, aggregate="none")
    assert raw.shape == (2, 4, 2)


def test_ensemble_predict_no_grad(tmp_path):
    n = 2
    _write_checkpoints(tmp_path, n)
    models = load_ensemble(tmp_path, model_fn=lambda: _TinyNet())
    x = torch.ones(2, 3)
    with torch.enable_grad():
        mean, _ = ensemble_predict(models, x)
    assert not mean.requires_grad
