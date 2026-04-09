"""Tests for the vanilla PyTorch functional API.

Covers: setup_independent_worker, gather_ensemble_metrics (gloo + nccl),
checkpoint roundtrip, and backend auto-detection.
"""

import os
import unittest.mock as mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from torch_distributed_ensemble.functional import (
    gather_ensemble_metrics,
    setup_independent_worker,
)


# ---------------------------------------------------------------------------
# CPU / gloo workers
# ---------------------------------------------------------------------------


def _worker_weights_diverge(rank, world_size, results):
    """Each worker trains a tiny model for 1 step; weights should diverge."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    setup_independent_worker(base_seed=42, backend="gloo")

    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(8, 4)
    y = model(x).sum()
    y.backward()
    optimizer.step()

    flat = model.weight.data.detach().flatten()
    gathered = gather_ensemble_metrics(flat)

    results[rank] = gathered.clone()
    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def _worker_gather_metrics(rank, world_size, results):
    """Each worker creates a rank-dependent tensor and gathers."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    setup_independent_worker(base_seed=0, backend="gloo")

    local = torch.tensor([rank * 1.0])
    gathered = gather_ensemble_metrics(local)

    results[rank] = gathered.clone()
    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def _worker_checkpoint_roundtrip(rank, world_size, results, tmpdir):
    """Save per-rank checkpoints, then verify own roundtrips and cross-rank differs."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29515"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    setup_independent_worker(base_seed=42, backend="gloo")

    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(8, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    ckpt_path = os.path.join(tmpdir, f"member_{rank}.pt")
    torch.save({"state_dict": model.state_dict(), "epoch": 5}, ckpt_path)
    trained_weights = model.weight.data.clone()

    dist.barrier()

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    assert torch.equal(model.weight.data, trained_weights)

    other_rank = 1 - rank
    other_path = os.path.join(tmpdir, f"member_{other_rank}.pt")
    other_ckpt = torch.load(other_path, weights_only=True)
    model.load_state_dict(other_ckpt["state_dict"])
    assert not torch.allclose(model.weight.data, trained_weights)

    results[rank] = True
    dist.destroy_process_group()


def test_weights_diverge():
    world_size = 2
    results = mp.Manager().dict()
    mp.spawn(_worker_weights_diverge, args=(world_size, results), nprocs=world_size, join=True)

    w0 = results[0]
    assert not torch.allclose(w0[0], w0[1]), "Weights should diverge across workers"


def test_gather_ensemble_metrics():
    world_size = 2
    results = mp.Manager().dict()
    mp.spawn(_worker_gather_metrics, args=(world_size, results), nprocs=world_size, join=True)

    gathered = results[0]
    assert gathered.shape == (2, 1)
    assert torch.allclose(gathered, torch.tensor([[0.0], [1.0]]))


def test_checkpoint_roundtrip(tmp_path):
    world_size = 2
    results = mp.Manager().dict()
    mp.spawn(
        _worker_checkpoint_roundtrip,
        args=(world_size, results, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    assert results[0] is True
    assert results[1] is True


# ---------------------------------------------------------------------------
# GPU / nccl workers
# ---------------------------------------------------------------------------


def _worker_nccl_diverge(rank, world_size, results):
    """Train a tiny model on each GPU with NCCL; weights should diverge."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29510"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    setup_independent_worker(base_seed=42, backend="nccl")

    model = torch.nn.Linear(4, 4).cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(8, 4, device=f"cuda:{rank}")
    y = model(x).sum()
    y.backward()
    optimizer.step()

    flat = model.weight.data.detach().flatten()
    gathered = gather_ensemble_metrics(flat)

    results[rank] = gathered.cpu().clone()
    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def _worker_nccl_gather(rank, world_size, results):
    """Each GPU creates a rank-dependent tensor; gather via NCCL."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    setup_independent_worker(base_seed=0, backend="nccl")

    local = torch.tensor([rank * 1.0], device=f"cuda:{rank}")
    gathered = gather_ensemble_metrics(local)

    results[rank] = gathered.cpu().clone()
    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


import pytest


@pytest.mark.multigpu
def test_nccl_weights_diverge():
    """Two GPUs train independently via NCCL — weights must differ."""
    world_size = 2
    results = mp.Manager().dict()
    mp.spawn(_worker_nccl_diverge, args=(world_size, results), nprocs=world_size, join=True)

    gathered = results[0]
    assert not torch.allclose(gathered[0], gathered[1]), "NCCL weights should diverge"


@pytest.mark.multigpu
def test_nccl_gather_metrics():
    """all_gather works correctly over NCCL."""
    world_size = 2
    results = mp.Manager().dict()
    mp.spawn(_worker_nccl_gather, args=(world_size, results), nprocs=world_size, join=True)

    gathered = results[0]
    assert gathered.shape == (2, 1)
    assert torch.allclose(gathered, torch.tensor([[0.0], [1.0]]))


@pytest.mark.gpu
def test_cuda_available():
    """Sanity check that CUDA works on this node."""
    assert torch.cuda.is_available()
    t = torch.tensor([1.0]).cuda()
    assert t.device.type == "cuda"


# ---------------------------------------------------------------------------
# Backend auto-detection
# ---------------------------------------------------------------------------


def test_setup_independent_worker_backend_autodetect():
    """backend=None should select 'nccl' with CUDA or 'gloo' without."""
    with (
        mock.patch("torch.distributed.init_process_group") as mock_init,
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch("torch.distributed.get_world_size", return_value=1),
        mock.patch("torch.cuda.is_available", return_value=False),
    ):
        setup_independent_worker(base_seed=0, backend=None)
        called_backend = mock_init.call_args[1]["backend"]
        assert called_backend == "gloo"

    with (
        mock.patch("torch.distributed.init_process_group") as mock_init,
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch("torch.distributed.get_world_size", return_value=1),
        mock.patch("torch.cuda.set_device"),
        mock.patch("torch.cuda.is_available", return_value=True),
    ):
        setup_independent_worker(base_seed=0, backend=None)
        called_backend = mock_init.call_args[1]["backend"]
        assert called_backend == "nccl"
