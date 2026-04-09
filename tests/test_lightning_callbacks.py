"""Tests for Lightning callbacks.

Covers: DistributedSeeder (via validate_weights worker mode),
RegressionEnsembleMetrics, EnsembleMetrics, PerRankCheckpoint, PerRankBestCheckpoint.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from torch_distributed_ensemble.lightning.callbacks import PerRankBestCheckpoint


def _run_cpu_worker(mode, tmp_path, timeout=120):
    """Launch a CPU Lightning worker and return the subprocess result.

    Uses nested srun when the outer SLURM job has 3+ tasks (2 for the worker
    + 1 for the pytest process itself), falling back to standalone mode otherwise.
    """
    worker_script = Path(__file__).parent / "_workers_lightning_cpu.py"
    env = os.environ.copy()
    env["TEST_TMPDIR"] = str(tmp_path)
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "0"))
    if slurm_ntasks >= 3:
        cmd = ["srun", "--ntasks=2", sys.executable, str(worker_script), mode, "--via-srun"]
    else:
        for key in [k for k in env if k.startswith(("PMI_", "PMIX_"))]:
            del env[key]
        cmd = [sys.executable, str(worker_script), mode]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# RegressionEnsembleMetrics
# ---------------------------------------------------------------------------


def test_regression_ensemble_metrics_logs_correct_keys(tmp_path):
    result = _run_cpu_worker("regression_metrics", tmp_path)
    assert result.returncode == 0, (
        f"Worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    world_size = 2
    for rank in range(world_size):
        m = json.loads((tmp_path / f"metrics_rank{rank}.json").read_text())
        assert "val/ens_mse" in m, f"rank {rank}: missing val/ens_mse"
        assert "val/avg_member_mse" in m, f"rank {rank}: missing val/avg_member_mse"
        assert "val/ensemble_gain" in m, f"rank {rank}: missing val/ensemble_gain"
        assert "val/ens_std" in m, f"rank {rank}: missing val/ens_std"
        for i in range(world_size):
            assert f"val/member_{i}_mse" in m, f"rank {rank}: missing val/member_{i}_mse"
        assert m["val/avg_member_mse"] - m["val/ens_mse"] == pytest.approx(m["val/ensemble_gain"], abs=1e-5)


# ---------------------------------------------------------------------------
# EnsembleMetrics (classification)
# ---------------------------------------------------------------------------


def test_ensemble_metrics_logs_correct_keys(tmp_path):
    result = _run_cpu_worker("ensemble_metrics", tmp_path)
    assert result.returncode == 0, (
        f"Worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    world_size = 2
    for rank in range(world_size):
        m = json.loads((tmp_path / f"metrics_rank{rank}.json").read_text())
        assert "val/ens_nll" in m, f"rank {rank}: missing val/ens_nll"
        for i in range(world_size):
            assert f"val/member_{i}_nll" in m, f"rank {rank}: missing val/member_{i}_nll"
        assert "val/disagreement" in m, f"rank {rank}: missing val/disagreement"
        assert 0.0 <= m["val/disagreement"] <= 1.0, f"rank {rank}: disagreement out of range"


# ---------------------------------------------------------------------------
# PerRankCheckpoint
# ---------------------------------------------------------------------------


def test_per_rank_checkpoint_saves_final_epoch(tmp_path):
    result = _run_cpu_worker("per_rank_checkpoint", tmp_path)
    assert result.returncode == 0, (
        f"Worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    dirpath = tmp_path / "checkpoints"
    for rank in range(2):
        ckpt_path = dirpath / f"final-rank={rank}.ckpt"
        assert ckpt_path.exists(), f"checkpoint not saved for rank {rank}"
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "state_dict" in ckpt, f"rank {rank}: checkpoint missing state_dict"


# ---------------------------------------------------------------------------
# PerRankBestCheckpoint
# ---------------------------------------------------------------------------


def test_per_rank_best_checkpoint_saves_files(tmp_path):
    result = _run_cpu_worker("best_checkpoint", tmp_path)
    assert result.returncode == 0, (
        f"Worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    dirpath = tmp_path / "best"
    for rank in range(2):
        ckpt_path = dirpath / f"best-rank={rank}.pt"
        assert ckpt_path.exists(), f"checkpoint not saved for rank {rank}"
        ckpt = torch.load(ckpt_path, weights_only=True)
        assert "epoch" in ckpt
        assert "score" in ckpt
        assert "state_dict" in ckpt
        assert isinstance(ckpt["score"], float)


def test_per_rank_best_checkpoint_rank_substitution(tmp_path):
    result = _run_cpu_worker("best_checkpoint_rank_metric", tmp_path)
    assert result.returncode == 0, (
        f"Worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    dirpath = tmp_path / "member_best"
    for rank in range(2):
        assert (dirpath / f"best-rank={rank}.pt").exists()


def test_per_rank_best_checkpoint_mode_validation():
    with pytest.raises(ValueError, match="mode must be"):
        PerRankBestCheckpoint(monitor="val/loss", mode="invalid")
