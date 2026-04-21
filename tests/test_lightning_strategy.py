import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from torch_distributed_ensemble.lightning.strategy import DistributedEnsembleStrategy


def test_strategy_no_sync():
    strategy = DistributedEnsembleStrategy()

    assert strategy.broadcast("hello") == "hello"

    t = torch.tensor(1.0)
    result = strategy.reduce(t)
    assert torch.equal(result, t)


def test_strategy_reduce_boolean_decision_is_local():
    strategy = DistributedEnsembleStrategy()
    assert strategy.reduce_boolean_decision(True) is True
    assert strategy.reduce_boolean_decision(False) is False
    assert strategy.reduce_boolean_decision(True, all=False) is True


def test_strategy_remove_checkpoint_bypasses_rank_zero_gate(tmp_path):
    strategy = DistributedEnsembleStrategy()
    f = tmp_path / "dummy.ckpt"
    f.write_bytes(b"x")
    strategy.remove_checkpoint(f)
    assert not f.exists()


# ---------------------------------------------------------------------------
# End-to-end test: per-rank weights are preserved after reloading checkpoints
# ---------------------------------------------------------------------------


def _run_worker(mode, tmp_path, timeout=120):
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
        # Standalone: strip PMI vars so Cray MPICH does not abort on inherited closed fd
        for key in [k for k in env if k.startswith(("PMI_", "PMIX_"))]:
            del env[key]
        cmd = [sys.executable, str(worker_script), mode]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


def test_validate_preserves_per_rank_weights(tmp_path):
    """After reloading per-rank checkpoints, predictions must differ across ranks (CPU/gloo).

    Lightning manages process spawning (devices=2); no manual mp.spawn.
    """
    result = _run_worker("validate_weights", tmp_path)
    assert result.returncode == 0, (
        f"CPU worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (tmp_path / "PASSED").exists(), (
        f"Worker did not signal success.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# GPU/NCCL variant: subprocess with Lightning's native multi-GPU launcher
# ---------------------------------------------------------------------------


@pytest.mark.multigpu
def test_validate_preserves_per_rank_weights_gpu(tmp_path):
    """GPU/NCCL: after loading per-rank checkpoints, predictions must differ across ranks.

    Uses a standalone worker script launched as a subprocess so that Lightning's
    native DDP launcher handles all distributed setup (no manual env var hacking).
    """
    worker_script = Path(__file__).parent / "_workers_lightning_gpu.py"
    env = os.environ.copy()
    env["TEST_TMPDIR"] = str(tmp_path)
    # Strip PMI vars so Cray MPICH does not abort on the inherited closed fd
    for key in [k for k in env if k.startswith(("PMI_", "PMIX_"))]:
        del env[key]

    result = subprocess.run(
        [sys.executable, str(worker_script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"GPU worker failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (tmp_path / "PASSED").exists(), (
        f"Worker did not signal success.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
