"""CPU Lightning workers — launched by pytest via subprocess.

Lightning manages all process spawning (devices=2, gloo backend).
Each mode is selected by a positional CLI argument.

Two launch modes:
  Standalone (default): strips SLURM_*, PMI_*, PMIX_* env vars so that
      LightningEnvironment (not SLURMEnvironment) is used, and blocks mpi4py
      import so Cray MPICH does not try to use the inherited (closed) PMI socket.
  SLURM-native (--via-srun): caller must launch via `srun --ntasks=2`; Lightning
      detects SLURMEnvironment and reads SLURM_PROCID for rank assignment.

Usage:
    TEST_TMPDIR=/tmp/foo python tests/_workers_lightning_cpu.py <mode>
    TEST_TMPDIR=/tmp/foo srun --ntasks=2 python tests/_workers_lightning_cpu.py <mode> --via-srun

Modes: validate_weights, regression_metrics, best_checkpoint, best_checkpoint_rank_metric,
       per_rank_checkpoint, ensemble_metrics
"""

import os
import sys

_VIA_SRUN = "--via-srun" in sys.argv
if "--via-srun" in sys.argv:
    sys.argv.remove("--via-srun")

if not _VIA_SRUN:
    # Block mpi4py so Lightning's MPIEnvironment.detect() raises ImportError instead
    # of triggering Cray MPICH init (which aborts when PMI_FD points to a closed fd).
    sys.modules["mpi4py"] = None  # type: ignore[assignment]
    # Strip SLURM + PMI vars: PMI_FD points to a closed fd in subprocess (close_fds=True),
    # causing Cray MPICH to abort. Strip PMI_/PMIX_ → MPICH falls back to singleton mode.
    # Strip SLURM_ → LightningEnvironment is used, so devices=N is respected.
    for _k in [_k for _k in os.environ if _k.startswith(("SLURM_", "PMI_", "PMIX_"))]:
        del os.environ[_k]

import json
from pathlib import Path

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_distributed_ensemble.lightning.callbacks import (
    DistributedSeeder,
    EnsembleMetrics,
    PerRankBestCheckpoint,
    PerRankCheckpoint,
    RegressionEnsembleMetrics,
)
from torch_distributed_ensemble.lightning.strategy import DistributedEnsembleStrategy

WORLD_SIZE = 2


# ---------------------------------------------------------------------------
# Shared model definitions
# ---------------------------------------------------------------------------


class _TinyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _ClassificationModule(L.LightningModule):
    """Classification module returning softmax probs + targets for EnsembleMetrics."""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self.net(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return {"probs": self(x).detach(), "target": y}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _RegressionModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return {"preds": self(x).detach(), "target": y}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _cpu_trainer(callbacks, max_epochs=3):
    return L.Trainer(
        accelerator="cpu",
        devices=WORLD_SIZE,
        strategy=DistributedEnsembleStrategy(process_group_backend="gloo"),
        callbacks=callbacks,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        use_distributed_sampler=False,
    )


def _classification_loaders():
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    dl = DataLoader(TensorDataset(x, y), batch_size=16)
    return dl, dl


def _regression_loaders():
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = torch.randn(32)
    dl = DataLoader(TensorDataset(x, y), batch_size=16)
    return dl, dl


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def main_validate_weights():
    """Fit with DistributedSeeder, save per-rank checkpoints, reload and check predictions differ."""
    tmpdir = Path(os.environ["TEST_TMPDIR"])

    torch.manual_seed(0)
    x_train, y_train = torch.randn(32, 4), torch.randint(0, 2, (32,))
    x_val, y_val = torch.randn(16, 4), torch.randint(0, 2, (16,))
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

    # Phase 1: fit — DistributedSeeder gives each rank a different seed so weights diverge
    trainer = _cpu_trainer(callbacks=[DistributedSeeder(base_seed=42)], max_epochs=5)
    model = _TinyModule()
    trainer.fit(model, train_dl, val_dl)

    rank = trainer.global_rank
    world_size = trainer.world_size
    ckpt_path = tmpdir / f"member_{rank}.pt"
    torch.save(model.state_dict(), ckpt_path)
    trainer.strategy.barrier()

    # Phase 2: reload per-rank checkpoint and evaluate within the active distributed context
    device = trainer.strategy.root_device
    fresh_model = _TinyModule().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    fresh_model.load_state_dict(state)
    fresh_model.eval()

    with torch.no_grad():
        local_logits = torch.cat([fresh_model(batch[0].to(device)) for batch in val_dl])

    gathered = [torch.zeros_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered, local_logits)  # ty: ignore[possibly-missing-attribute]
    all_logits = torch.stack(gathered)  # (world_size, N, 2)

    if trainer.is_global_zero:
        assert all_logits.shape[0] == 2, f"Expected 2 ranks, got {all_logits.shape[0]}"
        assert not torch.allclose(all_logits[0], all_logits[1]), (
            "CPU: predictions are identical across ranks — weights were synchronised"
        )
        (tmpdir / "PASSED").touch()


def main_regression_metrics():
    """Fit with RegressionEnsembleMetrics; each rank writes its callback_metrics to a JSON file."""
    tmpdir = Path(os.environ["TEST_TMPDIR"])
    train_dl, val_dl = _regression_loaders()

    trainer = _cpu_trainer(callbacks=[RegressionEnsembleMetrics()], max_epochs=3)
    trainer.fit(_RegressionModule(), train_dl, val_dl)

    rank = trainer.global_rank
    metrics = {k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
    (tmpdir / f"metrics_rank{rank}.json").write_text(json.dumps(metrics))
    trainer.strategy.barrier()

    if trainer.is_global_zero:
        (tmpdir / "PASSED").touch()


def main_best_checkpoint():
    """Fit with PerRankBestCheckpoint; checkpoints saved to TEST_TMPDIR/best/."""
    tmpdir = Path(os.environ["TEST_TMPDIR"])
    train_dl, val_dl = _regression_loaders()

    dirpath = tmpdir / "best"
    cb = PerRankBestCheckpoint(monitor="val/ens_mse", dirpath=dirpath, mode="min")
    trainer = _cpu_trainer(callbacks=[RegressionEnsembleMetrics(), cb], max_epochs=5)
    trainer.fit(_RegressionModule(), train_dl, val_dl)

    if trainer.is_global_zero:
        (tmpdir / "PASSED").touch()


def main_best_checkpoint_rank_metric():
    """Fit with PerRankBestCheckpoint using {rank} substitution in monitor."""
    tmpdir = Path(os.environ["TEST_TMPDIR"])
    train_dl, val_dl = _regression_loaders()

    dirpath = tmpdir / "member_best"
    cb = PerRankBestCheckpoint(monitor="val/member_{rank}_mse", dirpath=dirpath, mode="min")
    trainer = _cpu_trainer(callbacks=[RegressionEnsembleMetrics(), cb], max_epochs=5)
    trainer.fit(_RegressionModule(), train_dl, val_dl)

    if trainer.is_global_zero:
        (tmpdir / "PASSED").touch()


def main_per_rank_checkpoint():
    """Fit with PerRankCheckpoint; final-epoch checkpoint saved per rank."""
    tmpdir = Path(os.environ["TEST_TMPDIR"])
    train_dl, val_dl = _classification_loaders()

    dirpath = tmpdir / "checkpoints"
    cb = PerRankCheckpoint(dirpath=str(dirpath))
    trainer = _cpu_trainer(callbacks=[DistributedSeeder(base_seed=42), cb], max_epochs=3)
    trainer.fit(_TinyModule(), train_dl, val_dl)

    if trainer.is_global_zero:
        (tmpdir / "PASSED").touch()


def main_ensemble_metrics():
    """Fit with EnsembleMetrics; each rank writes its callback_metrics to a JSON file."""
    tmpdir = Path(os.environ["TEST_TMPDIR"])
    train_dl, val_dl = _classification_loaders()

    trainer = _cpu_trainer(callbacks=[EnsembleMetrics()], max_epochs=3)
    trainer.fit(_ClassificationModule(), train_dl, val_dl)

    rank = trainer.global_rank
    metrics = {k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
    (tmpdir / f"metrics_rank{rank}.json").write_text(json.dumps(metrics))
    trainer.strategy.barrier()

    if trainer.is_global_zero:
        (tmpdir / "PASSED").touch()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_MODES = {
    "validate_weights": main_validate_weights,
    "regression_metrics": main_regression_metrics,
    "best_checkpoint": main_best_checkpoint,
    "best_checkpoint_rank_metric": main_best_checkpoint_rank_metric,
    "per_rank_checkpoint": main_per_rank_checkpoint,
    "ensemble_metrics": main_ensemble_metrics,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _MODES:
        print(f"Usage: {sys.argv[0]} <{'|'.join(_MODES)}>\n", file=sys.stderr)
        sys.exit(1)
    _MODES[sys.argv[1]]()
