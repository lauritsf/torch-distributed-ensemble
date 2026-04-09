"""GPU Lightning worker — launched by pytest via subprocess.

Lightning handles all distributed setup (no manual env var hacking).
Each process re-runs this script; Lightning coordinates them.

Usage (called by test_lightning_strategy.py):
    TEST_TMPDIR=/tmp/foo python tests/_workers_lightning_gpu.py
"""

import os
import sys
from pathlib import Path

# Block mpi4py and strip PMI_*/PMIX_* env vars so Cray MPICH does not abort
# on the inherited (closed) PMI socket fd when running as a subprocess.
sys.modules["mpi4py"] = None  # type: ignore[assignment]
for _k in [_k for _k in os.environ if _k.startswith(("SLURM_", "PMI_", "PMIX_"))]:
    del os.environ[_k]

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_distributed_ensemble.lightning.callbacks import DistributedSeeder
from torch_distributed_ensemble.lightning.strategy import DistributedEnsembleStrategy


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


def main():
    tmpdir = os.environ["TEST_TMPDIR"]
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, f"Need >=2 GPUs, got {num_gpus}"

    # Deterministic data (same on all ranks)
    torch.manual_seed(0)
    x_train = torch.randn(32, 4)
    y_train = torch.randint(0, 2, (32,))
    x_val = torch.randn(16, 4)
    y_val = torch.randint(0, 2, (16,))

    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

    # Phase 1: fit — weights diverge via DistributedSeeder
    trainer = L.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=DistributedEnsembleStrategy(),
        callbacks=[DistributedSeeder(base_seed=42)],
        max_epochs=5,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        use_distributed_sampler=False,
    )
    model = _TinyModule()
    trainer.fit(model, train_dl, val_dl)

    # Save per-rank checkpoint
    rank = trainer.global_rank
    world_size = trainer.world_size
    ckpt_path = Path(tmpdir) / f"member_{rank}.pt"
    torch.save(model.state_dict(), ckpt_path)
    trainer.strategy.barrier()

    # Phase 2: load per-rank checkpoint into fresh model, evaluate manually
    # (like the 01b example — reuse the still-active distributed context)
    device = trainer.strategy.root_device
    fresh_model = _TinyModule().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    fresh_model.load_state_dict(state)
    fresh_model.eval()

    with torch.no_grad():
        local_logits = []
        for batch in val_dl:
            x = batch[0].to(device)
            local_logits.append(fresh_model(x))
        local_logits = torch.cat(local_logits)  # (N, 2)

    # All-gather predictions across ranks
    gathered = [torch.zeros_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered, local_logits)  # ty: ignore[possibly-missing-attribute]
    all_logits = torch.stack(gathered)  # (world_size, N, 2)

    if trainer.is_global_zero:
        assert all_logits.shape[0] == 2, f"Expected 2 ranks, got {all_logits.shape[0]}"
        assert not torch.allclose(all_logits[0], all_logits[1]), (
            "GPU: Validation predictions are identical across ranks — "
            "weights were synchronized when they should be independent"
        )
        # Signal success to pytest
        (Path(tmpdir) / "PASSED").touch()


if __name__ == "__main__":
    main()
