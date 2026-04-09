# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7.1",
#     "torch-distributed-ensemble[lightning]",
#     "hydra-core",
#     "hydra-optuna-sweeper",
# ]
# ///
"""HPO for a tabular regression ensemble using Hydra + Optuna.

Demonstrates ensemble-specific hyperparameter optimization on UCI wine-quality
regression. The configuration is preset to the best-found config where ensemble-
tuned HPO strongly diverges from member-tuned HPO:

**Config:** tanh activation, no scheduler, no dropout, 400 epochs, lr=3e-3.
- Ensemble-optimal weight decay: 1e-5  → individual members overfit (mem MSE 0.74)
  but ensemble averages out errors (ens MSE 0.556) — pure diversity exploitation
- Member-optimal weight decay: 0.534  (53,367x higher) → controls each member
  individually (mem MSE 0.633), but the ensemble gains nothing from diversity (ens MSE 0.634)
- **Result:** Ensemble MSE 0.556 vs 0.634 (+12.1% gain)

This is the strongest form of the phenomenon: with very low regularization,
members fit to different local optima and their errors are largely uncorrelated.
Ensemble averaging dramatically reduces variance. Member-tuned HPO cannot
afford this — it must regularize each member individually.

Per-trial results are appended to
``examples/03_hpo_ensemble/outputs/results_{dataset}_{n_train}_{objective}[_{N}seeds].jsonl``.
Each trial trains over multiple data splits (``training.data_seeds``) and reports the average
objective across splits, with per-seed breakdowns stored in the JSONL for post-analysis.

Usage:
    # Single run (smoke test):
    uv run python examples/03_hpo_ensemble/train.py

    # Full HPO sweep (ensemble objective):
    uv run python examples/03_hpo_ensemble/train.py --multirun objective=ensemble

    # Full HPO sweep (member objective):
    uv run python examples/03_hpo_ensemble/train.py --multirun objective=member
"""

import json
import urllib.request
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from torch_distributed_ensemble import (
    DistributedEnsembleStrategy,
    DistributedSeeder,
)
from torch_distributed_ensemble.functional import gather_ensemble_metrics

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "uci"
LOG_DIR = Path(__file__).resolve().parent / "outputs"

UCI_SOURCES = {
    "yacht": "yacht",
    "boston": "bostonHousing",
    "concrete": "concrete",
    "energy": "energy",
    "wine": "wine-quality-red",
    "kin8nm": "kin8nm",
    "naval": "naval-propulsion-plant",
    "power": "power-plant",
}
BASE_URL = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets"


# ── Data ─────────────────────────────────────────────────────────────────────


def load_uci(name: str):
    """Download and cache a UCI regression dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{name}.pt"
    if not cache.exists():
        github_dir = UCI_SOURCES[name]
        url = f"{BASE_URL}/{github_dir}/data/data.txt"
        print(f"Downloading {name} from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "python/3"})
        with urllib.request.urlopen(req) as resp:
            data = np.loadtxt(resp.read().decode().splitlines())
        X = torch.tensor(data[:, :-1], dtype=torch.float32)
        y = torch.tensor(data[:, -1], dtype=torch.float32)
        torch.save({"X": X, "y": y}, cache)
        print(f"Cached {name}: {X.shape[0]} samples, {X.shape[1]} features")
    d = torch.load(cache, weights_only=False)
    return d["X"], d["y"]


def make_dataloaders(cfg: DictConfig, data_seed: int = 0) -> tuple[DataLoader, DataLoader]:
    X, y = load_uci(cfg.data.dataset)
    # Deterministic shuffle + split
    gen = torch.Generator().manual_seed(data_seed)
    perm = torch.randperm(len(X), generator=gen)
    X, y = X[perm], y[perm]

    n_train = cfg.data.get("n_train")
    if n_train is None:
        n_train = int(0.8 * len(X))
    n_train = min(n_train, len(X) - 50)

    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    # Normalize
    mu_x, std_x = X_tr.mean(0), X_tr.std(0).clamp(min=1e-8)
    X_tr, X_val = (X_tr - mu_x) / std_x, (X_val - mu_x) / std_x
    mu_y, std_y = y_tr.mean(), y_tr.std().clamp(min=1e-8)
    y_tr, y_val = (y_tr - mu_y) / std_y, (y_val - mu_y) / std_y

    bs = cfg.data.batch_size
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)
    return train_loader, val_loader


# ── Model ────────────────────────────────────────────────────────────────────


class TabularMLP(L.LightningModule):
    def __init__(self, cfg: DictConfig, n_features: int):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Select activation function
        act_name = cfg.model.get("activation", "relu").lower()
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        ActClass = activations.get(act_name, nn.ReLU)

        layers: list[nn.Module] = []
        in_d = n_features
        h = cfg.model.hidden_size
        for _ in range(cfg.model.num_layers):
            layers.extend([nn.Linear(in_d, h), ActClass()])
            if cfg.model.dropout > 0:
                layers.append(nn.Dropout(p=cfg.model.dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        return {"preds": preds, "target": y}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        # Optional cosine annealing scheduler
        if self.cfg.training.get("use_cosine_scheduler", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.max_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        return optimizer


# ── Callbacks ────────────────────────────────────────────────────────────────


class RegressionEnsembleMetrics(Callback):
    def __init__(self):
        self._outputs: list[dict] = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.cat([o["preds"] for o in self._outputs])
        targets = torch.cat([o["target"] for o in self._outputs])
        self._outputs.clear()

        member_mse = nn.functional.mse_loss(preds, targets)
        pl_module.log("val/member_mse", member_mse, rank_zero_only=False, sync_dist=False)

        all_preds = gather_ensemble_metrics(preds)
        M = all_preds.shape[0]
        ens_mean = all_preds.mean(dim=0)
        ens_mse = nn.functional.mse_loss(ens_mean, targets)
        ens_std = all_preds.std(dim=0).mean() if M > 1 else torch.tensor(float("nan"))

        for m in range(M):
            m_mse = nn.functional.mse_loss(all_preds[m], targets)
            pl_module.log(f"val/member_{m}_mse", m_mse, rank_zero_only=False, sync_dist=False)

        pl_module.log("val/ens_mse", ens_mse, rank_zero_only=False, sync_dist=False)
        pl_module.log("val/ens_std", ens_std, rank_zero_only=False, sync_dist=False)

        if trainer.global_rank == 0:
            print(
                f"\n[Epoch {trainer.current_epoch}] ens_mse={ens_mse:.4f} "
                f"member_mse={member_mse:.4f} gain={member_mse - ens_mse:+.4f}"
            )


class AvgMemberMetrics(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        mse_vals = []
        for k, v in metrics.items():
            if k.startswith("val/member_") and k.endswith("_mse") and k != "val/member_mse":
                mse_vals.append(v)
        if mse_vals:
            pl_module.log("val/avg_member_mse", torch.stack(mse_vals).mean(), rank_zero_only=False, sync_dist=False)


# ── Main ─────────────────────────────────────────────────────────────────────


def run_single_seed(cfg: DictConfig, num_members: int, data_seed: int) -> dict:
    """Train one ensemble on a single data split and return metrics."""
    train_loader, val_loader = make_dataloaders(cfg, data_seed=data_seed)
    n_features = train_loader.dataset.tensors[0].shape[1]  # ty: ignore[unresolved-attribute]

    trainer = L.Trainer(
        accelerator="gpu",
        devices=num_members,
        strategy=DistributedEnsembleStrategy(),
        plugins=[LightningEnvironment()],  # single-task job; spawns GPU workers internally
        callbacks=[
            DistributedSeeder(base_seed=cfg.training.base_seed),
            RegressionEnsembleMetrics(),
            AvgMemberMetrics(),
        ],
        max_epochs=cfg.training.max_epochs,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )

    model = TabularMLP(cfg, n_features)
    trainer.fit(model, train_loader, val_loader)

    inf = torch.tensor(float("inf"))
    cb = trainer.callback_metrics
    return {
        "data_seed": data_seed,
        "ens_mse": cb.get("val/ens_mse", inf).item(),
        "avg_member_mse": cb.get("val/avg_member_mse", inf).item(),
        "ens_std": cb.get("val/ens_std", inf).item(),
        "_trainer": trainer,
    }


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("This example requires at least 1 GPU.")

    num_members = min(cfg.ensemble.num_members, num_gpus)
    data_seeds = list(cfg.training.get("data_seeds", [0]))

    seed_results = []
    last_trainer = None
    for data_seed in data_seeds:
        metrics = run_single_seed(cfg, num_members, data_seed)
        last_trainer = metrics.pop("_trainer")
        seed_results.append(metrics)
        print(f"[seed={data_seed}] ens_mse={metrics['ens_mse']:.4f} avg_member_mse={metrics['avg_member_mse']:.4f}")

    # Aggregate across seeds
    ens_mse_mean = float(np.mean([r["ens_mse"] for r in seed_results]))
    avg_member_mse_mean = float(np.mean([r["avg_member_mse"] for r in seed_results]))
    ens_std_mean = float(np.mean([r["ens_std"] for r in seed_results]))

    if cfg.objective == "ensemble":
        obj_val = ens_mse_mean
    else:
        obj_val = avg_member_mse_mean

    # Write JSONL (rank 0 only)
    if last_trainer is not None and last_trainer.is_global_zero:
        n_seeds = len(data_seeds)
        suffix = f"_{n_seeds}seeds" if n_seeds > 1 else ""
        result = {
            "ens_mse": ens_mse_mean,
            "avg_member_mse": avg_member_mse_mean,
            "ens_std": ens_std_mean,
            "seeds": seed_results,
            "params": {"weight_decay": cfg.training.weight_decay},
        }
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ds = cfg.data.dataset
        n_tr = cfg.data.get("n_train") or "all"
        jsonl_path = LOG_DIR / f"results_{ds}_{n_tr}_{cfg.objective}{suffix}.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    return obj_val


if __name__ == "__main__":
    main()
