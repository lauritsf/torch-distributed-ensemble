# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7.1",
#     "torch-distributed-ensemble[lightning]",
#     "matplotlib",
#     "torchvision",
# ]
# ///
"""Joint early stopping on CIFAR-10 with a big MLP (Lightning).

Trains a deep ensemble where each GPU trains its own MLP classifier on CIFAR-10.
A wide MLP without convolutions will memorize the training set, producing clear
overfitting curves — individual members' val NLL rises while the ensemble's
remains more stable. This makes the case for joint early stopping on ensemble
NLL rather than per-member metrics.

After training, each rank loads its saved checkpoint and evaluates on the test
set in a distributed fashion — predictions are all-gathered to form the ensemble.

Usage:
    uv run python examples/01_joint_early_stopping/main.py
"""

from pathlib import Path

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError
from torchvision import transforms
from torchvision.datasets import CIFAR10

from torch_distributed_ensemble import (
    DistributedEnsembleStrategy,
    DistributedSeeder,
    EnsembleMetrics,
    PerRankBestCheckpoint,
)

# ── Constants ────────────────────────────────────────────────────────────────

LR = 1e-3
BATCH_SIZE = 2048
MAX_EPOCHS = 200
PATIENCE = 20
HIDDEN_SIZE = 2048
NUM_LAYERS = 4
NUM_CLASSES = 10
NUM_FEATURES = 3 * 32 * 32  # 3072
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = Path(__file__).resolve().parent / "outputs"


# ── Data ─────────────────────────────────────────────────────────────────────


def make_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Pre-load CIFAR-10 into tensors, return (train, val, test) DataLoaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten to 3072
        ]
    )

    train_ds = CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_ds = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)

    # Load entire datasets into tensors (one-time cost).
    # Use num_workers=0 to avoid memory pressure when many ranks do this in parallel.
    def to_tensors(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
        return next(iter(loader))

    X_train, y_train = to_tensors(train_ds)
    X_test, y_test = to_tensors(test_ds)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ── Model ────────────────────────────────────────────────────────────────────


class CIFAR10MLP(L.LightningModule):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = NUM_FEATURES
        for _ in range(NUM_LAYERS):
            layers += [nn.Linear(in_dim, HIDDEN_SIZE), nn.ReLU()]
            in_dim = HIDDEN_SIZE
        layers.append(nn.Linear(HIDDEN_SIZE, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return {"probs": probs, "target": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


# ── Distributed Evaluation ──────────────────────────────────────────────────


@torch.no_grad()
def distributed_evaluate(
    strategy: str, ckpt_dir: Path, test_loader: DataLoader, rank: int, world_size: int, device: torch.device
) -> dict | None:
    """Each rank loads its own checkpoint, runs inference, then all-gathers predictions.

    Returns results dict on rank 0, None on other ranks.
    """
    ckpt_path = ckpt_dir / strategy / f"best-rank={rank}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = CIFAR10MLP().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    local_probs_list = []
    local_targets_list = []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        local_probs_list.append(F.softmax(logits, dim=1))
        local_targets_list.append(y.to(device))

    local_probs = torch.cat(local_probs_list)  # (N, C)
    targets = torch.cat(local_targets_list)  # (N,)

    gathered = [torch.zeros_like(local_probs) for _ in range(world_size)]
    dist.all_gather(gathered, local_probs)  # ty: ignore[possibly-missing-attribute]
    all_probs = torch.stack(gathered)  # (M, N, C)

    epoch_t = torch.tensor([ckpt["epoch"]], device=device)
    all_epochs_t = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_epochs_t, epoch_t)  # ty: ignore[possibly-missing-attribute]

    if rank != 0:
        return None

    metrics_fn = MetricCollection(
        {
            "acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
            "ece": MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15),
        }
    ).to(device)

    ens_probs = all_probs.mean(dim=0)
    ens_nll = F.nll_loss(torch.log(ens_probs + 1e-8), targets).item()
    ens_m = metrics_fn(ens_probs, targets)
    metrics_fn.reset()

    member_nlls, member_accs, member_eces = [], [], []
    for i in range(world_size):
        member_nlls.append(F.nll_loss(torch.log(all_probs[i] + 1e-8), targets).item())
        m = metrics_fn(all_probs[i], targets)
        member_accs.append(m["acc"].item())
        member_eces.append(m["ece"].item())
        metrics_fn.reset()

    return {
        "epochs": [int(e.item()) for e in all_epochs_t],
        "ens_nll": ens_nll,
        "ens_acc": ens_m["acc"].item(),
        "ens_ece": ens_m["ece"].item(),
        "member_nlls": member_nlls,
        "member_accs": member_accs,
        "member_eces": member_eces,
    }


def _print_test_results(results: dict):
    """Print a comparison table of test-set results."""
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (distributed)")
    print("=" * 70)

    for strategy, label in [("member_best", "Member-best checkpoints"), ("ens_best", "Ensemble-best checkpoints")]:
        res = results[strategy]
        print(f"\n{label} (epochs: {res['epochs']}):")
        print(f"  Ensemble  — NLL: {res['ens_nll']:.4f}  Acc: {res['ens_acc']:.4f}  ECE: {res['ens_ece']:.4f}")
        for i, (nll, acc, ece) in enumerate(
            zip(res["member_nlls"], res["member_accs"], res["member_eces"], strict=True)
        ):
            print(f"  Member {i}  — NLL: {nll:.4f}  Acc: {acc:.4f}  ECE: {ece:.4f}")

    mb = results["member_best"]
    eb = results["ens_best"]
    print(f"\nEnsemble NLL:  member-best={mb['ens_nll']:.4f}  ens-best={eb['ens_nll']:.4f}")
    print(f"Ensemble Acc:  member-best={mb['ens_acc']:.4f}  ens-best={eb['ens_acc']:.4f}")
    print(f"Ensemble ECE:  member-best={mb['ens_ece']:.4f}  ens-best={eb['ens_ece']:.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────────


def _save_plot(csv_path: Path, save_dir: Path, test_results: dict):
    """Plot training curves (NLL, Acc, ECE) and test-set checkpoint comparison."""
    import csv as csvmod

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(csv_path) as f:
        reader = csvmod.DictReader(f)
        rows = list(reader)

    rows = [r for r in rows if r.get("val/ens_nll")]
    if not rows:
        print("No validation metrics found in CSV — skipping plot.")
        return

    epochs = [int(r["epoch"]) for r in rows]
    member_nll_keys = sorted(k for k in rows[0] if k.startswith("val/member_") and k.endswith("_nll"))
    num_members = len(member_nll_keys)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    # Extract all metrics from CSV
    def extract(metric_suffix):
        ens = [float(r[f"val/ens_{metric_suffix}"]) for r in rows]
        members = {}
        for key in member_nll_keys:
            idx = key.replace("val/member_", "").replace("_nll", "")
            col = f"val/member_{idx}_{metric_suffix}"
            members[f"member_{idx}"] = [float(r[col]) for r in rows]
        return ens, members

    ens_nll, member_nlls = extract("nll")
    ens_acc, member_accs = extract("acc")
    ens_ece, member_eces = extract("ece")

    ens_best_idx = int(np.argmin(ens_nll))

    # ── Figure layout: 2 rows x 3 cols ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ── Top row: training curves ────────────────────────────────────────────
    for col, (ens_vals, member_vals, ylabel, title) in enumerate(
        [
            (ens_nll, member_nlls, "NLL", "Validation NLL"),
            (ens_acc, member_accs, "Accuracy", "Validation Accuracy"),
            (ens_ece, member_eces, "ECE", "Validation ECE"),
        ]
    ):
        ax = axes[0, col]
        minimize = ylabel in ("NLL", "ECE")
        member_best_idxs = {
            name: int(np.argmin(vals) if minimize else np.argmax(vals)) for name, vals in member_vals.items()
        }

        for i, (name, vals) in enumerate(member_vals.items()):
            c = colors[i % len(colors)]
            ax.plot(epochs, vals, alpha=0.4, lw=1, color=c, label=name)
            best = member_best_idxs[name]
            ax.plot(epochs[best], vals[best], "v", color=c, ms=6, zorder=5)

        ax.plot(epochs, ens_vals, "k-", lw=2.5, label="ensemble")
        ax.axvline(epochs[ens_best_idx], color="k", ls="--", lw=1, alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6, loc="upper left")

    # ── Bottom row: test-set bar charts (one per metric) ────────────────────
    mb = test_results["member_best"]
    eb = test_results["ens_best"]

    bar_labels = [
        "Avg member\n(member-best)",
        "Ensemble\n(member-best)",
        "Avg member\n(ens-best)",
        "Ensemble\n(ens-best)",
    ]
    bar_colors = ["#7faadc", "#4c72b0", "#7fdc8f", "#55a868"]
    x = np.arange(len(bar_labels))

    for col, (metric_name, ylabel, vals) in enumerate(
        [
            ("NLL", "NLL", [np.mean(mb["member_nlls"]), mb["ens_nll"], np.mean(eb["member_nlls"]), eb["ens_nll"]]),
            (
                "Accuracy",
                "Accuracy",
                [np.mean(mb["member_accs"]), mb["ens_acc"], np.mean(eb["member_accs"]), eb["ens_acc"]],
            ),
            ("ECE", "ECE", [np.mean(mb["member_eces"]), mb["ens_ece"], np.mean(eb["member_eces"]), eb["ens_ece"]]),
        ]
    ):
        ax = axes[1, col]
        bars = ax.bar(x, vals, color=bar_colors)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Test {metric_name}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Joint Early Stopping — CIFAR-10 MLP ({num_members} members)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = save_dir / "joint_early_stopping_cifar.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("This example requires at least 1 GPU.")

    train_loader, val_loader, test_loader = make_dataloaders()

    metrics = MetricCollection(
        {
            "acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
            "ece": MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15),
        }
    )

    ckpt_dir = LOG_DIR / "checkpoints"
    logger = CSVLogger(save_dir=str(LOG_DIR), name="", version="")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy=DistributedEnsembleStrategy(),
        callbacks=[
            DistributedSeeder(base_seed=42),
            EnsembleMetrics(metrics=metrics),
            PerRankBestCheckpoint(monitor="val/ens_nll", dirpath=ckpt_dir / "ens_best"),
            PerRankBestCheckpoint(monitor="val/member_{rank}_nll", dirpath=ckpt_dir / "member_best"),
            EarlyStopping(monitor="val/ens_nll", patience=PATIENCE, mode="min"),
        ],
        max_epochs=MAX_EPOCHS,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=50,
    )

    model = CIFAR10MLP()
    trainer.fit(model, train_loader, val_loader)

    # ── Distributed evaluation ──────────────────────────────────────────────
    # Each rank loads its own checkpoint, runs inference, then all-gathers.
    # The distributed context is still active after trainer.fit().
    rank = trainer.global_rank
    device = torch.device(f"cuda:{rank}")

    test_results = {}
    for strategy in ["ens_best", "member_best"]:
        result = distributed_evaluate(strategy, ckpt_dir, test_loader, rank, num_gpus, device)
        if result is not None:
            test_results[strategy] = result

    if trainer.is_global_zero:
        csv_path = Path(logger.log_dir) / "metrics.csv"
        print(f"\nMetrics CSV: {csv_path}")
        _print_test_results(test_results)
        _save_plot(csv_path, LOG_DIR, test_results)


if __name__ == "__main__":
    main()
