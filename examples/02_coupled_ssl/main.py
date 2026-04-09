# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7.1",
#     "torch-distributed-ensemble",
#     "torchvision",
#     "torchmetrics",
#     "matplotlib",
# ]
# ///
"""Coupled semi-supervised ensemble consensus on CIFAR-10 (vanilla PyTorch).

Implements the idea from "Semi-Supervised Learning for Molecular Graphs via
Ensemble Consensus" — each ensemble member trains on a small labeled subset
(5k samples) with standard cross-entropy, plus a KL divergence term on
unlabeled data that pushes each member's predictions toward the ensemble
average (computed via all_gather, no gradient flow).

With M=1 member the KL loss is zero (self-consensus), reducing to supervised
baseline. With M>1 members, the consensus loss acts as a regularizer that
improves both individual and ensemble accuracy.

Usage:
    # Run both experiments sequentially, then plot:
    torchrun --nproc_per_node=N examples/02_coupled_ssl/main.py

    # Or submit two parallel jobs (faster):
    torchrun --nproc_per_node=N examples/02_coupled_ssl/main.py --mode uncoupled
    torchrun --nproc_per_node=N examples/02_coupled_ssl/main.py --mode coupled
    # After both finish:
    python examples/02_coupled_ssl/main.py --mode plot
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from torch_distributed_ensemble import gather_ensemble_metrics, setup_independent_worker

# ── Constants ────────────────────────────────────────────────────────────────

N_LABELED_PER_CLASS = 500  # 5k total labeled
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LAMBDA_KL = 0.5
DATA_SPLIT_SEED = 0  # fixed split identical across all ranks
NUM_CLASSES = 10

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = Path(__file__).resolve().parent / "outputs"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ── Data ─────────────────────────────────────────────────────────────────────


def make_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Pre-load CIFAR-10, split into labeled/unlabeled/test loaders."""
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_ds = CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=normalize)
    test_ds = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=normalize)

    # Pre-load into tensors (num_workers=0 to avoid memory pressure)
    def to_tensors(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
        return next(iter(loader))

    X_train, y_train = to_tensors(train_ds)
    X_test, y_test = to_tensors(test_ds)

    # Stratified split: N_LABELED_PER_CLASS per class → labeled, rest → unlabeled
    gen = torch.Generator().manual_seed(DATA_SPLIT_SEED)
    labeled_idx = []
    unlabeled_idx = []
    for c in range(NUM_CLASSES):
        class_idx = (y_train == c).nonzero(as_tuple=True)[0]
        perm = class_idx[torch.randperm(len(class_idx), generator=gen)]
        labeled_idx.append(perm[:N_LABELED_PER_CLASS])
        unlabeled_idx.append(perm[N_LABELED_PER_CLASS:])

    labeled_idx = torch.cat(labeled_idx)
    unlabeled_idx = torch.cat(unlabeled_idx)

    labeled_loader = DataLoader(
        TensorDataset(X_train[labeled_idx], y_train[labeled_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    # All ranks must see the same unlabeled batch each step so that
    # all_gather collects predictions for the same images (ensemble consensus).
    # Use a shared generator seeded with DATA_SPLIT_SEED; each rank creates
    # an identical generator so the shuffle order is identical across ranks.
    unlabeled_gen = torch.Generator().manual_seed(DATA_SPLIT_SEED)
    unlabeled_loader = DataLoader(
        TensorDataset(X_train[unlabeled_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=unlabeled_gen,
        num_workers=2,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    return labeled_loader, unlabeled_loader, test_loader


# ── Model ────────────────────────────────────────────────────────────────────


def make_resnet18_cifar() -> nn.Module:
    """ResNet18 adapted for CIFAR-10: 3x3 conv1, no maxpool."""
    from torchvision.models import resnet18

    model = resnet18(num_classes=NUM_CLASSES)
    # Replace 7x7 stride-2 conv with 3x3 stride-1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool
    model.maxpool = nn.Identity()
    return model


# ── Augmentation ─────────────────────────────────────────────────────────────

train_aug = v2.Compose(
    [
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
    ]
)


# ── Training ─────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_loader: DataLoader,
    unlabeled_iter,
    device: torch.device,
    lambda_kl: float = LAMBDA_KL,
) -> dict:
    """One epoch: one pass through labeled data, pulling from infinite unlabeled iterator."""
    model.train()
    total_ce, total_kl, n_steps = 0.0, 0.0, 0

    for x_l, y_l in labeled_loader:
        (x_u,) = next(unlabeled_iter)
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_u = x_u.to(device)

        # Labeled: cross-entropy
        ce_loss = F.cross_entropy(model(train_aug(x_l)), y_l)

        # Unlabeled: ensemble consensus via KL divergence
        log_p = F.log_softmax(model(train_aug(x_u)), dim=1)
        probs = log_p.exp().detach()  # detach before gathering
        all_probs = gather_ensemble_metrics(probs)  # (M, B, C), no grad
        ens_probs = all_probs.mean(dim=0)  # consensus target
        kl_loss = F.kl_div(log_p, ens_probs.detach(), reduction="batchmean")

        loss = ce_loss + lambda_kl * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        n_steps += 1

    return {"ce_loss": total_ce / n_steps, "kl_loss": total_kl / n_steps}


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict | None:
    """Evaluate individual and ensemble metrics. Returns dict on rank 0."""
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassCalibrationError,
    )

    model.eval()
    all_local_probs, all_targets = [], []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        all_local_probs.append(F.softmax(logits, dim=1))
        all_targets.append(y.to(device))

    local_probs = torch.cat(all_local_probs)  # (N, C)
    targets = torch.cat(all_targets)  # (N,)

    # Gather predictions from all members
    gathered = [torch.zeros_like(local_probs) for _ in range(world_size)]
    dist.all_gather(gathered, local_probs)  # ty: ignore[possibly-missing-attribute]
    all_probs = torch.stack(gathered)  # (M, N, C)

    if rank != 0:
        return None

    metrics_fn = MetricCollection(
        {
            "acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
            "ece": MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15),
        }
    ).to(device)

    # Ensemble metrics
    ens_probs = all_probs.mean(dim=0)
    ens_nll = F.nll_loss(torch.log(ens_probs + 1e-8), targets).item()
    ens_m = metrics_fn(ens_probs, targets)
    metrics_fn.reset()

    # Per-member metrics
    member_accs, member_nlls, member_eces = [], [], []
    for i in range(world_size):
        nll = F.nll_loss(torch.log(all_probs[i] + 1e-8), targets).item()
        m = metrics_fn(all_probs[i], targets)
        member_accs.append(m["acc"].item())
        member_nlls.append(nll)
        member_eces.append(m["ece"].item())
        metrics_fn.reset()

    return {
        "ens_acc": ens_m["acc"].item(),
        "ens_nll": ens_nll,
        "ens_ece": ens_m["ece"].item(),
        "member_accs": member_accs,
        "member_nlls": member_nlls,
        "member_eces": member_eces,
    }


# ── Plot ─────────────────────────────────────────────────────────────────────


def _save_plot(coupled_history: list[dict], uncoupled_history: list[dict], save_dir: Path):
    """Plot coupled vs uncoupled: accuracy curves and final metrics comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    world_size = len(coupled_history[0]["member_accs"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: ensemble accuracy curves, coupled vs uncoupled
    coupled_epochs = [h["epoch"] for h in coupled_history]
    uncoupled_epochs = [h["epoch"] for h in uncoupled_history]

    ax1.plot(uncoupled_epochs, [h["ens_acc"] for h in uncoupled_history], "C1-", lw=2, label="Uncoupled ensemble")
    for i in range(world_size):
        ax1.plot(
            uncoupled_epochs,
            [h["member_accs"][i] for h in uncoupled_history],
            color="C1",
            alpha=0.25,
            lw=0.8,
            label="Uncoupled members" if i == 0 else None,
        )

    ax1.plot(coupled_epochs, [h["ens_acc"] for h in coupled_history], "C0-", lw=2, label="Coupled ensemble")
    for i in range(world_size):
        ax1.plot(
            coupled_epochs,
            [h["member_accs"][i] for h in coupled_history],
            color="C0",
            alpha=0.25,
            lw=0.8,
            label="Coupled members" if i == 0 else None,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Accuracy over Training")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: final metrics bar chart, 4 bars per metric
    cf = coupled_history[-1]
    uf = uncoupled_history[-1]

    metrics = ["Accuracy", "NLL", "ECE"]
    uncoupled_member = [np.mean(uf["member_accs"]), np.mean(uf["member_nlls"]), np.mean(uf["member_eces"])]
    uncoupled_ens = [uf["ens_acc"], uf["ens_nll"], uf["ens_ece"]]
    coupled_member = [np.mean(cf["member_accs"]), np.mean(cf["member_nlls"]), np.mean(cf["member_eces"])]
    coupled_ens = [cf["ens_acc"], cf["ens_nll"], cf["ens_ece"]]

    x = np.arange(len(metrics))
    w = 0.2
    bars_um = ax2.bar(x - 1.5 * w, uncoupled_member, w, label="Uncoupled member", color="#d4a0a0")
    bars_ue = ax2.bar(x - 0.5 * w, uncoupled_ens, w, label="Uncoupled ensemble", color="#c44e52")
    bars_cm = ax2.bar(x + 0.5 * w, coupled_member, w, label="Coupled member", color="#7faadc")
    bars_ce = ax2.bar(x + 1.5 * w, coupled_ens, w, label="Coupled ensemble", color="#4c72b0")
    for bars in [bars_um, bars_ue, bars_cm, bars_ce]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_title("Final Test Metrics")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Coupled vs Uncoupled SSL — CIFAR-10 ResNet18 ({world_size} members)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = save_dir / "coupled_ssl.png"
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def run_experiment(
    rank: int,
    world_size: int,
    device: torch.device,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    test_loader: DataLoader,
    lambda_kl: float,
    label: str,
) -> list[dict]:
    """Train one ensemble run and return the history (rank 0 only)."""
    # Re-seed so each run starts from fresh random weights
    torch.manual_seed(42 + rank)
    model = make_resnet18_cifar().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    # Infinite unlabeled iterator: persists across epochs so it never "restarts"
    # at epoch boundaries and all ranks see the same batches (shared generator).
    unlabeled_iter = itertools.cycle(unlabeled_loader)

    for epoch in range(1, EPOCHS + 1):
        train_stats = train_one_epoch(model, optimizer, labeled_loader, unlabeled_iter, device, lambda_kl=lambda_kl)
        scheduler.step()

        metrics = evaluate(model, test_loader, device, rank, world_size)

        if rank == 0 and metrics is not None:
            metrics["epoch"] = epoch
            history.append(metrics)

            if epoch % 10 == 0 or epoch == 1:
                avg_acc = sum(metrics["member_accs"]) / len(metrics["member_accs"])
                print(
                    f"[{label} Epoch {epoch:3d}] "
                    f"CE: {train_stats['ce_loss']:.4f}  "
                    f"KL: {train_stats['kl_loss']:.4f}  "
                    f"LR: {scheduler.get_last_lr()[0]:.4f}  "
                    f"Avg Acc: {avg_acc:.4f}  "
                    f"Ens Acc: {metrics['ens_acc']:.4f}  "
                    f"Ens NLL: {metrics['ens_nll']:.4f}",
                    flush=True,
                )

    if rank == 0 and history:
        final = history[-1]
        print(f"\n===== {label} Final Results =====")
        for i, (acc, nll, ece) in enumerate(
            zip(final["member_accs"], final["member_nlls"], final["member_eces"], strict=True)
        ):
            print(f"  Member {i}  — Acc: {acc:.4f}  NLL: {nll:.4f}  ECE: {ece:.4f}")
        print(f"  Ensemble — Acc: {final['ens_acc']:.4f}  NLL: {final['ens_nll']:.4f}  ECE: {final['ens_ece']:.4f}")

    return history


def _load_history(path: Path) -> list[dict] | None:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def _save_history(history: list[dict], path: Path) -> None:
    with path.open("w") as f:
        json.dump(history, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "uncoupled", "coupled", "plot"],
        default="all",
        help=(
            "all: run both experiments then plot (default); "
            "uncoupled/coupled: run one experiment and save JSON; "
            "plot: load saved JSONs and regenerate figure (no torchrun needed)"
        ),
    )
    args, _ = parser.parse_known_args()  # ignore torchrun-injected args

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    uncoupled_path = LOG_DIR / "uncoupled_history.json"
    coupled_path = LOG_DIR / "coupled_history.json"

    # ── Plot-only mode (no distributed setup needed) ──────────────────────────
    if args.mode == "plot":
        uncoupled_history = _load_history(uncoupled_path)
        coupled_history = _load_history(coupled_path)
        if uncoupled_history is None or coupled_history is None:
            missing = [p for p, h in [(uncoupled_path, uncoupled_history), (coupled_path, coupled_history)] if h is None]
            raise FileNotFoundError(f"Missing result files: {missing}")
        _save_plot(coupled_history, uncoupled_history, LOG_DIR)
        return

    # ── Distributed training modes ────────────────────────────────────────────
    rank, world_size = setup_independent_worker(base_seed=42)
    device = torch.device(f"cuda:{rank}")
    labeled_loader, unlabeled_loader, test_loader = make_dataloaders()

    uncoupled_history: list[dict] = []
    coupled_history: list[dict] = []

    if args.mode in ("all", "uncoupled"):
        if rank == 0:
            print("=== Uncoupled (supervised only, lambda_kl=0) ===", flush=True)
        uncoupled_history = run_experiment(
            rank, world_size, device, labeled_loader, unlabeled_loader, test_loader, lambda_kl=0.0, label="Uncoupled"
        )
        if rank == 0 and uncoupled_history:
            _save_history(uncoupled_history, uncoupled_path)
            print(f"Uncoupled history saved to {uncoupled_path}", flush=True)

    if args.mode in ("all", "coupled"):
        if rank == 0:
            print("\n=== Coupled (lambda_kl=0.5) ===", flush=True)
        coupled_history = run_experiment(
            rank, world_size, device, labeled_loader, unlabeled_loader, test_loader, lambda_kl=LAMBDA_KL, label="Coupled"
        )
        if rank == 0 and coupled_history:
            _save_history(coupled_history, coupled_path)
            print(f"Coupled history saved to {coupled_path}", flush=True)

    if args.mode == "all" and rank == 0 and coupled_history and uncoupled_history:
        _save_plot(coupled_history, uncoupled_history, LOG_DIR)

    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


if __name__ == "__main__":
    main()
