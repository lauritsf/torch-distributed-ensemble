# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7.1",
#     "torch-distributed-ensemble",
#     "matplotlib",
#     "rdkit",
#     "pillow",
# ]
# ///
"""Active learning for HTS hit finding with deep ensemble surrogates.

Searches for active compounds in a pool of ~101k molecules from the Cav3
T-type Calcium Channel HTS assay (Butkiewicz et al., 2013). Only 0.7% of
molecules are active — a needle-in-a-haystack problem where random screening
wastes most oracle calls on inactives.

Each torchrun worker trains an independent binary classifier on 1024-bit
Morgan fingerprints. Predictions are gathered via all_gather to compute
ensemble mean P(active) and disagreement σ. UCB acquisition (μ + κ·σ)
balances exploitation (high predicted activity) with exploration (high
ensemble disagreement), finding active compounds far faster than random.

Dataset: Cav3 T-type Calcium Channels (Butkiewicz et al., 2013) via TDC.
~101k compounds, 703 actives (0.7% hit rate).
Source: https://doi.org/10.1177/1087057113510740

Usage:
    torchrun --nproc_per_node=N examples/04_hts_hit_finding/main.py
"""

import csv
import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from torch_distributed_ensemble import gather_ensemble_metrics, setup_independent_worker

# --- Hyperparameters ---
N_INITIAL = 100  # initial labeled pool size
BO_ITERS = 100  # number of active learning iterations
BATCH_K = 20  # molecules selected per iteration
TRAIN_EPOCHS = 30  # epochs per iteration
TRAIN_LR = 1e-3
TRAIN_BATCH = 256
FP_BITS = 1024  # Morgan fingerprint length
UCB_KAPPA = 2.0  # exploration weight in UCB = μ + κ·σ

DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/6894445"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TSV_PATH = DATA_DIR / "cav3_butkiewicz.tsv"
CACHE_PATH = DATA_DIR / "cav3_morgan1024.pt"
LOG_DIR = Path(__file__).resolve().parent / "outputs"


# --- Per-iteration snapshot for visualization ---
@dataclass
class IterSnapshot:
    mu: torch.Tensor  # (pool_size,) ensemble mean P(active)
    sigma: torch.Tensor  # (pool_size,) ensemble std
    ucb: torch.Tensor | None  # (pool_size,) UCB values (None for random)
    pool_idx: torch.Tensor  # (pool_size,) global indices of pool molecules
    selected_global: torch.Tensor  # (BATCH_K,) global indices selected
    hits_so_far: int  # cumulative hits found
    n_labeled: int


@dataclass
class BOResult:
    hits_history: list[int] = field(default_factory=list)
    snapshots: list[IterSnapshot] = field(default_factory=list)
    initial_labeled: torch.Tensor | None = None


def download_and_featurize(rank):
    """Rank 0: download Cav3 TSV, compute Morgan fingerprints, save cache."""
    if rank == 0:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not CACHE_PATH.exists():
            if not TSV_PATH.exists():
                print(f"Downloading Cav3 HTS data to {TSV_PATH} ...")
                req = urllib.request.Request(DATA_URL, headers={"User-Agent": "python-requests/2.31"})
                with urllib.request.urlopen(req) as resp, open(TSV_PATH, "wb") as f:
                    f.write(resp.read())
                print("Done.")

            print("Computing Morgan fingerprints ...")
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator

            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
            fps, targets = [], []

            with open(TSV_PATH) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    smiles = row["Drug"].strip('"')
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    fp = gen.GetFingerprintAsNumPy(mol)
                    fps.append(torch.from_numpy(fp).float())
                    targets.append(int(row["Y"]))

            X = torch.stack(fps)
            y = torch.tensor(targets, dtype=torch.float32)
            torch.save({"X": X, "y": y}, CACHE_PATH)
            n_active = int(y.sum().item())
            print(f"Cached {len(y)} molecules ({n_active} active, {n_active / len(y) * 100:.1f}%) to {CACHE_PATH}")

    dist.barrier()  # ty: ignore[possibly-missing-attribute]

    data = torch.load(CACHE_PATH, weights_only=False)
    return data["X"], data["y"]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FP_BITS, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def train_model(model, X, y):
    """Train binary classifier with class-weighted BCE."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)

    # Class weighting to handle extreme imbalance
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=X.device)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True)

    for _ in range(TRAIN_EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_pool(model, X_pool):
    """Predict P(active) on pool."""
    model.eval()
    preds = []
    for i in range(0, len(X_pool), 2048):
        logits = model(X_pool[i : i + 2048])
        preds.append(torch.sigmoid(logits))
    return torch.cat(preds)


def run_al(rank, world_size, X_all, y_all, device, use_ucb=True):
    """Run active learning loop. Returns BOResult with hit discovery history."""
    n = len(y_all)
    n_active_total = int(y_all.sum().item())
    result = BOResult()

    # Stratified initial selection: guarantee some actives in seed set
    torch.manual_seed(0)
    active_idx = y_all.nonzero(as_tuple=True)[0]
    inactive_idx = (y_all == 0).nonzero(as_tuple=True)[0]
    n_init_active = max(5, int(N_INITIAL * n_active_total / n))  # ~proportional, min 5
    n_init_inactive = N_INITIAL - n_init_active

    chosen_active = active_idx[torch.randperm(len(active_idx))[:n_init_active]]
    chosen_inactive = inactive_idx[torch.randperm(len(inactive_idx))[:n_init_inactive]]
    initial = torch.cat([chosen_active, chosen_inactive])

    labeled_mask = torch.zeros(n, dtype=torch.bool)
    labeled_mask[initial] = True
    result.initial_labeled = initial.clone()

    hits_so_far = int(y_all[labeled_mask].sum().item())
    result.hits_history.append(hits_so_far)

    for it in range(BO_ITERS):
        torch.manual_seed(42 + rank + it * world_size)
        model = MLP().to(device)

        X_labeled = X_all[labeled_mask].to(device)
        y_labeled = y_all[labeled_mask].to(device)
        train_model(model, X_labeled, y_labeled)

        pool_idx = (~labeled_mask).nonzero(as_tuple=True)[0]
        X_pool = X_all[pool_idx].to(device)
        local_preds = predict_pool(model, X_pool)

        all_preds = gather_ensemble_metrics(local_preds)  # (M, pool_size)
        mu = all_preds.mean(dim=0)
        sigma = all_preds.std(dim=0)

        if use_ucb:
            ucb = mu + UCB_KAPPA * sigma
            if rank == 0:
                topk = ucb.topk(BATCH_K).indices
            else:
                topk = torch.empty(BATCH_K, dtype=torch.long, device=device)
            dist.broadcast(topk, src=0)  # ty: ignore[possibly-missing-attribute]
        else:
            ucb = None
            if rank == 0:
                topk = torch.randperm(len(pool_idx), device=device)[:BATCH_K]
            else:
                topk = torch.empty(BATCH_K, dtype=torch.long, device=device)
            dist.broadcast(topk, src=0)  # ty: ignore[possibly-missing-attribute]

        selected_global = pool_idx[topk.cpu()]
        labeled_mask[selected_global] = True

        new_hits = int(y_all[selected_global].sum().item())
        hits_so_far += new_hits
        result.hits_history.append(hits_so_far)

        if rank == 0:
            result.snapshots.append(
                IterSnapshot(
                    mu=mu.cpu(),
                    sigma=sigma.cpu(),
                    ucb=ucb.cpu() if ucb is not None else None,
                    pool_idx=pool_idx.clone(),
                    selected_global=selected_global.clone(),
                    hits_so_far=hits_so_far,
                    n_labeled=int(labeled_mask.sum().item()),
                )
            )

        if rank == 0 and (it + 1) % 10 == 0:
            tag = "UCB" if use_ucb else "Random"
            n_labeled = labeled_mask.sum().item()
            print(
                f"  [{tag}] Iter {it + 1:3d} | labeled: {n_labeled} | "
                f"hits: {hits_so_far}/{n_active_total} ({hits_so_far / n_active_total * 100:.1f}%)"
            )

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _compute_pca_2d(X_all):
    """Project fingerprints to 2D via PCA (torch.pca_lowrank)."""
    X_centered = X_all - X_all.mean(dim=0)
    _, _, V = torch.pca_lowrank(X_centered, q=2)
    return X_centered @ V


def _fig_to_pil(fig):
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()


def _save_gif(frames, path, duration_ms=200):
    durations = [duration_ms] * len(frames)
    if len(durations) > 1:
        durations[-1] = duration_ms * 5
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def make_gif_pca_landscape(ucb_result, X_all, y_all, out_path):
    """GIF 1: PCA projection colored by UCB, actives shown as green diamonds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    coords = _compute_pca_2d(X_all).numpy()
    active_mask = y_all.numpy() == 1

    # Fixed axes
    pad = 0.05
    xr = coords[:, 0].max() - coords[:, 0].min()
    yr = coords[:, 1].max() - coords[:, 1].min()
    xlim = (coords[:, 0].min() - pad * xr, coords[:, 0].max() + pad * xr)
    ylim = (coords[:, 1].min() - pad * yr, coords[:, 1].max() + pad * yr)

    # Fixed colorbar from global UCB range (robust)
    all_ucb = torch.cat([s.ucb for s in ucb_result.snapshots if s.ucb is not None])
    ucb_lo, ucb_hi = all_ucb.quantile(0.01).item(), all_ucb.quantile(0.99).item()
    if ucb_hi <= ucb_lo:
        ucb_hi = ucb_lo + 1e-8
    global_norm = Normalize(vmin=ucb_lo, vmax=ucb_hi)

    frames = []
    labeled_set = set(ucb_result.initial_labeled.tolist())

    for it, snap in enumerate(ucb_result.snapshots):
        fig, ax = plt.subplots(figsize=(8, 6))

        pool = snap.pool_idx.numpy()
        pool_coords = coords[pool]
        vals = snap.ucb.numpy() if snap.ucb is not None else snap.mu.numpy()
        label = "UCB" if snap.ucb is not None else "P(active)"

        sc = ax.scatter(
            pool_coords[:, 0],
            pool_coords[:, 1],
            c=vals,
            cmap="viridis",
            s=1,
            alpha=0.3,
            norm=global_norm,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)

        # Show all true actives as faint green diamonds (ground truth)
        ax.scatter(
            coords[active_mask, 0],
            coords[active_mask, 1],
            c="none",
            edgecolors="limegreen",
            s=15,
            marker="D",
            linewidths=0.5,
            alpha=0.4,
            label=f"True actives ({active_mask.sum()})",
        )

        # Labeled so far
        labeled_arr = list(labeled_set)
        ax.scatter(coords[labeled_arr, 0], coords[labeled_arr, 1], c="gray", s=3, alpha=0.2)

        # Selected this iteration
        sel = snap.selected_global.numpy()
        sel_active = y_all[sel].numpy() == 1
        ax.scatter(
            coords[sel[~sel_active], 0],
            coords[sel[~sel_active], 1],
            c="red",
            s=60,
            marker="*",
            edgecolors="k",
            linewidths=0.3,
            zorder=5,
        )
        ax.scatter(
            coords[sel[sel_active], 0],
            coords[sel[sel_active], 1],
            c="gold",
            s=100,
            marker="*",
            edgecolors="k",
            linewidths=0.5,
            zorder=6,
            label=f"Hit! ({sel_active.sum()})",
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Iter {it + 1}/{BO_ITERS} — hits: {snap.hits_so_far}")
        ax.legend(loc="upper right", fontsize=7, markerscale=0.8)

        frames.append(_fig_to_pil(fig))
        plt.close(fig)
        labeled_set.update(sel.tolist())

    _save_gif(frames, out_path)
    print(f"  PCA landscape GIF saved to {out_path}")


def make_gif_histograms(ucb_result, out_path):
    """GIF 2: Histograms of P(active) and σ over pool."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    all_sigma = torch.cat([s.sigma for s in ucb_result.snapshots])

    mu_bins = np.linspace(0, 1, 51)  # P(active) is always in [0, 1]
    sigma_max = all_sigma.quantile(0.99).item()
    sigma_bins = np.linspace(0, sigma_max, 51)

    mu_ymax = max(np.histogram(s.mu.numpy().clip(0, 1), bins=mu_bins)[0].max() for s in ucb_result.snapshots)
    sigma_ymax = max(
        np.histogram(s.sigma.numpy().clip(0, sigma_max), bins=sigma_bins)[0].max() for s in ucb_result.snapshots
    )

    frames = []
    for it, snap in enumerate(ucb_result.snapshots):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        mu = snap.mu.numpy()
        sigma = snap.sigma.numpy()

        pool_set = {g.item(): i for i, g in enumerate(snap.pool_idx)}
        sel_pool_idx = [pool_set[g.item()] for g in snap.selected_global if g.item() in pool_set]

        ax1.hist(mu, bins=mu_bins, color="steelblue", alpha=0.7, edgecolor="none")
        if sel_pool_idx:
            for v in mu[sel_pool_idx]:
                ax1.axvline(v, color="red", lw=1, alpha=0.7)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, mu_ymax * 1.1)
        ax1.set_xlabel("Ensemble mean P(active)")
        ax1.set_ylabel("Count")
        ax1.set_title("Predicted activity on pool")

        ax2.hist(sigma, bins=sigma_bins, color="coral", alpha=0.7, edgecolor="none")
        if sel_pool_idx:
            for v in sigma[sel_pool_idx]:
                ax2.axvline(v, color="red", lw=1, alpha=0.7)
        ax2.set_xlim(0, sigma_max)
        ax2.set_ylim(0, sigma_ymax * 1.1)
        ax2.set_xlabel("Ensemble σ")
        ax2.set_ylabel("Count")
        ax2.set_title("Ensemble disagreement on pool")

        fig.suptitle(f"Iter {it + 1}/{BO_ITERS} — labeled: {snap.n_labeled} | hits: {snap.hits_so_far}", fontsize=11)
        fig.tight_layout()
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    _save_gif(frames, out_path)
    print(f"  Histogram GIF saved to {out_path}")


def make_gif_dual_panel(ucb_result, random_result, n_active_total, n_pool, out_path):
    """GIF 3: Left = hit discovery curve, Right = μ vs σ scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oracle_calls = [N_INITIAL + i * BATCH_K for i in range(len(ucb_result.hits_history))]

    # Fixed μ-vs-σ axes
    all_mu = torch.cat([s.mu for s in ucb_result.snapshots])
    all_sigma = torch.cat([s.sigma for s in ucb_result.snapshots])
    sigma_max = all_sigma.quantile(0.99).item()
    sigma_pad = 0.05 * sigma_max

    # Expected random baseline: start from actual initial hits (stratified selection),
    # then use remaining pool hit rate for subsequent calls
    n_init_hits = ucb_result.hits_history[0]
    remaining_hit_rate = (n_active_total - n_init_hits) / (n_pool - N_INITIAL)
    expected_y = [n_init_hits + (x - N_INITIAL) * remaining_hit_rate for x in [oracle_calls[0], oracle_calls[-1]]]

    frames = []
    for it, snap in enumerate(ucb_result.snapshots):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        step = it + 2
        ax1.plot(
            oracle_calls[:step], ucb_result.hits_history[:step], "o-", ms=3, lw=2, color="C0", label="UCB (ensemble)"
        )
        ax1.plot(
            oracle_calls[:step],
            random_result.hits_history[:step],
            "s--",
            ms=3,
            lw=1.5,
            color="C1",
            alpha=0.7,
            label="Random",
        )
        # Expected hits for random baseline (accounts for stratified initial selection)
        ax1.plot(
            [oracle_calls[0], oracle_calls[-1]],
            expected_y,
            ":",
            color="gray",
            alpha=0.5,
            label="Random (expected)",
        )
        ax1.set_xlim(oracle_calls[0] - 20, oracle_calls[-1] + 20)
        ax1.set_ylim(0, max(max(ucb_result.hits_history), max(random_result.hits_history)) * 1.15)
        ax1.set_xlabel("Oracle calls (labeled molecules)")
        ax1.set_ylabel("Cumulative hits found")
        ax1.set_title("Hit discovery curve")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        mu = snap.mu.numpy()
        sigma = snap.sigma.numpy()
        ax2.scatter(mu, sigma, s=2, alpha=0.2, c="steelblue", rasterized=True)

        pool_set = {g.item(): i for i, g in enumerate(snap.pool_idx)}
        sel_pool_idx = [pool_set[g.item()] for g in snap.selected_global if g.item() in pool_set]
        if sel_pool_idx:
            ax2.scatter(
                mu[sel_pool_idx],
                sigma[sel_pool_idx],
                s=60,
                c="red",
                marker="*",
                edgecolors="k",
                linewidths=0.3,
                zorder=5,
                label="Selected",
            )
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-sigma_pad, sigma_max + sigma_pad)
        ax2.set_xlabel("Ensemble mean P(active)")
        ax2.set_ylabel("Ensemble σ")
        ax2.set_title("Pool predictions")
        ax2.legend(fontsize=8, loc="upper right")

        fig.suptitle(f"Iter {it + 1}/{BO_ITERS}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    _save_gif(frames, out_path)
    print(f"  Dual panel GIF saved to {out_path}")


def _save_static_plot(ucb_history, random_history, n_active_total, n_pool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    oracle_calls = [N_INITIAL + i * BATCH_K for i in range(len(ucb_history))]

    # Expected random baseline: start from actual initial hits (stratified selection),
    # then use remaining pool hit rate for subsequent calls
    n_init_hits = ucb_history[0]
    remaining_hit_rate = (n_active_total - n_init_hits) / (n_pool - N_INITIAL)
    expected_y = [n_init_hits + (x - N_INITIAL) * remaining_hit_rate for x in [oracle_calls[0], oracle_calls[-1]]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(oracle_calls, ucb_history, "o-", ms=3, lw=2, label="UCB (ensemble)")
    ax.plot(oracle_calls, random_history, "s--", ms=3, lw=1.5, alpha=0.7, label="Random")
    ax.plot(
        [oracle_calls[0], oracle_calls[-1]],
        expected_y,
        ":",
        color="gray",
        alpha=0.5,
        label="Random (expected)",
    )
    ax.set_xlabel("Oracle calls (labeled molecules)")
    ax.set_ylabel("Cumulative hits found")
    ax.set_title(f"HTS Hit Finding — Cav3 T-type Calcium ({n_active_total} actives in {n_pool} pool)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = LOG_DIR / "hit_discovery.png"
    fig.savefig(path, dpi=150)
    print(f"  Static plot saved to {path}")


def main():
    rank, world_size = setup_independent_worker(base_seed=42)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("=== Example 04b: HTS Hit Finding with Ensemble UCB ===")
        print(f"Ensemble size: {world_size}, AL iters: {BO_ITERS}, batch: {BATCH_K}, κ: {UCB_KAPPA}")

    X_all, y_all = download_and_featurize(rank)
    n_active_total = int(y_all.sum().item())

    if rank == 0:
        print(f"Pool: {len(y_all)} molecules, {n_active_total} actives ({n_active_total / len(y_all) * 100:.2f}%)")
        print("\nRunning UCB acquisition ...")

    ucb_result = run_al(rank, world_size, X_all, y_all, device, use_ucb=True)

    if rank == 0:
        print("\nRunning random baseline ...")

    random_result = run_al(rank, world_size, X_all, y_all, device, use_ucb=False)

    if rank == 0:
        total_queries = N_INITIAL + BO_ITERS * BATCH_K
        print(f"\n===== Results ({total_queries} oracle calls) =====")
        print(
            f"  UCB    hits: {ucb_result.hits_history[-1]}/{n_active_total} "
            f"({ucb_result.hits_history[-1] / n_active_total * 100:.1f}%)"
        )
        print(
            f"  Random hits: {random_result.hits_history[-1]}/{n_active_total} "
            f"({random_result.hits_history[-1] / n_active_total * 100:.1f}%)"
        )
        expected = int(total_queries * n_active_total / len(y_all))
        print(f"  Random (expected): ~{expected}")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print("\nGenerating visualizations ...")
        _save_static_plot(ucb_result.hits_history, random_result.hits_history, n_active_total, len(y_all))
        make_gif_pca_landscape(ucb_result, X_all, y_all, LOG_DIR / "pca_landscape.gif")
        make_gif_histograms(ucb_result, LOG_DIR / "histograms.gif")
        make_gif_dual_panel(ucb_result, random_result, n_active_total, len(y_all), LOG_DIR / "dual_panel.gif")
        print("\nDone!")

    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


if __name__ == "__main__":
    main()
