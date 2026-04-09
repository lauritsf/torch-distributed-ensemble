# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.7.1",
#     "torch-distributed-ensemble",
#     "matplotlib",
# ]
# ///
"""Heteroscedastic deep ensemble on a 1D regression task (vanilla PyTorch).

Each torchrun worker trains its own MLP with mean + variance heads using
Gaussian NLL loss. After training, predictions are gathered to show:
  - Ensemble mean tracks the true function better than individual members
  - Aleatoric uncertainty (learned noise) captures input-dependent noise
  - Epistemic uncertainty (member disagreement) is high where data is sparse

The regression task (x·sin(x) with heteroscedastic noise) is adapted from
Skafte et al., "Reliable training and estimation of variance networks"
(https://arxiv.org/abs/1906.03260), code: https://github.com/SkafteNicki/john

Usage:
    torchrun --nproc_per_node=N examples/00_heteroscedastic_regression/main.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from torch_distributed_ensemble import gather_ensemble_metrics, setup_independent_worker

N_ITER = 10000
LR = 1e-2
N_HIDDEN = 50


def target_fn(x):
    return x * torch.sin(x)


def make_data(device):
    """20 sparse noisy training points + dense test grid."""
    generator = torch.Generator().manual_seed(0)
    X_train = torch.rand(20, 1, generator=generator) * 10
    noise = 0.3 * torch.randn(20, generator=generator)
    noise += 0.3 * X_train.squeeze() * torch.randn(20, generator=generator)
    y_train = target_fn(X_train.squeeze()) + noise

    X_test = torch.linspace(-4, 14, 500).unsqueeze(1)
    y_test = target_fn(X_test.squeeze())
    return (
        X_train.to(device),
        y_train.to(device),
        X_test.to(device),
        y_test.to(device),
    )


class HeteroscedasticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(1, N_HIDDEN),
            nn.Sigmoid(),
            nn.Linear(N_HIDDEN, 1),
        )
        self.var_head = nn.Sequential(
            nn.Linear(1, N_HIDDEN),
            nn.Sigmoid(),
            nn.Linear(N_HIDDEN, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.mean_head(x).squeeze(-1), self.var_head(x).squeeze(-1)


def gaussian_nll(mean, var, target):
    return (var.log() + (mean - target) ** 2 / var).mean()


def main():
    rank, world_size = setup_independent_worker(base_seed=42)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test = make_data(device)
    torch.manual_seed(42 + rank)  # re-seed after data creation
    model = HeteroscedasticMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training loop ---
    model.train()
    for it in range(N_ITER):
        optimizer.zero_grad()
        mean, var = model(X_train)
        # Fixed variance for first half → stabilizes mean learning
        if it < N_ITER // 2:
            var = torch.full_like(var, 0.002)
        loss = gaussian_nll(mean, var, y_train)
        loss.backward()
        optimizer.step()

        if (it + 1) % 2000 == 0:
            all_loss = gather_ensemble_metrics(loss.detach())
            if rank == 0:
                parts = " | ".join(f"m{i}: {val:.4f}" for i, val in enumerate(all_loss))
                print(f"[Iter {it + 1:5d}] NLL  {parts}")

    # --- Final evaluation ---
    model.eval()
    with torch.no_grad():
        mean, var = model(X_test)

    all_means = gather_ensemble_metrics(mean)  # (world_size, 500)
    all_vars = gather_ensemble_metrics(var)

    if rank == 0:
        # Ensemble: mixture of Gaussians
        ens_mean = all_means.mean(dim=0)
        aleatoric = all_vars.mean(dim=0)
        epistemic = all_means.var(dim=0)
        total_var = aleatoric + epistemic

        ens_mse = ((ens_mean - y_test) ** 2).mean()
        member_mses = ((all_means - y_test.unsqueeze(0)) ** 2).mean(dim=1)

        print("\n===== Final Ensemble Report =====")
        for i, mse in enumerate(member_mses):
            print(f"  Member {i} | MSE: {mse:.4f}")
        print(f"  Ensemble | MSE: {ens_mse:.4f}")
        print(f"  Gain     | MSE: -{member_mses.mean() - ens_mse:.4f}")
        print(f"  Aleatoric (avg) | {aleatoric.mean():.4f}")
        print(f"  Epistemic (avg) | {epistemic.mean():.4f}")

        # --- Plot (rank 0 only) ---
        _save_plot(
            X_train.cpu(),
            y_train.cpu(),
            X_test.cpu(),
            y_test.cpu(),
            all_means.cpu(),
            all_vars.cpu(),
            ens_mean.cpu(),
            aleatoric.cpu(),
            epistemic.cpu(),
            total_var.cpu(),
            ens_mse.item(),
            member_mses.cpu(),
        )

    dist.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def _save_plot(
    X_train,
    y_train,
    X_test,
    y_test,
    all_means,
    all_vars,
    ens_mean,
    aleatoric,
    epistemic,
    total_var,
    ens_mse,
    member_mses,
):
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    x = X_test.squeeze().numpy()
    ens = ens_mean.numpy()
    total_std = total_var.sqrt().numpy()
    aleatoric_std = aleatoric.sqrt().numpy()
    epistemic_std = epistemic.sqrt().numpy()
    true_std = 0.3 + 0.3 * x

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Top: predictions
    ax1.plot(x, y_test.numpy(), ":", lw=2, label=r"$f(x) = x\,\sin(x)$")
    ax1.plot(X_train.squeeze().numpy(), y_train.numpy(), "o", ms=6, label="Training data")
    for i in range(all_means.shape[0]):
        ax1.plot(x, all_means[i].numpy(), alpha=0.4, lw=1, label=f"Member {i}")
    ax1.plot(x, ens, "k-", lw=2, label="Ensemble mean")
    ax1.fill_between(
        x, ens - 1.96 * total_std, ens + 1.96 * total_std, alpha=0.2, color="k", label="95% prediction interval"
    )
    ax1.set_ylabel("y")
    ax1.set_ylim(-15, 15)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"Heteroscedastic Deep Ensemble — Ens MSE: {ens_mse:.2f}")

    # Bottom: uncertainty decomposition
    ax2.plot(x, total_std, "k-", lw=2, label="Total std")
    ax2.plot(x, aleatoric_std, "b--", lw=2, label="Aleatoric std")
    ax2.plot(x, epistemic_std, "r--", lw=2, label="Epistemic std")
    mask = (x >= 0) & (x <= 10)
    ax2.plot(x[mask], true_std[mask], "g:", lw=2, label="True noise std")
    ax2.set_xlabel("x")
    ax2.set_ylabel("std")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.set_title("Uncertainty decomposition")

    fig.tight_layout()
    path = outputs_dir / "ensemble_predictions.png"
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    main()
