# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pillow",
# ]
# ///
"""Plot HPO comparison: ensemble-tuned vs member-tuned sweeps.

Reads fine_grid.jsonl (weight decay sweep) and HPO results from train.py,
then generates comparison figures showing how ensemble-tuned HPO prefers
lower regularization than member-tuned HPO.

Usage:
    uv run python examples/03_hpo_ensemble/plot.py
"""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LOG_DIR = Path(__file__).resolve().parent / "outputs"


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def best_so_far(results: list[dict], key: str) -> list[float]:
    vals = []
    best = float("inf")
    for r in results:
        best = min(best, r[key])
        vals.append(best)
    return vals


def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()


def _save_gif(frames, path, duration_ms=400):
    durations = [duration_ms] * len(frames)
    if len(durations) > 1:
        durations[-1] = duration_ms * 5
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


# ── Fine grid plot ───────────────────────────────────────────────────────────


def plot_fine_grid():
    """Plot member MSE and ensemble MSE vs weight decay from fine grid data.

    Shows a single panel for the best config (wine h=50 L=1) demonstrating
    how optimal weight decay differs for ensemble vs member objectives.
    """
    path = LOG_DIR / "fine_grid.jsonl"
    if not path.exists():
        print(f"No fine grid data at {path}")
        return

    rows = load_results(path)
    # Filter to wine n=1279 h=50 L=1 — the strongest result
    target = [
        r for r in rows if r["dataset"] == "wine" and r["n_train"] == 1279 and r["hidden"] == 50 and r["layers"] == 1
    ]
    if not target:
        print("No wine h=50 L=1 data in fine_grid.jsonl")
        return
    target.sort(key=lambda r: r["wd"])

    wds = [r["wd"] for r in target]
    mem_mse = [r["mem_mse"] for r in target]
    ens_mse = [r["ens_mse"] for r in target]
    gain = [r["mem_mse"] - r["ens_mse"] for r in target]

    best_mem_i = int(np.argmin(mem_mse))
    best_ens_i = int(np.argmin(ens_mse))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MSE curves
    ax1.plot(wds, mem_mse, "s-", color="C1", ms=6, lw=2, label="Avg member MSE")
    ax1.plot(wds, ens_mse, "o-", color="C0", ms=6, lw=2, label="Ensemble MSE")
    ax1.axvline(
        wds[best_mem_i], color="C1", ls="--", alpha=0.7, lw=1.5, label=f"Best for members (wd={wds[best_mem_i]:.2e})"
    )
    ax1.axvline(
        wds[best_ens_i], color="C0", ls="--", alpha=0.7, lw=1.5, label=f"Best for ensemble (wd={wds[best_ens_i]:.2e})"
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Weight Decay", fontsize=12)
    ax1.set_ylabel("Validation MSE", fontsize=12)
    ax1.set_title("Optimal weight decay differs by objective", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Ensemble gain
    ax2.fill_between(wds, 0, gain, alpha=0.3, color="C2")
    ax2.plot(wds, gain, "D-", color="C2", ms=5, lw=2, label="Ensemble gain")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.axvline(wds[best_mem_i], color="C1", ls="--", alpha=0.7, lw=1.5, label="Member optimum")
    ax2.axvline(wds[best_ens_i], color="C0", ls="--", alpha=0.7, lw=1.5, label="Ensemble optimum")
    ax2.set_xscale("log")
    ax2.set_xlabel("Weight Decay", fontsize=12)
    ax2.set_ylabel("Avg Member MSE − Ensemble MSE", fontsize=12)
    ax2.set_title("Ensemble gain: diversity benefit at low regularization", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Wine Quality (UCI) — 1-layer MLP, 4 ensemble members\n"
        f"Ensemble-tuned HPO prefers wd={wds[best_ens_i]:.2e} (ens MSE={ens_mse[best_ens_i]:.4f}), "
        f"member-tuned prefers wd={wds[best_mem_i]:.2e} "
        f"(ens MSE={ens_mse[best_ens_i]:.4f} → {ens_mse[best_mem_i]:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_path = LOG_DIR / "fine_grid_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved fine grid plot to {out_path}")


# ── HPO comparison ───────────────────────────────────────────────────────────


def plot_hpo_comparison(ens_results, mem_results):
    best_ens = min(ens_results, key=lambda r: r["ens_mse"])
    best_mem = min(mem_results, key=lambda r: r["avg_member_mse"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ens_wd = [r["params"]["weight_decay"] for r in ens_results]
    mem_wd = [r["params"]["weight_decay"] for r in mem_results]

    # (0) Best-so-far Ensemble MSE
    ax = axes[0]
    trials = np.arange(1, max(len(ens_results), len(mem_results)) + 1)
    ax.plot(
        trials[: len(ens_results)],
        best_so_far(ens_results, "ens_mse"),
        "o-",
        ms=3,
        lw=2,
        label="Tuned for ensemble MSE",
    )
    ax.plot(
        trials[: len(mem_results)],
        best_so_far(mem_results, "ens_mse"),
        "s-",
        ms=3,
        lw=1.5,
        label="Tuned for member MSE",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Ensemble MSE")
    ax.set_title("Best-so-far Ensemble MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1) Combined: weight decay vs both metrics, marker=sweep, color=metric
    ax = axes[1]
    ens_mse_vals = [r["ens_mse"] for r in ens_results]
    ens_mem_vals = [r["avg_member_mse"] for r in ens_results]
    mem_mse_vals = [r["ens_mse"] for r in mem_results]
    mem_mem_vals = [r["avg_member_mse"] for r in mem_results]

    # Ensemble sweep: circles
    ax.scatter(
        ens_wd,
        ens_mse_vals,
        c="C0",
        s=40,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
        marker="o",
        label="Ensemble sweep — ensemble MSE",
        zorder=3,
    )
    ax.scatter(
        ens_wd,
        ens_mem_vals,
        c="C1",
        s=40,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
        marker="o",
        label="Ensemble sweep — member MSE",
        zorder=3,
    )
    # Member sweep: squares
    ax.scatter(
        mem_wd,
        mem_mse_vals,
        c="C0",
        s=40,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
        marker="s",
        label="Member sweep — ensemble MSE",
        zorder=3,
    )
    ax.scatter(
        mem_wd,
        mem_mem_vals,
        c="C1",
        s=40,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
        marker="s",
        label="Member sweep — member MSE",
        zorder=3,
    )
    # Best-trial highlights: same size/alpha as other markers, red edge
    ax.scatter(
        [best_ens["params"]["weight_decay"]],
        [best_ens["ens_mse"]],
        c="C0",
        s=40,
        alpha=0.6,
        marker="o",
        edgecolors="red",
        linewidths=2,
        zorder=5,
    )
    ax.scatter(
        [best_ens["params"]["weight_decay"]],
        [best_ens["avg_member_mse"]],
        c="C1",
        s=40,
        alpha=0.6,
        marker="o",
        edgecolors="red",
        linewidths=2,
        zorder=5,
    )
    ax.scatter(
        [best_mem["params"]["weight_decay"]],
        [best_mem["ens_mse"]],
        c="C0",
        s=40,
        alpha=0.6,
        marker="s",
        edgecolors="red",
        linewidths=2,
        zorder=5,
    )
    ax.scatter(
        [best_mem["params"]["weight_decay"]],
        [best_mem["avg_member_mse"]],
        c="C1",
        s=40,
        alpha=0.6,
        marker="s",
        edgecolors="red",
        linewidths=2,
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("MSE")
    ax.set_title("Weight Decay vs MSE\n(● ensemble sweep, ■ member sweep | blue=ensemble MSE, orange=member MSE)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"HPO Comparison: Ensemble-tuned vs Member-tuned (weight decay only)\n"
        f"Best ens MSE: {best_ens['ens_mse']:.4f} (ensemble, wd={best_ens['params']['weight_decay']:.2e}) vs "
        f"{best_mem['ens_mse']:.4f} (member, wd={best_mem['params']['weight_decay']:.2e})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_path = LOG_DIR / "hpo_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")


# ── Sweep GIF ────────────────────────────────────────────────────────────────


def make_sweep_gif(ens_results, mem_results):
    """GIF: side-by-side 1D sweeps, one frame per trial."""
    n_trials = max(len(ens_results), len(mem_results))

    all_ens_mse = [r["ens_mse"] for r in ens_results + mem_results]
    all_mem_mse = [r["avg_member_mse"] for r in ens_results + mem_results]
    mse_q95 = np.quantile(all_ens_mse + all_mem_mse, 0.95)
    mse_min = min(min(all_ens_mse), min(all_mem_mse))
    ylim = (mse_min * 0.95, min(mse_q95 * 1.2, max(all_ens_mse + all_mem_mse) * 1.05))

    frames = []

    for t in range(n_trials):
        fig, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, results, title, color, obj_key in [
            (ax_e, ens_results, "Ensemble-tuned", "C0", "ens_mse"),
            (ax_m, mem_results, "Member-tuned", "C1", "avg_member_mse"),
        ]:
            n = min(t + 1, len(results))
            best_idx = 0
            best_val = results[0][obj_key]
            for i in range(1, n):
                if results[i][obj_key] < best_val:
                    best_val = results[i][obj_key]
                    best_idx = i

            for i in range(n):
                if i == best_idx or i == n - 1:
                    continue
                wd = results[i]["params"]["weight_decay"]
                ax.scatter([wd], [results[i]["ens_mse"]], c="gray", s=25, alpha=0.25, edgecolors="none", zorder=2)

            best_r = results[best_idx]
            ax.scatter(
                [best_r["params"]["weight_decay"]],
                [best_r["ens_mse"]],
                c=color,
                s=200,
                marker="*",
                edgecolors="k",
                linewidths=0.8,
                zorder=5,
                label=f"Best ({obj_key}={best_val:.4f})",
            )

            if n - 1 != best_idx:
                curr = results[n - 1]
                ax.scatter(
                    [curr["params"]["weight_decay"]],
                    [curr["ens_mse"]],
                    c="red",
                    s=80,
                    edgecolors="k",
                    linewidths=0.5,
                    zorder=4,
                    label=f"Current ({obj_key}={curr[obj_key]:.4f})",
                )

            ax.set_xscale("log")
            ax.set_xlabel("Weight Decay")
            ax.set_ylabel("Ensemble MSE")
            ax.set_ylim(ylim)
            ax.set_title(f"{title} (trial {n}/{len(results)})", fontsize=11)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"HPO Sweep — Trial {t + 1}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    out_path = LOG_DIR / "hpo_sweep.gif"
    _save_gif(frames, out_path)
    print(f"Saved sweep GIF to {out_path}")


if __name__ == "__main__":
    # Always plot fine grid if available
    plot_fine_grid()

    # Prefer 3-seed results if available, fall back to single-seed
    ens_path = LOG_DIR / "results_wine_all_ensemble_3seeds.jsonl"
    mem_path = LOG_DIR / "results_wine_all_member_3seeds.jsonl"
    if not ens_path.exists():
        ens_path = LOG_DIR / "results_wine_all_ensemble.jsonl"
    if not mem_path.exists():
        mem_path = LOG_DIR / "results_wine_all_member.jsonl"

    if ens_path.exists() and mem_path.exists():
        ens_results = load_results(ens_path)
        mem_results = load_results(mem_path)
        plot_hpo_comparison(ens_results, mem_results)
        make_sweep_gif(ens_results, mem_results)
    else:
        print(f"HPO results not found at {ens_path} / {mem_path}")
