import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.nn import functional as F
from torchmetrics import MetricCollection


class DistributedSeeder(Callback):
    """Seed each GPU's random state at the start of training.

    Args:
        base_seed: Base random seed. Each rank gets ``base_seed + rank``
            (when ``unique=True``) or ``base_seed`` (when ``unique=False``).
        unique: If True (default), each rank gets a unique seed and non-zero
            ranks re-initialize model weights so ensemble members diverge.
            If False, all ranks get the same seed and weights are not
            re-initialized (useful for ablations or debugging).
    """

    def __init__(self, base_seed=42, unique=True):
        self.base_seed = base_seed
        self.unique = unique

    def on_fit_start(self, trainer, pl_module):
        rank = trainer.global_rank
        if self.unique:
            L.seed_everything(self.base_seed + rank)
            if rank != 0:
                self._reset_weights(pl_module)
        else:
            L.seed_everything(self.base_seed)

    @staticmethod
    def _reset_weights(module):
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class EnsembleMetrics(Callback):
    """Gather validation predictions across GPUs and log ensemble metrics.

    Expects ``validation_step`` to return ``{"probs": tensor, "target": tensor}``.

    After each validation epoch, all-gathers predictions from every rank and
    computes metrics on both the ensemble (mean of member probabilities) and
    each individual member. All ranks compute identical values, so Lightning's
    ``EarlyStopping(monitor="val/ens_nll")`` works without any special handling.

    NLL is always computed internally. Additional metrics can be supplied via
    a torchmetrics ``MetricCollection``::

        from torchmetrics import MetricCollection
        from torchmetrics.classification import MulticlassAccuracy

        EnsembleMetrics(metrics=MetricCollection({
            "acc": MulticlassAccuracy(num_classes=7),
        }))

    Logged metrics per epoch:
        - ``val/ens_nll`` and ``val/member_N_nll`` — always
        - ``val/ens_<name>`` and ``val/member_N_<name>`` — for each metric in the collection
        - ``val/disagreement`` — fraction of samples where member argmax predictions differ
    """

    def __init__(self, metrics: MetricCollection | None = None):
        self.metrics = metrics
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._ens_metrics: MetricCollection | None = None
        self._member_metrics: list[MetricCollection] | None = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._preds.append(outputs["probs"])
        self._targets.append(outputs["target"])

    def on_validation_epoch_end(self, trainer, pl_module):
        local_p = torch.cat(self._preds)
        local_t = torch.cat(self._targets)
        all_p = trainer.strategy.all_gather(local_p)
        all_t = trainer.strategy.all_gather(local_t)

        # Normalize shape: all_gather may add batch dims in distributed mode
        if all_p.ndim == 2:
            all_p = all_p.unsqueeze(0)
        if all_p.ndim == 4:
            all_p = all_p.squeeze(1)
        if all_t.ndim == 1:
            all_t = all_t.unsqueeze(0)

        target = all_t[0]
        num_members = all_p.shape[0]

        # Lazily create per-member metric clones on the right device
        if self.metrics is not None and self._ens_metrics is None:
            device = target.device
            self._ens_metrics = self.metrics.clone(prefix="val/ens_").to(device)
            self._member_metrics = [
                self.metrics.clone(prefix=f"val/member_{i}_").to(device) for i in range(num_members)
            ]

        # ── Ensemble metrics ─────────────────────────────────────────
        ens_probs = all_p.mean(dim=0)
        ens_nll = F.nll_loss(torch.log(ens_probs + 1e-8), target)

        logged = {"val/ens_nll": ens_nll}
        if self._ens_metrics is not None:
            logged.update(self._ens_metrics(ens_probs, target))

        # ── Per-member metrics ───────────────────────────────────────
        for i in range(num_members):
            p = all_p[i]
            logged[f"val/member_{i}_nll"] = F.nll_loss(torch.log(p + 1e-8), target)
            if self._member_metrics is not None:
                logged.update(self._member_metrics[i](p, target))

        # ── Disagreement ─────────────────────────────────────────────
        member_preds = all_p.argmax(dim=2)
        logged["val/disagreement"] = (member_preds.max(dim=0).values != member_preds.min(dim=0).values).float().mean()

        # Log on all ranks (values are identical) — no sync needed
        pl_module.log_dict(logged, rank_zero_only=False, sync_dist=False)

        # Print report (rank 0 only)
        if trainer.is_global_zero:
            print(f"\n[Epoch {trainer.current_epoch}] Ensemble NLL: {ens_nll:.4f}", end="")
            for key in sorted(k for k in logged if k.startswith("val/ens_") and k != "val/ens_nll"):
                print(f"  {key.split('/')[-1]}: {logged[key]:.4f}", end="")
            print()

        self._preds.clear()
        self._targets.clear()

        # Reset torchmetrics state for next epoch
        if self._ens_metrics is not None and self._member_metrics is not None:
            self._ens_metrics.reset()
            for mc in self._member_metrics:
                mc.reset()


class PerRankCheckpoint(Callback):
    """Save a separate checkpoint file for every GPU."""

    def __init__(self, dirpath="checkpoints"):
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            rank = trainer.global_rank
            filename = f"{self.dirpath}/final-rank={rank}.ckpt"
            trainer.save_checkpoint(filename)


class RegressionEnsembleMetrics(Callback):
    """Gather validation predictions across GPUs and log ensemble regression metrics.

    Expects ``validation_step`` to return ``{"preds": tensor, "target": tensor}``
    where both tensors have shape ``(batch,)``.

    After each validation epoch, all-gathers predictions from every rank and
    computes MSE-based metrics on both the ensemble (mean of member predictions)
    and each individual member. All ranks compute identical values, so Lightning's
    ``EarlyStopping(monitor="val/ens_mse")`` works without any special handling.

    Logged metrics per epoch:

    - ``val/ens_mse`` — MSE of the ensemble mean prediction
    - ``val/member_N_mse`` for N=0..M-1 — per-member MSE
    - ``val/avg_member_mse`` — mean of per-member MSE (member-tuned HPO objective)
    - ``val/ensemble_gain`` — ``avg_member_mse - ens_mse`` (positive = ensemble helps)
    - ``val/ens_std`` — mean per-sample ensemble std (diversity measure)
    """

    def __init__(self):
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._preds.append(outputs["preds"])
        self._targets.append(outputs["target"])

    def on_validation_epoch_end(self, trainer, pl_module):
        local_p = torch.cat(self._preds)
        local_t = torch.cat(self._targets)
        all_p = trainer.strategy.all_gather(local_p)
        all_t = trainer.strategy.all_gather(local_t)

        # Normalize shape: all_gather returns (N,) on single device, (M, N) on multi
        if all_p.ndim == 1:
            all_p = all_p.unsqueeze(0)
        if all_t.ndim == 1:
            all_t = all_t.unsqueeze(0)

        target = all_t[0]
        num_members = all_p.shape[0]

        ens_mean = all_p.mean(dim=0)
        ens_mse = F.mse_loss(ens_mean, target)

        logged: dict[str, torch.Tensor] = {"val/ens_mse": ens_mse}

        member_mses = []
        for i in range(num_members):
            m_mse = F.mse_loss(all_p[i], target)
            logged[f"val/member_{i}_mse"] = m_mse
            member_mses.append(m_mse)

        avg_member_mse = torch.stack(member_mses).mean()
        logged["val/avg_member_mse"] = avg_member_mse
        logged["val/ensemble_gain"] = avg_member_mse - ens_mse

        if num_members > 1:
            ens_std = all_p.std(dim=0).mean()
        else:
            ens_std = torch.tensor(0.0, device=ens_mse.device)
        logged["val/ens_std"] = ens_std

        pl_module.log_dict(logged, rank_zero_only=False, sync_dist=False)

        if trainer.is_global_zero:
            print(
                f"\n[Epoch {trainer.current_epoch}] "
                f"Ensemble MSE: {ens_mse:.4f}  "
                f"Avg Member MSE: {avg_member_mse:.4f}  "
                f"Gain: {(avg_member_mse - ens_mse):+.4f}"
            )

        self._preds.clear()
        self._targets.clear()


class PerRankBestCheckpoint(Callback):
    """Save each rank's model whenever a monitored metric improves.

    Unlike :class:`PerRankCheckpoint` (which saves only at the final epoch),
    this callback tracks the best value of a given metric and saves a
    per-rank checkpoint whenever it improves.  All ranks save independently
    to rank-specific filenames so files never collide.

    Useful for joint early stopping: monitor ``"val/ens_nll"`` or
    ``"val/ens_mse"`` (logged by :class:`EnsembleMetrics` /
    :class:`RegressionEnsembleMetrics`) to capture each rank's weights at the
    ensemble-optimal epoch.

    The monitor string may contain ``{rank}`` which is resolved to the
    actual rank at runtime, allowing per-member metrics such as
    ``"val/member_{rank}_nll"`` to be monitored independently by each rank.

    Checkpoints are saved as ``{dirpath}/best-rank={rank}.pt`` and contain::

        {"epoch": int, "score": float, "state_dict": OrderedDict}

    Args:
        monitor: Metric key to watch (logged by a callback or the model).
            May contain ``{rank}`` to select a rank-specific metric.
        dirpath: Directory to save checkpoint files.
        mode: ``"min"`` (default) or ``"max"``.

    Example::

        PerRankBestCheckpoint(monitor="val/ens_nll", dirpath="checkpoints/ens_best")
        PerRankBestCheckpoint(monitor="val/member_{rank}_nll", dirpath="checkpoints/member_best")
    """

    def __init__(self, monitor: str, dirpath: str | Path = "checkpoints", mode: str = "min"):
        self.monitor = monitor
        self.dirpath = Path(dirpath)
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.mode = mode
        self._best = float("inf") if mode == "min" else float("-inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        rank = trainer.global_rank
        monitor_key = self.monitor.replace("{rank}", str(rank))
        val = trainer.callback_metrics.get(monitor_key)
        if val is None:
            return

        score = val.item()
        improved = (score < self._best) if self.mode == "min" else (score > self._best)
        if not improved:
            return

        self._best = score
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / f"best-rank={rank}.pt"
        torch.save(
            {"epoch": trainer.current_epoch, "score": score, "state_dict": pl_module.state_dict()},
            path,
        )
