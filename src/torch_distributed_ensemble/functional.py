import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_independent_worker(base_seed=42, backend=None):
    """Initialize a distributed worker for independent ensemble training.

    Each worker gets a unique seed (base_seed + rank) so models diverge.
    No gradient synchronization is set up — workers train independently.

    Args:
        base_seed: Base random seed. Each rank gets ``base_seed + rank``.
        backend: PyTorch distributed backend. Defaults to ``"nccl"`` when
            CUDA is available, ``"gloo"`` otherwise.
    """
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)  # ty: ignore[possibly-missing-attribute]
    rank = dist.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = dist.get_world_size()  # ty: ignore[possibly-missing-attribute]
    if backend == "nccl":
        torch.cuda.set_device(rank)
    torch.manual_seed(base_seed + rank)
    return rank, world_size


def gather_ensemble_metrics(local_tensor):
    """All-gather a tensor from every worker and stack the results.

    Returns a tensor of shape (world_size, *local_tensor.shape).
    """
    world_size = dist.get_world_size()  # ty: ignore[possibly-missing-attribute]
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)  # ty: ignore[possibly-missing-attribute]
    return torch.stack(gathered)


def load_ensemble(dirpath, model_fn, device="cpu", pattern="best-rank={rank}.pt"):
    """Load saved per-rank checkpoints into a list of models.

    Scans *dirpath* for files whose names match *pattern* (where ``{rank}``
    is a placeholder for the integer rank), sorts them by rank, and loads
    each into a fresh model instance created by *model_fn*.

    Compatible with checkpoints saved by :class:`PerRankBestCheckpoint`
    (``best-rank={rank}.pt``) and :class:`PerRankCheckpoint`
    (``final-rank={rank}.ckpt``).  Files containing a dict with a
    ``"state_dict"`` key are unwrapped automatically; otherwise the file is
    assumed to be a raw state dict.

    Args:
        dirpath: Directory containing checkpoint files.
        model_fn: Zero-argument callable that returns a fresh ``nn.Module``.
        device: Device to move models to (default ``"cpu"``).
        pattern: Filename pattern with ``{rank}`` placeholder.

    Returns:
        List of ``nn.Module`` instances in eval mode, sorted by rank.

    Example::

        models = load_ensemble("checkpoints/", model_fn=MyModel)
        mean, std = ensemble_predict(models, x)
    """
    dirpath = Path(dirpath)
    regex = re.compile("^" + re.escape(pattern).replace(r"\{rank\}", r"(\d+)") + "$")

    matches: list[tuple[int, Path]] = []
    for p in dirpath.iterdir():
        m = regex.match(p.name)
        if m:
            matches.append((int(m.group(1)), p))

    if not matches:
        raise FileNotFoundError(f"No checkpoint files matching pattern {pattern!r} found in {dirpath}")

    matches.sort(key=lambda t: t[0])

    models: list[nn.Module] = []
    for _, ckpt_path in matches:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model = model_fn()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    return models


@torch.no_grad()
def ensemble_predict(models, x, aggregate="mean"):
    """Run inference with all ensemble members and aggregate predictions.

    Args:
        models: List of ``nn.Module`` instances, all on the same device as *x*.
        x: Input tensor.
        aggregate: ``"mean"`` (default) returns a ``(mean, std)`` tuple where
            each tensor has the same shape as a single model's output.
            ``"none"`` returns the raw ``(M, *output_shape)`` stack.

    Returns:
        If *aggregate* is ``"mean"``: ``(mean, std)`` tuple.
        If *aggregate* is ``"none"``: tensor of shape ``(M, *output_shape)``.

    Example::

        mean, std = ensemble_predict(models, x)          # uncertainty estimation
        raw = ensemble_predict(models, x, aggregate="none")  # full member predictions
    """
    preds = torch.stack([model(x) for model in models])
    if aggregate == "none":
        return preds
    return preds.mean(dim=0), preds.std(dim=0)
