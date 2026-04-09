# torch-distributed-ensemble

Train the members of a deep ensemble in parallel across multiple GPUs.

## Why

I work with deep ensembles. There's no obvious way to train them in parallel in PyTorch, so people usually run members sequentially in one job, or launch separate jobs and combine the results afterwards.

Instead, you can use `torch.distributed` for this: create a process group without DDP, and each worker trains its own model independently while still being able to `all_gather` predictions or metrics when needed. This library wraps that pattern, along with per-rank seeding, per-rank checkpointing, and ensemble metric callbacks. You can probably also do this easily in JAX (`shard_map` etc.), but this is how to do it in PyTorch.

## Usage

With Lightning:

```python
import lightning as L
from torch_distributed_ensemble import (
    DistributedEnsembleStrategy, DistributedSeeder, EnsembleMetrics, PerRankCheckpoint,
)

trainer = L.Trainer(
    accelerator="gpu", devices=4,
    strategy=DistributedEnsembleStrategy(),
    callbacks=[
        DistributedSeeder(base_seed=42),
        EnsembleMetrics(),
        PerRankCheckpoint(dirpath="ckpts"),
    ],
    use_distributed_sampler=False, enable_checkpointing=False,
)
trainer.fit(model, train_loader, val_loader)
```

For plain `torchrun`, use `setup_independent_worker` and `gather_ensemble_metrics`. See [`examples/00_heteroscedastic_regression`](examples/00_heteroscedastic_regression/).

## Examples

A mix of patterns I found useful; some of them relate to my own research. All outputs in this repository were produced on 8 GPUs (one full LUMI node). Suggestions for additional examples are welcome.

| | |
|---|---|
| [`00_heteroscedastic_regression`](examples/00_heteroscedastic_regression/) | aleatoric vs. epistemic uncertainty on a 1D toy |
| [`01_joint_early_stopping`](examples/01_joint_early_stopping/) | stopping CIFAR-10 ensembles at the ensemble NLL optimum |
| [`02_coupled_ssl`](examples/02_coupled_ssl/) | SSL with members pulled toward the running ensemble average |
| [`03_hpo_ensemble`](examples/03_hpo_ensemble/) | HPO targeting ensemble vs. member performance |
| [`04_hts_hit_finding`](examples/04_hts_hit_finding/) | active learning on 101k molecules with ensemble UCB |

## Install

```bash
pip install git+https://github.com/lauritsf/torch-distributed-ensemble
pip install "torch-distributed-ensemble[lightning] @ git+https://github.com/lauritsf/torch-distributed-ensemble"
```

Python 3.12+, PyTorch 2.7+, optional Lightning 2.5+.

## Status

I've run this on two clusters: DTU's Titans (NVIDIA) and LUMI (AMD MI250X). Some brittleness remains. Lightning, SLURM, and GPU count detection get exposed differently from one cluster to the next, and getting things running may still take some environment variable alchemy. If something breaks for you, please open an issue.

## Citation

A `CITATION.cff` is included if you'd like to cite this.

## License

MIT. [Laurits Fredsgaard](https://laurits.me/).
