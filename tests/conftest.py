import pytest
import torch


def pytest_collection_modifyitems(config, items):
    gpu_count = torch.cuda.device_count()

    skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
    skip_multigpu = pytest.mark.skip(reason=f"Need 2+ GPUs, have {gpu_count} (request more via srun --gres=gpu:N)")

    for item in items:
        if "gpu" in item.keywords and gpu_count < 1:
            item.add_marker(skip_gpu)
        if "multigpu" in item.keywords and gpu_count < 2:
            item.add_marker(skip_multigpu)
