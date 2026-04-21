from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer.states import TrainerFn


class DistributedEnsembleStrategy(DDPStrategy):
    """DDP strategy that skips weight synchronization.

    Each GPU trains its own independent model. Gradient all-reduce and
    weight broadcasting are disabled, but ``all_gather`` still works for
    cross-worker metric communication.
    """

    def setup(self, trainer):
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        assert self.model is not None
        self.precision_plugin.convert_module(self.model)
        self.model_to_device()
        if trainer.state.fn == TrainerFn.FITTING:
            self.configure_ddp()
            self.setup_optimizers(trainer)
        # else: intentionally skip _sync_module_states — each rank keeps its own weights
        self.setup_precision_plugin()
        if trainer.state.fn == TrainerFn.FITTING:
            from lightning.fabric.utilities.optimizer import _optimizers_to_device

            _optimizers_to_device(self.optimizers, self.root_device)

    def configure_ddp(self):
        self.model = self.model.to(self.root_device)  # ty: ignore[unresolved-attribute]

    def reduce(self, tensor, group=None, reduce_op=None):
        return tensor

    def broadcast(self, obj, src=0):
        return obj

    def save_checkpoint(self, checkpoint, filepath, storage_options=None):
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        # Each rank decides independently; the base class sums across ranks and
        # collapses local True to False because our reduce is a no-op.
        return decision

    def remove_checkpoint(self, filepath) -> None:
        # Per-rank files: every rank must clean up its own, not just rank 0.
        self.checkpoint_io.remove_checkpoint(filepath)
