from egg.core.callbacks import Callback
from egg.core import Interaction
import pathlib
import torch

class InteractionSaver(Callback):
    def __init__(
        self,
        exp_dir: str 
    ):
        self.exp_dir = exp_dir

    @staticmethod
    def dump_interactions(
        logs: Interaction, mode: str, epoch: int, dump_dir: str = "./interactions"
    ):
        dump_dir = pathlib.Path(dump_dir) / mode
        dump_dir.mkdir(exist_ok=True, parents=True)
        torch.save(logs, dump_dir / f"interactions_epoch{epoch}")

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.dump_interactions(logs, "validation", epoch, self.exp_dir)
