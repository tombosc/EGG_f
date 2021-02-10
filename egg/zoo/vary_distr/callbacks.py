from egg.core.callbacks import Callback
from egg.core import Interaction
import pathlib
import torch
import os
import json

class InteractionSaver(Callback):
    def __init__(self, exp_dir: str, every_epochs: int):
        self.exp_dir = exp_dir
        self.every_epochs = every_epochs

    @staticmethod
    def dump_interactions(
        logs: Interaction, mode: str, epoch: int, dump_dir: str = "./interactions"
    ):
        dump_dir = pathlib.Path(dump_dir) / mode
        dump_dir.mkdir(exist_ok=True, parents=True)
        torch.save(logs, dump_dir / f"interactions_epoch{epoch}")

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if (epoch % self.every_epochs) == 0:
            self.dump_interactions(logs, "validation", epoch, self.exp_dir)

class LRScheduler(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.scheduler.step()

class FileJsonLogger(Callback):
    def __init__(self, exp_dir, filename, print_train_loss=False):
        self.print_train_loss = print_train_loss
        self.abs_path = os.path.join(exp_dir, filename)
        print("Will log to", self.abs_path)

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))
        output_message = json.dumps(dump)
        with open(self.abs_path, 'a') as f:
            f.write(output_message + "\n")

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)
