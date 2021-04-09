import egg.core as core
from egg.zoo.language_bottleneck.intervention import _hashable_tensor
import numpy as np
from collections import defaultdict


def entropy_list(counts):
    H = 0
    n = float(sum(v for v in counts))

    for freq in counts:
        p = freq / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(messages):
    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0
    return entropy_list(freq_table.values())


class ComputeEntropy(core.Callback):
    def __init__(self, dataset, is_gs, device, var_length, bin_by):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.is_gs = is_gs
        self.var_length = var_length
        self.bin_by = bin_by
     
    def on_test_end(self, _loss: float, _logs: core.Interaction, _epoch: int):
        game = self.trainer.game
        game.eval()
        entropy = self.compute_entropy(game)
        _logs.aux.update(entropy)
        game.train()

    def compute_entropy(self, game):
        interactions = \
            core.dump_interactions(game, self.dataset, gs=self.is_gs, device=self.device,
                                      variable_length=self.var_length)
        binned_messages = defaultdict(list)
        for i in range(interactions.size):
            msg = interactions.message[i] 
            if self.bin_by >= 0:
                bin_ = interactions.sender_input[i, self.bin_by].item()
            else:
                bin_ = 0  # no bin
            binned_messages[bin_].append(msg)

        entropy_messages = {}
        for bin_, msgs in binned_messages.items():
            entropy_messages["entropy_" + str(bin_)] = entropy(msgs)
        return entropy_messages
        #  return dict(
        #      codewords_entropy=entropy_messages,
        #  )

class LogNorms(core.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_test_end(self, _loss: float, _logs: core.Interaction, _epoch: int):
        norms = {}
        for name, p in self.model.named_parameters():
            norms['norm_' + name] = p.norm()
        _logs.aux.update(norms)

class LRAnnealer(core.Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, _loss, _logs, _epoch):
        self.scheduler.step()

    def on_test_end(self, _loss, _logs, _epoch):
        for group in self.scheduler.optimizer.param_groups:
            _logs.aux.update({'lr': np.asarray([group['lr']])})
        
