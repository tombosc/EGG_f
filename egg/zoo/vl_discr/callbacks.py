import egg.core as core
from egg.zoo.language_bottleneck.intervention import _hashable_tensor
import numpy as np
from collections import defaultdict
from random import shuffle
import torch
from egg.core.callbacks import InteractionSaver as InteractionSaverBase


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
    def __init__(self, dataset, is_gs, device, var_length, bin_by,
            var_message_length):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.is_gs = is_gs
        self.var_length = var_length
        self.bin_by = bin_by
        # used to log not only entropy, but also message lengths
        self.var_message_length = var_message_length
     
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
        binned_lengths = defaultdict(list)
        for i in range(interactions.size):
            msg = interactions.message[i] 
            bin_ = self.bin_by(interactions.aux['n_necessary_features'][i])
            binned_messages[bin_].append(msg)
            if self.var_message_length:
                L = interactions.message_length[i]
                binned_lengths[bin_].append(L)

        stats = {}
        for bin_, msgs in binned_messages.items():
            stats["entropy_" + str(bin_)] = entropy(msgs)
            if self.var_message_length:
                mean_L = torch.stack(binned_lengths[bin_]).mean().unsqueeze(0)
                stats["length_" + str(bin_)] = mean_L
        return stats
        #  return dict(
        #      codewords_entropy=entropy_messages,
        #  )

class PostTrainAnalysis(core.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_test_end(self, _loss: float, _logs: core.Interaction, _epoch: int):
        self.last_interaction = _logs
        sdr_inputs = self.last_interaction.sender_input
        labels = self.last_interaction.labels
        messages = self.last_interaction.message
        rcv_inputs = self.last_interaction.receiver_input
        group_messages = defaultdict(list)
        group_inputs = defaultdict(list)
        groups = set()
        for j, (m, sdr_i, rcv_i) in enumerate(zip(messages, sdr_inputs, rcv_inputs)):
            group = sdr_i[0].item()
            groups.add(group)
            group_messages[group].append((m, j))
            group_inputs[group].append((rcv_i, j))

        # permute messages and inputs, compute accuracy several times
        results = defaultdict(list)
        self.model.eval()
        with torch.no_grad():
            for i in range(10):
                for g in groups:
                    m = group_messages[g]
                    i = group_inputs[g]
                    shuffle(m)
                    shuffle(i)
                    #  import pdb; pdb.set_trace()
                    rcv_out = self.model.receiver(
                        torch.vstack([e[0] for e in m]),
                        torch.vstack([e[0] for e in i]),
                    )
                    if type(rcv_out) == tuple:  # reinforce
                        rcv_out = rcv_out[0]
                    label = labels[[e[1] for e in i]]
                    pred_y = (rcv_out > 0.5).long()
                    acc = (pred_y == label).float().mean(0)
                    results[g].append(acc)
        shuffle_accuracy_results = {}
        for g, l in results.items():
            mean = np.vstack(l).mean(0)
            for bit, m in enumerate(mean):
                field_name = 'shuflacc_'+ str(g) + '_' + str(bit)
                shuffle_accuracy_results[field_name] = m
            print("acc{} = {}".format(g, mean.tolist()))
        _logs.aux.update(shuffle_accuracy_results)

        #  import pdb; pdb.set_trace()


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
        
class InteractionSaver(InteractionSaverBase):
    def __init__(
        self,
        train_epochs=None,
        test_epochs=None,
        folder_path="./interactions",
        save_early_stopping=False,
    ):
        super(InteractionSaver, self).__init__(train_epochs, test_epochs,
                                               folder_path)
        self.save_early_stopping = save_early_stopping

    def on_early_stopping(self, train_loss, train_interaction, epoch,
                          validation_loss, validation_interaction):
        self.dump_interactions(train_interaction, "train", epoch, self.folder_path)
        self.dump_interactions(validation_interaction, "validation", epoch, self.folder_path)
        
