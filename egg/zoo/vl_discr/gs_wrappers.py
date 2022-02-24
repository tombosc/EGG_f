from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical, Categorical

from egg.core.interaction import LoggingStrategy

class SenderReceiverRnnGSST(nn.Module):
    """ Straight-through ONLY adapter!
    """

    def __init__(
        self,
        sender,
        receiver,
        loss,
        length_cost=0.0,
        ada_len_cost_thresh=0,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of 
            (1) a loss tensor of shape (batch size, 1) 
            (2) another loss tensor of shape (batch size, 1)
            (3) the dict with auxiliary information of the same shape.
          The loss (1) and (2) will be minimized during training. Importantly,
          (1) and (2) are accumulated differently: (1) is only considered once the eos token has been
          emitted whereas (2) is accumulated until the eos token has been
          emitted. loss and the auxiliary information aggregated over
          all batches in the dataset. 
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.

        """
        super(SenderReceiverRnnGSST, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )
        self.ada_len_cost_thresh = ada_len_cost_thresh

    def forward(self, sender_input, labels, receiver_input=None):
        message, distribs = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        aux_info = {}
        z = 0.0
        for step in range(receiver_output.size(1)):
            #  print("STEP", step, len(distribs))
            if step != (receiver_output.size(1) - 1):
                assert(distribs[step] is not None)
            step_loss, cumul_loss, step_aux, cumul_aux = self.loss(
                sender_input,
                message[:, step, ...],
                distribs[step],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
            )
            eos_mask = message[:, step, 0]

            add_mask = eos_mask * not_eosed_before
            z += add_mask
            if self.ada_len_cost_thresh > 0:
                length_loss_coef = step_loss < self.ada_len_cost_thresh
            else:
                length_loss_coef = 1
            #  import pdb; pdb.set_trace()
            loss += (
                step_loss * add_mask + 
                cumul_loss * (not_eosed_before) +
                (self.length_cost * add_mask * length_loss_coef * (1.0 + step))
            )
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)
            for name, value in cumul_aux.items():
                prev_V = aux_info.get(name, None)
                if prev_V is not None:
                    aux_info[name] += value * (not_eosed_before)
                else:
                   aux_info[name] = value * (not_eosed_before)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
            step_loss * not_eosed_before
            + self.length_cost * (step + 1.0) * not_eosed_before
        )
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)
        aux_info["length"] = expected_length
        mean_probs = []
        for d in distribs:
            if d:
                mean_probs.append(d.probs.mean(0))
        marginal = torch.stack(mean_probs).mean(0)
        aux_info["H_m"] = Categorical(probs=marginal).entropy().unsqueeze(0)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=expected_length.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction
