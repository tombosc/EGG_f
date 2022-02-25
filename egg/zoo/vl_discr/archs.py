import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from egg.core.gs_wrappers import gumbel_softmax_sample, RelaxedEmbedding
import egg.core as core
import math
from egg.core.interaction import LoggingStrategy
from egg.core.util import find_lengths
from dataclasses import dataclass
from collections import namedtuple
from simple_parsing.helpers import Serializable
import unittest

def eos_mask(msg):
    n_D = len(msg.size())
    assert(n_D == 2 or n_D == 3)
    L = find_lengths(msg, one_hot_encoded=(n_D == 3))
    bs, max_len = msg.size()[:2]
    mask = (torch.arange(max_len).unsqueeze(0).expand(bs, -1).to(msg.device) >=
                L.unsqueeze(1))
    return mask

class FixedPositionalEmbeddings(nn.Module):
    """ As in Vaswani 2017:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, batch_first, pos_max=100):
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        assert(d_model % 2 == 0)
        arange = torch.arange(0, d_model//2).double()
        denom = torch.exp(torch.log(torch.tensor([10000.])) * 2 * arange / self.d_model)
        arange_pos = torch.arange(0, pos_max).double()
        pre_sin = torch.mm(arange_pos.view(pos_max, 1),
                           1 / denom.view(1, d_model // 2))
        sin = torch.sin(pre_sin)
        cos = torch.cos(pre_sin)
        # (L, d)
        self.emb = torch.cat((sin, cos), 1).float()

    def forward(self, x):
        self.emb = self.emb.to(x.device)
        if self.batch_first:
            bs, L, d = x.size()
            return x + self.emb[:L,:].expand((bs, L, d))
        else:
            L, bs, d = x.size()
            return x + self.emb[:L,:].unsqueeze(1).expand((L, bs, d))


@dataclass
class Hyperparameters(Serializable):
    sender_nlayers: int = 2
    receiver_nlayers: int = 1
    sender_hidden: int = 200  # size of hidden layer of Sender (default: 10)
    #  receiver_hidden: int = 10  # size of hidden layer of Receiver (default: 10)
    sender_cell: str = 'tfm'
    receiver_cell: str = 'tfm'
    dropout: float = 0.1
    sender_emb: int = 32  # size of embeddings of Sender (default: 10)
    #  receiver_emb: int = 10  # size of embeddings of Receiver (default: 10)
    max_len: int = 3
    vocab_size: int = 64
    mode: str = 'gs'
    temperature: float = 1.0
    # these are more like optimisation params, but...
    ada_len_cost_thresh: float = 0.0
    free_symbols: int = 0
    length_cost: float = 0.0
    initial_target_bias: float = 10.0

def make_circulant(v):
    """ Return circulant matrix out of vector.
    """
    n = v.size(0)
    M = torch.zeros((n, n))
    for i in range(n):
        M[i, i:] = v[:n-i]
        M[i, :i] = v[n-i:]
    return M.to(v.device)


def exponential_distance_vector(n, k):
    assert(k>0)
    # take exponential of distance
    u = (torch.arange(n).float() * k).exp()
    # take the negative, as it is going to only *discourage* attending to
    # far-away positions, not *encourage* close positions
    return - u + 1


class Embedder(nn.Module):
    """ Turn a discrete 2D matrix encoding feature of objects into a continuous
    2D matrix
    """
    def __init__(self, dim_emb, n_properties, n_values, n_max_distractors):
        super(Embedder, self).__init__()
        self.value_embedding = nn.Embedding(1 + n_values * n_properties, dim_emb)
        #  self.transform = nn.Sequential(
        #      nn.Linear(n_properties * dim_emb, dim_emb),
        #      nn.ReLU(),
        #      nn.LayerNorm(dim_emb),
        #  )
        self.mark_absent = nn.Parameter(torch.randn(size=(1, 1, dim_emb)))
        self.n_properties = n_properties
        self.idx_offset = nn.Parameter(torch.arange(0, n_properties) *
                n_values, requires_grad=False)


    def forward(self, x, mask_features=None):
        bs, n_th_roles, n_attr = x.size()
        obj_emb = self.value_embedding(x + self.idx_offset)
        if mask_features is not None:
            one_H = F.one_hot(mask_features + 1,
                    num_classes=self.n_properties+1)
            one_H = ~(one_H.sum(1)[:, 1:].unsqueeze(1).unsqueeze(3).bool())  # bs, 1, n_prop, 1
            obj_emb = obj_emb.masked_fill(one_H, 0)
            obj_emb = obj_emb.sum(2) / one_H.sum(2)
            # vector of size n_properties with 1 indicates feature is to be sent
        else:
            obj_emb = obj_emb.sum(2) / self.n_properties
        #  reshaped = obj_emb.view(bs, n_th_roles, -1)
        #  obj_emb = self.transform(reshaped)
        padding = (x.sum(2) == 0).unsqueeze(2)
        # this overrides the previous transformations/embeddings of NA values.
        obj_emb = obj_emb.masked_fill(padding, 0)
        obj_emb += padding * self.mark_absent
        return obj_emb, padding.squeeze(2)

class Sender(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout,
            n_properties, n_values, n_max_distractors, 
            initial_target_bias, 
            max_len,
            n_layers=3, n_head=8):
        super(Sender, self).__init__()
        self.max_len = max_len
        self.embedder = Embedder(dim_emb, n_properties, n_values, n_max_distractors)
        self.vocab_size = vocab_size
        activation = 'gelu'
        encoder_layer = nn.TransformerEncoderLayer(dim_emb, n_head, dim_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # embedding for marking positions to send
        self.mark_target = nn.Parameter(torch.randn(size=(dim_emb,)))
        K = n_max_distractors + 1
        mask_bias = torch.zeros(size=(K,K)).float()
        mask_bias[:, 0] += initial_target_bias
        self.mask_bias = nn.Parameter(mask_bias)
        self.predict_n_necessary = nn.Sequential(
            nn.Linear(dim_emb*2, 5),
        )
        self.feature_pos_embedding = nn.Embedding(1+n_properties, dim_emb)

    def forward(self, x_and_features):
        x, features = x_and_features
        #  z = self.feature_pos_embedding(features+1).mean(1).unsqueeze(1)
        x, mask = self.embedder(x, mask_features=features)
        x[:, 0, :] += self.mark_target
        # debugging: return only the first object without transformation
        #  dbg_mask = torch.zeros((x.size(0), 1)).bool().to(x.device)
        #  return x[:, 0, :].unsqueeze(0), dbg_mask
        y = self.encoder(x.transpose(0, 1),
                mask=self.mask_bias.to(x.device), src_key_padding_mask=mask)
        pred_n = self.predict_n_necessary(
            torch.cat((y.max(0)[0], y.mean(0)), axis=1).detach()
        )
        return y, mask, pred_n

class Receiver(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout,
            n_properties, n_values, n_max_distractors,
            max_len,
            n_layers=3, n_head=8
            ):
        super(Receiver, self).__init__()
        self.msg_embedding = RelaxedEmbedding(vocab_size, dim_emb)
        self.pos_msg_embedding = FixedPositionalEmbeddings(dim_emb,
            batch_first=True)
        # them all in one matrix, we need to add offsets
        self.embedder = Embedder(dim_emb, n_properties, n_values, n_max_distractors)
        activation = 'gelu'
        self.tfm = nn.Transformer(
            d_model=dim_emb,
            nhead=n_head,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
        )
        self.out_log_prob = nn.Linear(dim_emb, 1)
        self.n_max_distractors = n_max_distractors
        self.n_properties = n_properties
        self.n_values = n_values


    def forward(self, msg, x):
        embed_msg = self.msg_embedding(msg)
        embed_msg = self.pos_msg_embedding(embed_msg)
        x, mask = self.embedder(x)
        msg_padding = eos_mask(msg)
        embed_msg = embed_msg.masked_fill(msg_padding.unsqueeze(2), 0.)
        y = self.tfm(src=embed_msg.transpose(0, 1),
                     tgt=x.transpose(0, 1),
                     src_mask=None,
                     src_key_padding_mask=msg_padding,
                     memory_key_padding_mask=msg_padding,
                     tgt_key_padding_mask=mask,
        )
        bs = msg.size(0)
        obj_pred = self.out_log_prob(y).transpose(0, 1).squeeze()
        return obj_pred


class TransformerSenderGS(nn.Module):
    """ Code copied from reinforce_wrappers and adapted for Gumbel-Softmax.
    """
    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        max_len,
        num_layers,
        num_heads,
        hidden_size,
        temperature,
        dropout,
        generate_style="standard",
        causal=True,
    ):
        """
        :param agent: the agent to be wrapped, returns the "encoder" state vector, which is the unrolled into a message
        :param vocab_size: vocab size of the message
        :param embed_dim: embedding dimensions
        :param max_len: maximal length of the message (including <eos>)
        :param num_layers: number of transformer layers
        :param num_heads: number of attention heads
        :param hidden_size: size of the FFN layers
        :param causal: whether embedding of a particular symbol should only depend on the symbols to the left
        :param generate_style: Two alternatives: 'standard' and 'in-place'. Suppose we are generating 4th symbol,
            after three symbols [s1 s2 s3] were generated.
            Then,
            'standard': [s1 s2 s3] -> embeddings [[e1] [e2] [e3]] -> (s4 = argmax(linear(e3)))
            'in-place': [s1 s2 s3] -> [s1 s2 s3 <need-symbol>] \
                                   -> embeddings [[e1] [e2] [e3] [e4]] \
                                   -> (s4 = argmax(linear(e4)))
        """
        super(TransformerSenderGS, self).__init__()
        self.agent = agent
        assert generate_style in ["standard", "in-place"]
        self.generate_style = generate_style
        self.causal = causal

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.temperature = temperature
        activation = 'gelu'
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads,
                hidden_size, dropout, activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.embedding_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.special_symbol_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed_tokens = FixedPositionalEmbeddings(embed_dim, batch_first=False)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        self.embed_scale = math.sqrt(embed_dim)
        self.test_time_sampling_ = False

    def set_test_time_sampling(self, val):
        assert(type(val) == bool)
        self.test_time_sampling_ = val

    def generate_standard(self, encoder_state, encoder_state_mask):
        batch_size = encoder_state.size(1)
        device = encoder_state.device

        sequence = []
        logits = []
        distribs = []

        special_symbol = (
            self.special_symbol_embedding.expand(1, batch_size, -1).to(device)
        )
        input_no_pos = special_symbol

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(step + 1, step + 1).byte(), diagonal=1
                ).to(
                    device
                ).bool()  # noqa: E226
                #  attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            else:
                attn_mask = None

            input = self.pos_embed_tokens(input_no_pos)

            output = self.decoder(
                input, encoder_state, tgt_mask=attn_mask,
                memory_key_padding_mask=encoder_state_mask,
            )
            step_logits = self.embedding_to_vocab(output[-1])
            distrib, sample = gumbel_softmax_sample(
                    step_logits, self.temperature, self.training, True,
                    self.test_time_sampling_,
            )
            distribs.append(distrib)
            sequence.append(sample)
            new_embedding = torch.matmul(sample, self.embed_tokens.weight) * self.embed_scale
            input_no_pos = torch.cat([input_no_pos, new_embedding.unsqueeze(dim=0)], dim=0)
        return sequence, distribs

    def generate_beam_search(self, encoder_state, encoder_state_mask):
        beam_S = self.beam_size
        _, batch_S, embed_dim = encoder_state.size()
        device = encoder_state.device

        logits = []
        distribs = []

        special_symbol = (
            self.special_symbol_embedding.expand(1, batch_S * beam_S, -1).to(device)
        )
        input_no_pos = special_symbol

        # Can't treat special symbol as part of the sequence, since it is not
        # in the embedding matrix. 
        def embed_beams(beams):
            symbols = []
            for beams_row in beams:
                symbols.extend([beam.partial_msg for beam in beams_row]) 
            symbols = torch.tensor(symbols)  # (batch_S, beam_S)
            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            return torch.cat([special_symbol, new_embedding.transpose(0, 1)], dim=0)
        def get_final_sequence(beams):
            symbols = []
            for beams_row in beams:
                symbols.extend([beam.partial_msg for beam in beams_row]) 
            symbols = torch.tensor(symbols)[::beam_S]  # (batch_S, beam_S)
            #  return symbols.transpose(0, 1)
            vocab_S = self.embed_tokens.weight.size(0)
            one_H = torch.nn.functional.one_hot(symbols, num_classes=vocab_S)
            return one_H.transpose(0, 1).float()

        # init beams
        beams = []  # list of list of (sequence of symbols, log proba)
        BeamElement = namedtuple('BeamElement', ['partial_msg', 'log_proba'])
        for i in range(batch_S):
            beams_for_i = [BeamElement(partial_msg=(), log_proba=0.0) for _ in range(beam_S)]
            beams.append(beams_for_i)

        log_prob_beam = torch.zeros((batch_S * beam_S))
        # expand inputs of tfm for beam search
        encoder_state = encoder_state.unsqueeze(2).expand(-1, -1, beam_S, -1)
        encoder_state = encoder_state.reshape(-1, beam_S * batch_S, embed_dim)
        if encoder_state_mask is not None:
            encoder_state_mask = encoder_state_mask.unsqueeze(1).expand(-1, beam_S, -1)
            encoder_state_mask = encoder_state_mask.reshape(beam_S*batch_S, -1)

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(step + 1, step + 1).byte(), diagonal=1
                ).to(device).bool()  # noqa: E226
            else:
                attn_mask = None

            input = self.pos_embed_tokens(input_no_pos)
            
            output = self.decoder(
                input, encoder_state, tgt_mask=attn_mask,
                memory_key_padding_mask=encoder_state_mask,
            )
            logits = self.embedding_to_vocab(output[-1])  # (bs x B, V)
            logits = logits / self.temperature
            step_log_p = logits - torch.logsumexp(logits, 1, keepdim=True)
            log_prob_topk, symbols_topk = step_log_p.topk(beam_S, dim=1)
            new_beams = []
            for i, beams_row in enumerate(beams):
                # iterate of each element of the batch size
                # for each element, we're going to select beam_S best beams
                # among beam_S * beam_S stored in candidates
                begin = i*beam_S
                end = (i+1)*beam_S
                j = 0
                candidates = set()
                for row_log_p, row_symbols in zip(
                        log_prob_topk[begin:end], symbols_topk[begin:end]
                    ):
                    pre_beam = beams_row[j]
                    prev_symbols = pre_beam.partial_msg
                    for log_p, sym in zip(row_log_p, row_symbols):
                        real_log_p = log_p.item()
                        real_sym = sym.item()
                        if len(prev_symbols) and prev_symbols[-1] == 0:  # eos
                            real_log_p = 0
                            real_sym = 0
                        candidates.add(
                            BeamElement(
                                partial_msg=prev_symbols + (real_sym,),
                                log_proba = real_log_p + pre_beam.log_proba,
                            )
                        )
                    j += 1
                #  if step == 3:
                #      import pdb; pdb.set_trace()
                selected = sorted(candidates, key=lambda f: f.log_proba, reverse=True)[:beam_S]
                new_beams.append(selected)
            beams = new_beams
            input_no_pos = embed_beams(beams)
            #  distribs.append(distrib)
        sequence = get_final_sequence(beams)
        sequence = [s for s in sequence]
        return sequence, distribs


    def evaluate_proba_standard(self, sender_input, msgs, has_max_len):
        """ if we use this function on messages that are sampled using 
        functions from this class, we get msgs like 
        [sos, msg(0), ..., msg(max_len), eos] 
        where sos and eos are start and end of sentence special tokens.
        The sampling method implies that after max_len - 1 tokens were
        emitted, eos has probability 1. But when has_max_len=False,
        then we evaluate probabilities under a slightly difference probability
        distribution where sequence lengths are potentially inifinite.
        """
        encoder_state, encoder_state_mask = self.agent(sender_input)
        bs = encoder_state.size(1)
        device = encoder_state.device

        sequence = []
        logits = []
        distribs = []

        embedded_msgs = self.embed_tokens(msgs.transpose(0, 1)) * self.embed_scale

        special_symbol = (
            self.special_symbol_embedding.expand(1, bs, -1).to(device)
        )
        input_no_pos = torch.cat((special_symbol, embedded_msgs), 0)
        input = self.pos_embed_tokens(input_no_pos)
        L_size = input.size(0)

        # !always with causal option here!
        # at train time, we don't need it: in a for loop, we predict, embed,
        # concatenate, until the sequence is complete, and always use the last
        # token to make the prediction. Here, we use all the tokens at once 
        # (not within a for loop) to predict all the next words at once, so 
        # we need this mask.
        attn_mask = torch.triu(  # upper-tri w/o diagonal
            torch.ones(L_size, L_size), diagonal=1,
        ).to(device).bool()  # noqa: E226

        output = self.decoder(
            input, encoder_state, tgt_mask=attn_mask,
            memory_key_padding_mask=encoder_state_mask,
        )

        logits = self.embedding_to_vocab(output)
        logits = logits / self.temperature
        # normalize
        log_probas = logits - torch.logsumexp(logits, 2, keepdim=True)
        # for an input [sos, a, b, c, eos, eos, eos],
        # the 1st output should predict a, the 2nd b, etc. So we discard the
        # last one
        log_probas = log_probas[:-1]  # L, bs, V
        V = log_probas.size(-1)
        flat_log_probas = log_probas.transpose(0, 1).reshape((-1, V))
        flat_msg = msgs.view((-1))
        ar = torch.arange(flat_log_probas.size(0))
        sel_log_probas = flat_log_probas[ar, flat_msg]
        sel_log_probas = sel_log_probas.view((bs, L_size-1))
        L = find_lengths(msgs, one_hot_encoded=False)
        mask = (torch.arange(L_size-1).unsqueeze(0).expand(bs, -1).to(msgs.device) >=
                    L.unsqueeze(1))
        masked_log_probas = sel_log_probas.masked_fill(mask, 0)
        if has_max_len:
            # here all sequences have maximum length of
            # self.max_len (not counting the eos token)
            # therefore, probability of sequence with length greater than
            # max_len is 0!
            mask_too_long = (L > self.max_len + 1).unsqueeze(1).to(masked_log_probas.device)
            masked_log_probas = masked_log_probas.masked_fill(mask_too_long, float('-inf'))
            # and the eos token in the last position has probability 1!
            after_max_len = (torch.arange(L_size - 1) >=
                    self.max_len).unsqueeze(0).to(masked_log_probas.device)
            masked_log_probas = masked_log_probas.masked_fill(after_max_len, 0.0)
        return masked_log_probas.sum(1)

    def generate_inplace(self, encoder_state):
        raise NotImplementedError()
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = (
            self.special_symbol_embedding.expand(batch_size, -1)
            .unsqueeze(1)
            .to(encoder_state.device)
        )
        output = []
        for step in range(self.max_len):
            input = torch.cat(output + [special_symbol], dim=1)
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(step + 1, step + 1).byte(), diagonal=1
                ).to(
                    device
                )  # noqa: E226
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            else:
                attn_mask = None

            embedded = self.transformer(
                embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask
            )
            step_logits = F.log_softmax(
                self.embedding_to_vocab(embedded[:, -1, :]), dim=1
            )

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)
            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            output.append(new_embedding.unsqueeze(dim=1))

        return sequence, logits, entropy

    def forward(self, x):
        encoder_state, encoder_state_mask, pred_n = self.agent(x)

        if self.generate_style == "standard":
            sequence, distribs = self.generate_standard(encoder_state,
                    encoder_state_mask)
        elif self.generate_style == "beam_search":
            sequence, distribs = self.generate_beam_search(encoder_state,
                    encoder_state_mask)
        elif self.generate_style == "in-place":
            sequence, logits, entropy = self.generate_inplace(encoder_state)
        else:
            assert False, "Unknown generate style"
        sequence = torch.stack(sequence).transpose(1, 0)  # bs, L, vocab_size
        #  logits = torch.stack(logits).permute(1, 0)
        #  entropy = torch.stack(entropy).permute(1, 0)

        eos = torch.zeros_like(sequence[:, 0, :]).to(sequence.device).unsqueeze(1)
        eos[:, :, 0] = 1
        sequence = torch.cat([sequence, eos.long()], dim=1)
        #  logits = torch.cat([logits, zeros], dim=1)
        #  entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, distribs, pred_n

class SenderReceiverTransformerGS(nn.Module):
    """ Straight-through ONLY adapter!
    """

    def __init__(
        self,
        sender,
        receiver,
        loss_objs, 
        length_cost=0.0,
        ada_len_cost_thresh=0,
        free_symbols=0,
        train_logging_strategy = None,
        test_logging_strategy = None,
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
        super(SenderReceiverTransformerGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss_objs = loss_objs
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
        self.free_symbols = free_symbols

    def eval_proba_sender(self, sender_input, msgs, has_max_len=True):
        self.eval()
        return self.sender.evaluate_proba_standard(sender_input, msgs,
                has_max_len)

    def eval_loss_receiver(self, sender_input, labels, receiver_input, msgs,
            ids):
        self.eval()
        receiver_outputs = self.receiver(msgs, receiver_input, ids)

        losses = self.compute_loss(
            sender_input, labels, receiver_input, receiver_outputs, msgs,
            one_hot_message=False, ids=ids,
        )
        return losses, losses['sum'], losses['sum'] - losses['length']

    def compute_loss(self, sender_input, labels, receiver_input,
            receiver_outputs, message, pred_n, one_hot_message):
        loss_objs, aux = self.loss_objs(
            sender_input[0],
            message,
            receiver_input,
            receiver_outputs,
            labels,
        )
        # cross-entropy predict n_necessary
        pred_n_CE = F.cross_entropy(pred_n, labels[0]-1, reduction='none')
        def necessary_to_one_hot(feat):
            A = torch.zeros((feat.size(0), 5))
            for i, (f1, f2) in enumerate(feat):
                if f1 >= 0:
                    A[i, f1] = 1
                if f2 >= 0:
                    A[i, f2] = 1
            return A

        one_hot_necessary = necessary_to_one_hot(labels[3]).to(pred_n.device)
        pred_n_CE = F.binary_cross_entropy_with_logits(pred_n, one_hot_necessary, reduction='none')
        wlc = weighted_length_cost(
            loss_objs, 
            message,
            self.length_cost,
            self.ada_len_cost_thresh, 
            self.free_symbols
        )
        loss = (
            loss_objs +
            wlc +
            0.03 * pred_n_CE.sum(1)
        )
        out = {
           'sum': loss, 
           'objs': loss_objs,
           'w_length_cost': wlc,
           'pred_n_CE': pred_n_CE,
        }
        out.update(aux)
        return out

    def forward(self, sender_input, labels, receiver_input):
        message, distribs, pred_n = self.sender(sender_input)
        # turn all tokens after the 1st eos has been emitted to eos
        L = find_lengths(message, one_hot_encoded=True)
        bs, max_len, _ = message.size()
        mask = (torch.arange(max_len).unsqueeze(0).expand(bs, -1).to(message.device) >=
                    L.unsqueeze(1))
        message[:,:,0].masked_fill_(mask, 1)  # eos
        message[:,:,1:].masked_fill_(mask.unsqueeze(2), 0)  # eos

        receiver_outputs = self.receiver(message, receiver_input)
        loss = self.compute_loss(sender_input, labels, receiver_input,
            receiver_outputs, message, pred_n, one_hot_message=True)
        aux = {}
        aux["length"] = L.float()
        aux["loss_objs"] = loss['objs']
        aux["weighted_length_cost"] = loss['w_length_cost']
        aux["pred_n_CE"] = loss['pred_n_CE']
        aux["acc"] = loss['acc']
        aux["sender_input_to_send"] = sender_input[0].float()
        aux["n_necessary_features"] = labels[0].float()
        aux['msg'] = message.argmax(2).float()  # need floats everywhere in aux
        # I don't want to change the API of interactions so I add it here.
        aux['loss'] = loss['sum']

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        # TODO log the rest of recv in and out
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input[0],
            receiver_input=receiver_input.detach(),
            labels=labels,
            receiver_output=receiver_outputs.detach(),
            message=message.detach(),
            message_length=L.float().detach(),
            aux=aux,
        )
        return loss['sum'].mean(), interaction

def weighted_length_cost(loss_objs, message, thresh, length_coef, 
        free_symbols):
    if thresh > 0:
        length_loss_coef = loss_objs < thresh
    else:
        length_loss_coef = 1

    one_hot_message = (len(message.size()) == 3)
    if one_hot_message:
        unweighted_cost = ((1 - message[:, :, 0]).sum(1) - free_symbols)
    else:
        unweighted_cost = ((message != 0).sum(1) - free_symbols)
    if free_symbols > 0:
        # free_symbols is the # of free symbols besides eos
        unweighted_cost = unweighted_cost * (unweighted_cost > 0)

    return length_coef * length_loss_coef * unweighted_cost

class TesterLoss(unittest.TestCase):
    def test_penalty(self):
        msg_normal = torch.tensor([[1, 2, 3, 4, 0],
            [1, 2, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 2, 3, 0, 0]])
        msg_one_hot = F.one_hot(msg_normal)
        for one_hot in [True, False]:
            if one_hot:
                msg = msg_normal
            else:
                msg = msg_one_hot
            loss_objs = torch.tensor([4.0, 3.0, 1.0, 4.0])
            wlc = weighted_length_cost(loss_objs, msg, 
                thresh=3.5, length_coef=1.0, free_symbols=0)
            print("1", wlc)
            assert(torch.allclose(wlc, torch.tensor([0.0, 2.0, 1.0, 0.0])))
            wlc = weighted_length_cost(loss_objs, msg, 
                thresh=3.5, length_coef=1.0, free_symbols=1)
            print("2", wlc)
            assert(torch.allclose(wlc, torch.tensor([0.0, 1.0, 0.0, 0.0])))
            wlc = weighted_length_cost(loss_objs, msg, 
                thresh=5.0, length_coef=0.5, free_symbols=2)  # inactive thresh
            print("2", wlc)
            assert(torch.allclose(wlc, torch.tensor([1.0, 0.0, 0.0, 0.5])))
        

def load_game(hp, loss, data_cfg):
    sender = Sender(
        dim_emb=hp.sender_emb, dim_ff=hp.sender_hidden,
        vocab_size=hp.vocab_size, dropout=hp.dropout,
        n_properties=data_cfg.n_features,
        n_values=data_cfg.max_value,
        n_max_distractors=data_cfg.max_distractors,
        initial_target_bias=hp.initial_target_bias,
        max_len=hp.max_len,
        n_layers=hp.receiver_nlayers,
        n_head=8,
    )
    sender = TransformerSenderGS(
        agent=sender, vocab_size=hp.vocab_size,
        embed_dim=hp.sender_emb, max_len=hp.max_len,
        num_layers=hp.sender_nlayers,
        num_heads=8, hidden_size=hp.sender_hidden,
        temperature=hp.temperature,
        dropout=hp.dropout,
        causal=False,  # causal shouldn't matter, b/c only use the last token
    )
    receiver = Receiver(
        dim_emb=hp.sender_emb, dim_ff=hp.sender_hidden,
        vocab_size=hp.vocab_size, dropout=hp.dropout,
        n_properties=data_cfg.n_features,
        n_values=data_cfg.max_value,
        n_max_distractors=data_cfg.max_distractors,
        max_len=hp.max_len,
        n_layers=hp.receiver_nlayers,
        n_head=8,
    )
    game = SenderReceiverTransformerGS(sender, receiver,
            loss_objs=loss,
            length_cost = hp.length_cost,
            ada_len_cost_thresh = hp.ada_len_cost_thresh,
            free_symbols = hp.free_symbols,
    )
    return game
