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

_n_rolesets = 1639

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

def maybe_FA_expand(flat_attention, mask):
    if flat_attention:
        bs, n_roles = mask.size()
        M = mask.unsqueeze(2).expand(bs, n_roles, 18)
        return M.reshape(bs, n_roles*18)
    return mask

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
    sender_mask_padded: bool = False
    max_len: int = 3
    vocab_size: int = 64
    mode: str = 'gs'
    predict_roleset: bool = False
    temperature: float = 1.0
    # these are more like optimisation params, but...
    ada_len_cost_thresh: float = 0.0
    free_symbols: int = 0
    distance_reg_coef: float = 0.0
    length_cost: float = 0.0
    flat_attention: bool = False
    predict_classical_roles: bool = False
    version: float = 1.0

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
    2D matrix. If flat_attention:
    """
    def __init__(self, dim_emb, n_thematic_roles, flat_attention=False):
        super(Embedder, self).__init__()
        # we use smaller embeddings for roleset, that we then project.
        dim_small_emb = 10
        # we add 1 to the roleset, b/c there is a "0" dummy roleset if
        # we decide it should be hidden
        self.roleset_embedding = nn.Embedding(_n_rolesets+1, dim_small_emb)
        self.roleset_linear = nn.Linear(dim_small_emb, dim_emb)
        # there are 18 properties per object, 
        # each taking 4 values (0: N/A, 1, 2, 3 for 1, 3, 5 on Likert scale
        # idx_offset is used to take different embeddings for each property (OO
        # variant) or each (property, object) pair (flat variant)
        n_prop = 18
        if flat_attention:
            self.value_embedding = nn.Embedding(4 * n_prop * n_thematic_roles, dim_emb)
            self.idx_offset = torch.arange(0, n_thematic_roles * n_prop) * 4
        else:
            self.value_embedding = nn.Embedding(4 * n_prop, dim_emb)
            self.idx_offset = torch.arange(0, n_prop) * 4
        if not flat_attention:
            self.transform = nn.Sequential(
                nn.Linear(n_prop * dim_emb, dim_emb),
                nn.ReLU(),
                nn.LayerNorm(dim_emb),
            )
        else:
            self.transform = None
        # since each embedding for each attribute is different but we store
        # them all in one matrix, we need to add offsets
        self.pos_role_embedding = nn.Parameter(
            torch.randn((1, dim_emb))
        )
        # the positional embeddings typically used by transformers have
        # different meanings here: in FA variant, they're specific to each
        # role AND property pair; in OO, they're specific to each role.
        if not flat_attention:
            self.pos_obj_embeddings = nn.Parameter(
                torch.randn((1, n_thematic_roles, dim_emb))
            )
        else:
            self.pos_obj_embeddings = nn.Parameter(
                torch.randn((1, n_thematic_roles * n_prop, dim_emb))
            )
        self.flat_attention = flat_attention
        self.n_thematic_roles = n_thematic_roles
        self.mark_absent = nn.Parameter(torch.randn(size=(1, 1, dim_emb)))

    def forward(self, roleset, properties, thematic_roles):
        role_emb = self.roleset_embedding(roleset)
        role_emb = self.roleset_linear(role_emb)
        # (bs, n_th_roles, n_attr, dim_emb)
        bs, n_th_roles, n_attr = properties.size()
        if self.flat_attention:
            P = properties.view(bs, n_th_roles*n_attr)
            obj_emb = self.value_embedding(P + self.idx_offset.to(P.device))  
            obj_emb = obj_emb.view(bs, n_th_roles*n_attr, -1)
        else:
            obj_emb = self.value_embedding(properties +
                self.idx_offset.to(properties.device))  
            if self.transform:
                reshaped = obj_emb.view(bs, n_th_roles, -1)
                obj_emb = self.transform(reshaped)
            else:
                obj_emb = obj_emb.mean(2)

        # input thematic_roles:
        # -1=missing (sender, receiver) or hidden (receiver only),
        # 0=agent, 1=patient, etc.
        # for the receiver, whether an object is padding or hidden makes no
        # difference. Therefore, if the sender transmits info about a single
        # object, it doesn't know what role to expect for this object a priori.
        padding_or_hidden = maybe_FA_expand(
                self.flat_attention,
                thematic_roles == -1).unsqueeze(2)
        # this overrides the previous transformations/embeddings of NA values.
        obj_emb = obj_emb.masked_fill(padding_or_hidden, 0)
        obj_emb += padding_or_hidden * self.mark_absent
        # thematic roles are already "used", since the order of the objects in 
        # the properties matrix already encodes for the role, and we use
        # pos_obj_embeddings.
        obj_emb = obj_emb + self.pos_obj_embeddings
        role_emb = role_emb + self.pos_role_embedding
        return role_emb, obj_emb, padding_or_hidden.squeeze(2)

class Sender(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout, max_len,
            n_thematic_roles, mask_padded,
            n_layers=3, n_head=8, flat_attention=False):
        super(Sender, self).__init__()
        self.max_len = max_len
        self.embedder = Embedder(dim_emb, n_thematic_roles,
                flat_attention=flat_attention)
        self.vocab_size = vocab_size
        activation = 'gelu'
        encoder_layer = nn.TransformerEncoderLayer(dim_emb, n_head, dim_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # embedding for marking positions to send
        self.mark_tosend = nn.Parameter(torch.randn(size=(1, 1, dim_emb)))
        self.flat_attention = flat_attention
        self.mask_padded = mask_padded
        self.mask_inverse_debug_ = False

    def set_mask_inverse_debug(self, val):
        assert(type(val) == bool)
        self.mask_inverse_debug_ = val

    def forward(self, input_):
        roleset, properties, thematic_roles, marked = input_
        role_emb, obj_emb, mask_padding = self.embedder(roleset, properties, thematic_roles)
        if self.mask_inverse_debug_:
            marked = (1-marked)  # TODO this should be a boolean
            mask_padding = ~mask_padding
        # mark positions that need to be sent.
        # we do not mask the other positions, so that the sender can take into
        # account the other arguments that are visible to the receiver in order
        # to decide what to send (pragmatics)
        x = torch.cat((role_emb.unsqueeze(1), obj_emb), 1)

        mask = None  # by default, absent objects are selectable by attention
        if self.mask_padded:
            # in this variant, absent objects are never selected by attention
            # set absent object's mask to 1: ignored by Transformer
            # 0:1: notation to select 0, but keepdim!..
            roleset_dummy_mask = torch.zeros_like(mask_padding[:, 0:1])
            mask = torch.cat((roleset_dummy_mask, mask_padding), 1)

        # mark objects to send with special embedding
        if self.flat_attention:
            marked = marked[:, 1:]  # ignore roleset
            bs, n_arg = marked.size()
            expanded = marked.view(bs, n_arg, 1).expand(bs, n_arg, 18)
            flattened = expanded.reshape(bs, n_arg*18, 1)
            x[:, 1:] += (self.mark_tosend * flattened)
        else:
            x += (self.mark_tosend * marked.unsqueeze(2))
        # mask should be (bs, L), input should be (L, bs, dim)
        return self.encoder(x.transpose(0, 1), src_key_padding_mask=mask), mask

class Receiver(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout, max_len,
            n_thematic_roles, 
            version,
            n_layers=3, n_head=8, distance_reg_coef=0.0, predict_roleset=False,
            flat_attention=False, predict_classical_roles=False):
        super(Receiver, self).__init__()
        self.msg_embedding = RelaxedEmbedding(vocab_size, dim_emb)
        if version == 1.0:
            self.pos_msg_embedding = nn.Parameter(
                torch.randn((1, max_len+1, dim_emb))
            )
        elif version >= 1.1:
            self.pos_msg_embedding = FixedPositionalEmbeddings(dim_emb,
                    batch_first=True)
        # them all in one matrix, we need to add offsets
        self.version = version
        self.embedder = Embedder(dim_emb, n_thematic_roles, flat_attention=flat_attention)
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
        if predict_roleset:
            self.out_roleset = nn.Linear(dim_emb, _n_rolesets)
        else: 
            self.out_roleset = None
        if predict_classical_roles:
            # there needs to be an "absent" role label (ground-truth -1)
            self.out_role_output_dim = n_thematic_roles + 1
        else:
            # else, only predict whether the object is padding or real
            self.out_role_output_dim = 2

        if flat_attention:
            self.out_obj = nn.Linear(dim_emb, 4)
            self.out_role = nn.Linear(2*dim_emb, self.out_role_output_dim * n_thematic_roles)
        else:
            self.out_obj = nn.Linear(dim_emb, 4*18)
            self.out_role = nn.Linear(dim_emb, self.out_role_output_dim)
        self.distance_reg_coef = distance_reg_coef
        if distance_reg_coef > 0:
            v = exponential_distance_vector(max_len+1, distance_reg_coef)
            self.distance_matrix = make_circulant(v)
        self.flat_attention = flat_attention
        self.n_thematic_roles = n_thematic_roles


    def forward(self, msg, receiver_inputs, ids):
        """
        ids: can be used for debugging purposes.
        """
        # RelaxedEmbeddings works for both 1hot enc msgs and discrete matrices
        embed_msg = self.msg_embedding(msg)
        if self.version == 1.0:
            embed_msg += self.pos_msg_embedding
        elif self.version >= 1.1:
            embed_msg = self.pos_msg_embedding(embed_msg)
        roleset, properties, thematic_roles = receiver_inputs
        role_emb, obj_emb, _ = self.embedder(roleset, properties, thematic_roles)
        x = torch.cat((role_emb.unsqueeze(1), obj_emb), 1)
        if self.distance_reg_coef > 0:
            attn_mask = self.distance_matrix.to(x.device)
        else:
            attn_mask = None
        msg_padding = eos_mask(msg)
        embed_msg = embed_msg.masked_fill(msg_padding.unsqueeze(2), 0.)
        y = self.tfm(src=embed_msg.transpose(0, 1),
                     tgt=x.transpose(0, 1),
                     src_mask=attn_mask,
                     src_key_padding_mask=msg_padding,
                     memory_key_padding_mask=msg_padding,
                    )
        roleset_pred = self.out_roleset(y[0]) if self.out_roleset else None
        # role prediction:
        if self.flat_attention:
            max_y = y[1:].max(0)[0]
            avg_y = y[1:].mean(0)
            role_features = torch.cat((max_y, avg_y), dim=1)
            bs = y.size(1)
            role_pred = self.out_role(role_features).view(bs, self.n_thematic_roles,
                    self.out_role_output_dim)
        else:
            role_pred = self.out_role(y[1:]).transpose(0, 1)
        bs = msg.size(0)
        obj_pred = self.out_obj(y[1:]).transpose(0, 1).view(bs, 3, 18, 4)
        # here's how we can use ids to debug and make sure we get the exact
        # same values across calls. we can do the same thing in compute_loss
        #  id_ = 7592  # or whatever
        #  if torch.any(ids == id_):
        #      i = (ids == id_).nonzero()[0,0]
        #      print("RP", role_pred[i][0])
        return roleset_pred, role_pred, obj_pred


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
        version,
        generate_style="standard",
        causal=True,
        distance_reg_coef=0.0,
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
        self.version = version
        assert generate_style in ["standard", "in-place"]
        self.generate_style = generate_style
        self.causal = causal
        self.distance_reg_coef = distance_reg_coef
        if distance_reg_coef > 0:
            v = exponential_distance_vector(max_len+1, self.distance_reg_coef)
            self.distance_matrix = make_circulant(v)

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
        if self.version >= 1.1:
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
            if self.distance_reg_coef:
                M = self.distance_matrix[:step+1, :step+1].to(device)
                if attn_mask:
                    attn_mask += M
                else:
                    attn_mask = M

            if self.version >= 1.1:
                input = self.pos_embed_tokens(input_no_pos)
            else:
                input = input_no_pos

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
        if self.distance_reg_coef:
            # TODO abandonned. remove?
            raise NotImplementedError() 

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

            if self.version >= 1.1:
                input = self.pos_embed_tokens(input_no_pos)
            else:
                input = input_no_pos
            
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
        if self.version >= 1.1:
            input = self.pos_embed_tokens(input_no_pos)
        else:
            input = input_no_pos
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

        if self.distance_reg_coef:
            raise NotImplementedError()

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
        encoder_state, encoder_state_mask = self.agent(x)

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

        return sequence, distribs

class SenderReceiverTransformerGS(nn.Module):
    """ Straight-through ONLY adapter!
    """

    def __init__(
        self,
        sender,
        receiver,
        loss_rolesets,
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
        self.loss_rolesets = loss_rolesets
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
            receiver_outputs, message, one_hot_message, ids):
        if self.loss_rolesets:
            loss_rolesets = self.loss_rolesets(receiver_outputs[0], labels)
        else:
            loss_rolesets = 0
        maybe_packed_losses = self.loss_objs(
            sender_input,
            message,
            receiver_input,
            receiver_outputs[1], 
            receiver_outputs[2],
            labels,
        )
        if type(maybe_packed_losses) == tuple:
            loss_roles_entity_wise, loss_objs_entity_wise = maybe_packed_losses
            loss_roles = loss_roles_entity_wise.sum(1)
        else:
            loss_objs_entity_wise = maybe_packed_losses
            loss_roles = 0

        loss_objs = loss_objs_entity_wise.sum(1)

        if self.ada_len_cost_thresh:
            length_loss_coef = loss_objs < self.ada_len_cost_thresh
        else:
            length_loss_coef = 1

        if one_hot_message:
            unweighted_length_cost = (1 - message[:, :, 0]).sum(1)
        else:
            unweighted_length_cost = (message != 0).sum(1)
        if self.free_symbols > 0:
            # free_symbols is the # of free symbols besides eos
            unweighted_length_cost = unweighted_length_cost * (
                unweighted_length_cost > (self.free_symbols+1))

        weighted_length_cost = (self.length_cost * length_loss_coef *
            unweighted_length_cost)
        loss = (
            loss_rolesets +
            loss_objs +
            loss_roles +
            weighted_length_cost
        )
        out = {'sum': loss, 'rolesets': loss_rolesets, 'roles': loss_roles,
                'roles_D': loss_roles_entity_wise, 'objs': loss_objs,
                'objs_D': loss_objs_entity_wise, 'length': weighted_length_cost,
        }
        return out

    def forward(self, sender_input, labels, receiver_input, ids):
        message, distribs = self.sender(sender_input)
        # turn all tokens after the 1st eos has been emitted to eos
        L = find_lengths(message, one_hot_encoded=True)
        bs, max_len, _ = message.size()
        mask = (torch.arange(max_len).unsqueeze(0).expand(bs, -1).to(message.device) >=
                    L.unsqueeze(1))
        message[:,:,0].masked_fill_(mask, 1)  # eos
        message[:,:,1:].masked_fill_(mask.unsqueeze(2), 0)  # eos

        receiver_outputs = self.receiver(message, receiver_input, ids)
        loss = self.compute_loss(sender_input, labels, receiver_input,
            receiver_outputs, message, one_hot_message=True, ids=ids)
        aux = {}
        aux["length"] = L.float()
        aux["gram_funcs"] = labels[2].float()
        if not torch.all(labels[3] == 0).item():
            aux["permutation"] = labels[3]
        if self.loss_rolesets:
            aux["loss_rolesets"] = loss['rolesets']
        if type(loss['roles']) != int:
            aux["loss_classical_roles"] = loss['roles']
            aux["loss_classical_roles_D"] = loss['roles_D']
        aux["loss_objs"] = loss['objs']
        aux["loss_objs_D"] = loss['objs_D']
        aux["roleset"] = sender_input[0].float()
        aux["weighted_length_cost"] = loss['length']
        aux["sender_input_to_send"] = sender_input[3].float()
        aux['ids'] = ids.float()  # very useful to debug.
        aux['msg'] = message.argmax(2).float()  # need floats everywhere in aux
        # I don't want to change the API of interactions so I add it here.
        aux['loss'] = loss['sum']

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        # TODO log the rest of recv in and out
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input[1],
            receiver_input=receiver_input[1].detach(),
            labels=labels[1],
            receiver_output=receiver_outputs[2].detach(),
            message=message.detach(),
            message_length=L.float().detach(),
            aux=aux,
        )
        return loss['sum'].mean(), interaction

def load_game(hp, loss_objs, n_thematic_roles):
    receiver = Receiver(
        dim_emb=hp.sender_emb, dim_ff=hp.sender_hidden,
        vocab_size=hp.vocab_size, dropout=hp.dropout,
        max_len=hp.max_len,
        n_thematic_roles=n_thematic_roles, 
        version=hp.version,
        n_layers=hp.receiver_nlayers,
        distance_reg_coef=hp.distance_reg_coef,
        predict_roleset=hp.predict_roleset,
        flat_attention=hp.flat_attention,
        predict_classical_roles=hp.predict_classical_roles,
    )
    sender = Sender(
        dim_emb=hp.sender_emb, dim_ff=hp.sender_hidden,
        vocab_size=hp.vocab_size, dropout=hp.dropout,
        max_len=hp.max_len, 
        n_thematic_roles=n_thematic_roles, 
        mask_padded=hp.sender_mask_padded,
        n_layers=hp.sender_nlayers,
        flat_attention=hp.flat_attention,
    )
    sender = TransformerSenderGS(
        agent=sender, vocab_size=hp.vocab_size,
        embed_dim=hp.sender_emb, max_len=hp.max_len,
        num_layers=hp.sender_nlayers,
        num_heads=8, hidden_size=hp.sender_hidden,
        temperature=hp.temperature,
        dropout=hp.dropout,
        version=hp.version,
        causal=False,  # causal shouldn't matter, b/c only use the last token
        distance_reg_coef=hp.distance_reg_coef,
    )
    game = SenderReceiverTransformerGS(sender, receiver, 
            loss_rolesets=loss_rolesets if hp.predict_roleset else None,
            loss_objs=loss_objs,
            length_cost = hp.length_cost,
            ada_len_cost_thresh = hp.ada_len_cost_thresh,
            free_symbols = hp.free_symbols,
    )
    return game
