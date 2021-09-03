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
from simple_parsing.helpers import Serializable

_n_rolesets = 1639


def maybe_FA_expand(flat_attention, mask):
    if flat_attention:
        bs, n_roles = mask.size()
        M = mask.unsqueeze(2).expand(bs, n_roles, 18)
        return M.reshape(bs, n_roles*18)
    return mask

@dataclass
class Hyperparameters(Serializable):
    sender_nlayers: int = 2
    receiver_nlayers: int = 2
    sender_hidden: int = 10  # size of hidden layer of Sender (default: 10)
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
    distance_reg_coef: float = 0.0
    length_cost: float = 0.0
    flat_attention: bool = False
    predict_classical_roles: bool = False

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

    def forward(self, input_):
        roleset, properties, thematic_roles, marked = input_
        role_emb, obj_emb, mask_padding = self.embedder(roleset, properties, thematic_roles)
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
        return self.encoder(x.transpose(0, 1), src_key_padding_mask=mask)

class Receiver(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout, max_len,
            n_thematic_roles, 
            n_layers=3, n_head=8, distance_reg_coef=0.0, predict_roleset=False,
            flat_attention=False, predict_classical_roles=False):
        super(Receiver, self).__init__()
        self.msg_embedding = RelaxedEmbedding(vocab_size, dim_emb)
        self.pos_msg_embedding = nn.Parameter(
            torch.randn((1, max_len+1, dim_emb))
        )
        # them all in one matrix, we need to add offsets
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


    def forward(self, msg, receiver_inputs):
        embed_msg = self.msg_embedding(msg)
        # to debug:
        # embed_msg = torch.zeros_like(embed_msg)
        embed_msg += self.pos_msg_embedding
        roleset, properties, thematic_roles = receiver_inputs
        role_emb, obj_emb, _ = self.embedder(roleset, properties, thematic_roles)
        x = torch.cat((role_emb.unsqueeze(1), obj_emb), 1)
        if self.distance_reg_coef > 0:
            attn_mask = self.distance_matrix.to(x.device)
        else:
            attn_mask = None
        y = self.tfm(embed_msg.transpose(0, 1), x.transpose(0, 1),
                     src_mask=attn_mask)
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

        #  self.transformer = nn.TransformerDecoder(
        #      embed_dim=embed_dim,
        #      max_len=max_len,
        #      num_layers=num_layers,
        #      num_heads=num_heads,
        #      hidden_size=hidden_size,
        #  )

        self.embedding_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.special_symbol_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        self.embed_scale = math.sqrt(embed_dim)

    def generate_standard(self, encoder_state):
        batch_size = encoder_state.size(1)
        device = encoder_state.device

        sequence = []
        logits = []
        distribs = []

        special_symbol = (
            self.special_symbol_embedding.expand(1, batch_size, -1).to(device)
        )
        input = special_symbol

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(step + 1, step + 1).byte(), diagonal=1
                ).to(
                    device
                )  # noqa: E226
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            else:
                attn_mask = None
            if self.distance_reg_coef:
                M = self.distance_matrix[:step+1, :step+1].to(device)
                if attn_mask:
                    attn_mask += M
                else:
                    attn_mask = M

            output = self.decoder(
                input, encoder_state, tgt_mask=attn_mask,
            )
            step_logits = self.embedding_to_vocab(output[-1])
            distrib, sample = gumbel_softmax_sample(
                    step_logits, self.temperature, self.training, True,
                    False,
            )
            # TODO symbol dropout
            # TODO test time sampling

            #  distr = Categorical(logits=step_logits)
            #  sample = sample.long()
            distribs.append(distrib)
            sequence.append(sample)
            #  new_embedding = self.embed_tokens(sample) * self.embed_scale
            new_embedding = torch.matmul(sample, self.embed_tokens.weight) * self.embed_scale
            input = torch.cat([input, new_embedding.unsqueeze(dim=0)], dim=0)

        return sequence, distribs

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
        encoder_state = self.agent(x)

        if self.generate_style == "standard":
            sequence, distribs = self.generate_standard(encoder_state)
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

    def forward(self, sender_input, labels, receiver_input=None):
        message, distribs = self.sender(sender_input)
        # turn all tokens after the 1st eos has been emitted to eos
        L = find_lengths(message, one_hot_encoded=True)
        bs, max_len, _ = message.size()
        mask = (torch.arange(max_len).unsqueeze(0).expand(bs, -1).to(message.device) >=
                    L.unsqueeze(1))
        message[:,:,0].masked_fill_(mask, 1)  # eos
        message[:,:,1:].masked_fill_(mask.unsqueeze(2), 0)  # eos
        receiver_outputs = self.receiver(message, receiver_input)

        if self.loss_rolesets:
            loss_rolesets = self.loss_rolesets(receiver_outputs[0], labels)
        else:
            loss_rolesets = 0
        maybe_packed_losses = self.loss_objs(
            sender_input,
            message,
            distribs,
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
        unweighted_length_cost = (1 - message[:, :, 0]).sum(1)
        weighted_length_cost = (self.length_cost * length_loss_coef *
                unweighted_length_cost)
        loss = (
            loss_rolesets +
            loss_objs +
            loss_roles +
            weighted_length_cost
        )

        aux = {}
        aux["length"] = L.float()
        aux["gram_funcs"] = labels[2].float()
        if not torch.all(labels[3] == 0).item():
            aux["permutation"] = labels[3]
        if self.loss_rolesets:
            aux["loss_rolesets"] = loss_rolesets
        if type(loss_roles) != int:
            aux["loss_classical_roles"] = loss_roles
            aux["loss_classical_roles_D"] = loss_roles_entity_wise
        aux["loss_objs"] = loss_objs
        aux["loss_objs_D"] = loss_objs_entity_wise
        aux["roleset"] = sender_input[0].float()
        aux["weighted_length_cost"] = weighted_length_cost
        aux["sender_input_to_send"] = sender_input[3].float()

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
        return loss.mean(), interaction

def load_game(hp, loss_objs, n_thematic_roles):
    receiver = Receiver(
        dim_emb=hp.sender_emb, dim_ff=hp.sender_hidden,
        vocab_size=hp.vocab_size, dropout=hp.dropout,
        max_len=hp.max_len,
        n_thematic_roles=n_thematic_roles, 
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
        causal=False,  # causal shouldn't matter, b/c only use the last token
        distance_reg_coef=hp.distance_reg_coef,
    )
    game = SenderReceiverTransformerGS(sender, receiver, 
            loss_rolesets=loss_rolesets if hp.predict_roleset else None,
            loss_objs=loss_objs,
            length_cost = hp.length_cost,
            ada_len_cost_thresh = hp.ada_len_cost_thresh,
    )
    return game
