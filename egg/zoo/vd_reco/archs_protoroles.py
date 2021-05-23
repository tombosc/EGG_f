import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from egg.core.gs_wrappers import gumbel_softmax_sample, RelaxedEmbedding
import egg.core as core
import math
from egg.core.interaction import LoggingStrategy
from egg.core.util import find_lengths

_n_roles = 1639
_n_arg = 3


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
    def __init__(self, dim_emb, dropout, variant=1):
        super(Embedder, self).__init__()
        dim_small_emb = 10
        # we add 1 to the number of roles, b/c there is a "0" dummy role for
        # when it's hidden
        self.role_embedding = nn.Embedding(_n_roles+1, dim_small_emb)
        self.role_linear = nn.Linear(dim_small_emb, dim_emb)
        # there are 18 properties per object, 
        # each taking 4 values (0: N/A, 1, 2, 3 for 1, 3, 5 on Likert scale
        n_prop = 18
        self.obj_embedding = nn.Embedding(4 * n_prop, dim_emb)
        if variant == 1:
            self.transform = nn.Sequential(
                nn.Linear(n_prop * dim_emb, dim_emb),
                nn.ReLU(),
                nn.LayerNorm(dim_emb),
                #  nn.Linear(n_prop * dim_emb, 2*dim_emb),
                #  nn.ReLU(),
                #  nn.Dropout(dropout),
                #  nn.Linear(2*dim_emb, dim_emb),
            )
        else:
            self.transform = None
        # since each embedding for each attribute is different but we store
        # them all in one matrix, we need to add offsets
        self.idx_offset = torch.arange(0, n_prop)
        self.pos_role_embedding = nn.Parameter(
            torch.randn((1, dim_emb))
        )
        self.pos_obj_embeddings = nn.Parameter(
            torch.randn((1, _n_arg, dim_emb))
        )

    def forward(self, roleset, properties):
        role_emb = self.role_embedding(roleset)
        role_emb = self.role_linear(role_emb)
        obj_emb = self.obj_embedding(properties +
                self.idx_offset.to(properties.device))  
        # (bs, n_args, n_attr, dim_emb)
        if self.transform:
            bs, n_args, n_attr, dim_emb = obj_emb.size()
            reshaped = obj_emb.view(bs, n_args, -1)
            obj_emb = self.transform(reshaped)
        else:
            obj_emb = obj_emb.mean(2)
        obj_emb = obj_emb + self.pos_obj_embeddings
        role_emb = role_emb + self.pos_role_embedding
        return role_emb, obj_emb

class Sender(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout, max_len, n_layers=3, n_head=8):
        super(Sender, self).__init__()
        self.max_len = max_len
        self.embedder = Embedder(dim_emb, dropout)
        # hardcoded embedding size that's is small, b/c we have very few data
        # for rolesets, so for stat efficiency, keep it small
        self.vocab_size = vocab_size
        activation = 'gelu'
        encoder_layer = nn.TransformerEncoderLayer(dim_emb, n_head, dim_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # embedding for marking positions to send
        self.mark = nn.Parameter(torch.randn(size=(1, 1, dim_emb)))

    def forward(self, input_):
        roleset, properties, marked = input_
        role_emb, obj_emb = self.embedder(roleset, properties)
        # mark positions that need to be sent.
        # we do not mask the other positions, so that the sender can take into
        # account the other arguments that are visible to the receiver in order
        # to decide what to send (pragmatics)
        x = torch.cat((role_emb.unsqueeze(1), obj_emb), 1)
        x += (self.mark * marked.unsqueeze(2))
        # on the other hand, the mask only masks out absent objects
        mask = torch.zeros_like(x[:, :, 0]).bool()
        mask[:, 1:] = (properties == 0).all(2)  # "pos with true are ignored"
        # mask should be (bs, L), input should be (L, bs, dim)
        return self.encoder(x.transpose(0, 1), src_key_padding_mask=mask)

class Receiver(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, dropout, max_len,
            n_layers=3, n_head=8, distance_reg_coef=0.0):
        super(Receiver, self).__init__()
        self.msg_embedding = RelaxedEmbedding(vocab_size, dim_emb)
        self.pos_msg_embedding = nn.Parameter(
            torch.randn((1, max_len+1, dim_emb))
        )
        # them all in one matrix, we need to add offsets
        self.embedder = Embedder(dim_emb, dropout)
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
        self.out_role = nn.Linear(dim_emb, _n_roles)
        self.out_obj = nn.Linear(dim_emb, 4*18)
        self.distance_reg_coef = distance_reg_coef
        if distance_reg_coef > 0:
            v = exponential_distance_vector(max_len+1, distance_reg_coef)
            self.distance_matrix = make_circulant(v)


    def forward(self, msg, receiver_inputs):
        embed_msg = self.msg_embedding(msg)
        # to debug:
        # embed_msg = torch.zeros_like(embed_msg)
        embed_msg += self.pos_msg_embedding
        roleset, properties = receiver_inputs
        role_emb, obj_emb = self.embedder(roleset, properties)
        #  import pdb; pdb.set_trace()
        x = torch.cat((role_emb.unsqueeze(1), obj_emb), 1)
        if self.distance_reg_coef > 0:
            attn_mask = self.distance_matrix.to(x.device)
        else:
            attn_mask = None
        y = self.tfm(embed_msg.transpose(0, 1), x.transpose(0, 1),
                     src_mask=attn_mask)
        role_pred = self.out_role(y[0])
        bs = msg.size(0)
        obj_pred = self.out_obj(y[1:]).transpose(0, 1).view(bs, 3, 18, 4)
        return role_pred, obj_pred


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
        loss_roles,
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
        self.loss_roles = loss_roles
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

        loss_roles = self.loss_roles(receiver_outputs[0], labels)
        loss_objs = self.loss_objs(
            sender_input,
            message,
            distribs,
            receiver_input,
            receiver_outputs[1],
            labels,
        )
        if self.ada_len_cost_thresh:
            length_loss_coef = loss_objs < self.ada_len_cost_thresh
        else:
            length_loss_coef = 1
        unweighted_length_cost = (1 - message[:, :, 0]).sum(1)
        weighted_length_cost = (self.length_cost * length_loss_coef *
                unweighted_length_cost)
        loss = (
            loss_roles +
            loss_objs +
            weighted_length_cost
        )

        aux = {}
        aux["length"] = L.float()
        aux["loss_roles"] = loss_roles
        aux["loss_objs"] = loss_objs
        aux["roleset"] = sender_input[0].float()
        aux["weighted_length_cost"] = weighted_length_cost
        aux["sender_input_to_send"] = sender_input[2].float()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        # TODO log the rest of recv in and out
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input[1],
            receiver_input=receiver_input[1].detach(),
            labels=labels[1],
            receiver_output=receiver_outputs[1].detach(),
            message=message.detach(),
            message_length=L.float().detach(),
            aux=aux,
        )
        return loss.mean(), interaction
