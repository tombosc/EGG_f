# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass, fields
from .data_readers import Data
from simple_parsing import Serializable
import egg.core as core
from egg.core.baselines import EMABaseline, MeanBaseline

@dataclass
class EGGParameters(Serializable):
    random_seed: int
    batch_size: int
    checkpoint_dir: str
    optimizer: str 
    lr: float
    vocab_size: int
    max_len: int

    @classmethod
    def from_argparse(cls, args):
        """ Assumes that args is a namespace containing all the field names.
        """
        d = [getattr(args, f.name) for f in fields(cls)]
        print(d)
        return cls(*d)

    def get_dict_dirname(self):
        d = self.__dict__.copy()
        del d['random_seed']
        del d['checkpoint_dir']
        return d


#  class PositionalEncoding(nn.Module):
#      """ Stolen from tutorial:
#      https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#      """
#      def __init__(self, d_model, dropout=0.1, max_len=5000):
#          super(PositionalEncoding, self).__init__()
#          self.dropout = nn.Dropout(p=dropout)

#          pe = torch.zeros(max_len, d_model)
#          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#          div_term = torch.exp(torch.arange(0, d_model, 2).float() *
#                  (-torch.log(torch.Tensor([10000.0])) / d_model))
#          pe[:, 0::2] = torch.sin(position * div_term)
#          pe[:, 1::2] = torch.cos(position * div_term)
#          pe = pe.unsqueeze(0).transpose(0, 1)
#          self.register_buffer('pe', pe)

#      def forward(self, x):
#          x = x + self.pe[:x.size(0), :]
#          return self.dropout(x)

class SharedSubtractEncoder(nn.Module):
    """ For each object, embed its features (differently for each feature
    position), then average these to obtain a matrix (n_feat, dim_embed) M.
    Returns a new matrix $M'_i = (M_i - sum 1/j M_j)$.
    Such an encoder can be shared between sender and receiver and function as 
    their perceptual systems.
    """
    def __init__(self, n_features, dim_embed, max_value):
        super().__init__()
        self.padding_value = 0
        # since 0 is a padding value, all the actual features go from 1 to
        # max_value+1 (not included). For each feature, the embedding will be
        # different.
        self.embeddings = nn.Embedding(1 + n_features * max_value, dim_embed)
        self.n_features = n_features
        self.embed_vector = (torch.arange(n_features) * max_value).view(1, 1, n_features)

    def embed(self, x):
        device = x.device
        embedded = self.embeddings(x + self.embed_vector.to(device))
        return embedded.mean(2)

    def forward(self, x, first_is_target):
        """ if first_is_target=True, return the encoding of the first object
        only. Else, return the entire matrix.
        """
        bs, max_L, n_features = x.size()
        # create mask set to 1 where iff x[i, j] > n[i] elsewhere 0
        mask = torch.all((x == self.padding_value), dim=2)
        x = self.embed(x)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        # Subtract encoder
        #  n_objects = (~mask).long().sum(1, keepdim=True) # (bs,)
        #  mean_row = x.sum(1) / n_objects
        #  x = x - mean_row.unsqueeze(1) + (1 / n_objects.unsqueeze(2)) * x
        #  # TODO integrate info of n somehow?
        if first_is_target:
            return x[:, 0]
        else:
            return x

@dataclass
class Hyperparameters(Serializable):
    seed: int = 0
    embed_dim: int = 30
    validation_batch_size: int = 0
    length_coef: float = 0.
    log_length: bool = False
    sender_entropy_coef: float = 0.
    receiver_hidden: int = 30
    sender_type: str = 'simple'  # 'simple' or 'tfm'
    # tfm specific
    n_heads: int = 4
    n_layers: int = 2

    def __post_init__(self):
        assert(self.embed_dim > 0)
        assert(self.receiver_hidden > 0)

    def get_dict_dirname(self):
        d = self.__dict__.copy()
        if d['sender_type'] == 'simple':
            for u in ['n_heads', 'n_layers']:
                del d[u]
        del d['validation_batch_size']
        return d


class SimpleSender(nn.Module):
    """ Encode the first row of the matrix. 
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x, first_is_target=True)

class TransformerSender(nn.Module):
    """ Pragmatic: the target is "contextualized" before being encoded.
    """
    def __init__(self, encoder, embed_dim, max_objects, n_heads, n_layers):
        super().__init__()
        self.encoder = encoder
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim*4,
        )
        layer_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=layer_norm,
        )
        # target_embed will mark the target (the first row of sender_input)
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        embedded = self.encoder(x, first_is_target=False)
        embedded[:, 0] = embedded[:, 0] + self.target_embed
        out = self.transformer_encoder(
            src=embedded.transpose(0, 1),
            src_key_padding_mask=None,
        ).transpose(0, 1) # (bs, lmax, n_feat)
        return out[:, 0]

class DiscriReceiverEmbed(nn.Module):
    """ A basic discriminative receiver, like DiscriReceiver, but which expect
    integer (long) input to embed, not one-hot encoded.
    """
    def __init__(self, n_features, n_hidden, dim_embed, n_embeddings,
            encoder):
        super().__init__()
        self.padding_value = 0
        self.encoder = encoder
        self.fc1 = nn.Linear(dim_embed, n_hidden)

    def forward(self, x, _input):
        # mask: (bs, L) set to False iff all features are padded
        mask = torch.all(_input == self.padding_value, dim=-1)
        _input = self.encoder(_input, first_is_target=False)
        _input = self.fc1(_input).tanh()
        dots = torch.matmul(_input, torch.unsqueeze(x, dim=-1)).squeeze()
        dots = dots.masked_fill(mask, -float('inf'))
        return dots

def create_game(
    core_params: EGGParameters,
    data: Data.Config,
    hp: Hyperparameters,
    loss,
):
    shared_encoder = SharedSubtractEncoder(
        n_features=data.n_features,
        dim_embed=hp.embed_dim,
        max_value=data.max_value,
    )
    if hp.sender_type == 'simple':
        sender = SimpleSender(
            shared_encoder,
        )
    elif hp.sender_type == 'tfm':
        sender = TransformerSender(
            shared_encoder,
            embed_dim=hp.embed_dim,
            max_objects = data.max_distractors + 1,
            n_heads=hp.n_heads,
            n_layers=hp.n_layers,
        )

    else:
        raise ValueError()

    sender = core.RnnSenderReinforce( 
        sender,
        vocab_size=core_params.vocab_size,
        embed_dim=hp.embed_dim,
        #  hidden_size=opts.sender_hidden,
        hidden_size=hp.embed_dim,
        cell='lstm',
        max_len=core_params.max_len,
        condition_concat=True,
        always_sample=True,
    )
    receiver = DiscriReceiverEmbed(
            n_features=data.n_features,
            n_hidden=hp.receiver_hidden,
            dim_embed=hp.embed_dim,
            n_embeddings=data.max_value,
            encoder=shared_encoder,
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        vocab_size=core_params.vocab_size,
        embed_dim=hp.embed_dim,
        hidden_size=hp.receiver_hidden,
        cell='lstm',
    )
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=hp.sender_entropy_coef,
        receiver_entropy_coeff=0.,
        length_cost=hp.length_coef,
        log_length=hp.log_length,
        baseline_type=MeanBaseline,
    )
    return game



#  class PragmaticReceiver(nn.Module):
#      """ A transformer-based pragmatic receiver.
#      """
#      def __init__(self, n_features, n_hidden, dim_embed, n_embeddings,
#              vocab_size, max_msg_len):
#          super().__init__()
#          self.padding_value = 0
#          # broadcastable separator token to place between message and input
#          self.separator = nn.Parameter(torch.randn(1, 1, dim_embed))
#          self.embeddings_input = nn.Embedding(
#              n_embeddings + 1,
#              dim_embed,
#              padding_idx=self.padding_value,
#          )
#          self.embeddings_msg = nn.Embedding(
#              vocab_size + 1,
#              dim_embed,
#              padding_idx=self.padding_value,
#          )
#          # from docs:
#          # [src/tgt/memory]_mask: float: (-inf) masked, 0.0 else.
#          # "ensure that predictions for position i depend only on the unmasked positions j
#          # and are applied identically for each sequence in a batch."
#          # [src/tgt/memory]_key_padding_mask: ByteTensor: True masked
#          # same but "has a separate mask for each sequence in a batch"
#          # we are interested in the key_padding_mask!
#          self.transformer = nn.Transformer(
#              d_model=dim_embed,
#              nhead=4,
#              num_encoder_layers=2,
#              num_decoder_layers=2,
#              dim_feedforward=64,
#              dropout=0.1,
#              activation='relu',
#              custom_encoder=None,
#              custom_decoder=None,
#          )
#          self.fc_out = nn.Linear(dim_embed, 1)

#      def msg_mask(msg):
#          max_k = messages.size(1)
#          zero_mask = messages == 0
#          lengths = (zero_mask.cumsum(dim=1) > 0).sum(dim=1)


#      def forward(self, msg, input=None, lengths=None):
#          """ Lengths is ignored.
#          """
#          #  print("H1", msg.size(), input.size())
#          bs = msg.size(0)
#          msg_mask = self.msg_mask(msg)
#          input_mask = torch.all(input == 0, dim=-1)
#          # embed:
#          msg = self.embeddings_msg(msg)
#          input = self.embeddings_input(input)
#          # (bs, L_max, n_feat, embed_dim) -> (bs, L_max, embed_dim)
#          input = input.sum(dim=2)  # sum over different features
#          #  input = input.view(input.shape[:2] + (-1,))
#          separator = self.separator.expand(bs, -1, -1)
#          msg_and_sep = torch.cat((msg, separator), dim=1).transpose(0, 1)
#          msg_and_sep = self.positional_msg(msg_and_sep).transpose(0, 1)
#          zeros = torch.zeros((bs, 1), device=input.device, dtype=bool)
#          x = torch.cat((msg_and_sep, input), dim=1)
#          mask = torch.cat((msg_mask, zeros, input_mask), dim=1)
#          #  print("H2", x.size(), mask.size(), input.size(), input_mask.size())
#          agent_output = self.transformer(
#              src=x.transpose(0, 1),
#              tgt=input.transpose(0, 1),
#              src_key_padding_mask=mask,
#              tgt_key_padding_mask=input_mask,
#          ).transpose(0,1)
#          agent_output = self.fc_out(agent_output).squeeze(-1)
#          agent_output = agent_output.masked_fill(input_mask, -float('inf'))
#          logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
#          entropy = logits
#          return agent_output, logits, entropy

class FCAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_weights = nn.Linear(dim * 2, 1)
        self.attn_combine = nn.Linear(dim * 2, dim)

    def forward(self, q, M, mask):
        """ Shapes:
            * q=(bs, d)
            * M=(bs, d', d)
            * mask=(bs, d')
        Softmax over dimension 1 of M.
        """
        #  print("DBG Att q={}, M={}".format(q.size(), M.size()))
        L_max = M.size(1)
        expanded_q = q.unsqueeze(1).expand(-1, L_max, -1) # (bs, d', d)
        cat = torch.cat((M, expanded_q), 2) # (bs, d', 2d)
        attn_weights_unnorm = self.attn_weights(cat) # (bs, d', 1)
        #  print("S", mask.size(), attn_weights_unnorm.size())
        attn_weights_unnorm = attn_weights_unnorm.masked_fill(mask.unsqueeze(2),
                -float('inf'))
        attn_weights = F.softmax(attn_weights_unnorm, dim=1) # (bs, d', 1)
        attn_applied = torch.bmm(attn_weights.transpose(2, 1), M)
        #  print("DBG", attn_applied.size(), q.unsqueeze(1).size())
        output = torch.cat((attn_applied, q.unsqueeze(1)), 2)
        output = self.attn_combine(output)
        output = F.relu(output)
        #  print("DBG", output.size())
        return output, attn_weights_unnorm

class PragmaticSimpleReceiver(nn.Module):
    def __init__(self, input_encoder, hidden_size, max_msg_len, n_max_objects, vocab_size):
        super().__init__()
        self.input_encoder = input_encoder
        self.embedding_msg = nn.Embedding(vocab_size+1, hidden_size)
        self.msg_encoder = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.attention = FCAttention(hidden_size)
        self.max_msg_len = max_msg_len
        self.fc_out = nn.Linear(hidden_size, n_max_objects)

    def msg_mask(self, messages):
        zero_mask = messages == 0
        lengths = (zero_mask.cumsum(dim=1) > 0).sum(dim=1)

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size, device=device)

    def forward(self, msg, input=None, lengths=None):
        #  print("H1", msg.size(), input.size())
        bs = msg.size(0)
        msg_mask = self.msg_mask(msg)
        #  print(input.size())
        input_mask = torch.all(input == 0, dim=-1)  
        device = input.device
        # embed objects
        #  print("Pre in", input.size())
        input = self.input_encoder(input, first_is_target=False)
        msg_embed = self.embedding_msg(msg)
        h = self.init_hidden(device).expand(bs, -1)
        c = self.init_hidden(device).expand(bs, -1)
        hidden_states = []
        for i in range(self.max_msg_len + 1):
            attn_h, _ = self.attention(h, input, input_mask)
            attn_h = attn_h.squeeze(1)
            #  attn_w = attn_w.squeeze(2)
            hidden_states.append(attn_h)
            #  print("unrolling: h={}, c={}, msg_embed={}".format(
            #      attn_h.size(), c.size(), msg_embed.size(),
            #  ))
            if i == self.max_msg_len:
                break
            h, c = self.msg_encoder(msg_embed[:, i], (attn_h, c))

        hidden_states = torch.stack(hidden_states).transpose(1, 0)
        # (bs, L_max, d)
        # variant 1: use the last hidden state
        lengths = torch.clamp(lengths, max=self.max_msg_len)
        output = hidden_states[torch.arange(bs), lengths]#.relu()
        # variant 2: don't care: simply unroll to max_len and pick last
        #  output = hidden_states[:, -1]
        output = self.fc_out(output)
        output = output.masked_fill(input_mask, -float('inf'))
        #  print("OUT", output.size())
        logits = torch.zeros(output.size(0), device=device)
        entropy = logits
        return output, logits, entropy
