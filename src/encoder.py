import torch
import torch.nn as nn
import numpy as np
import constants as C
import utils
from sublayer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, n_vocab, max_seq_len, d_embed, n_layers, n_head, d_K, d_V, d_input, d_inner, dropout_rate=0.1):
        super().__init__()

        n_position = max_seq_len + 1

        self.embed = nn.Embedding(n_vocab, d_embed, padding_idx=C.PAD)
        self.position_enc = nn.Embedding.from_pretrained(
            utils.get_PE_table(n_position, d_embed, padding_idx=0),
            freeze=True
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                EncoderLayer(d_input, d_inner, n_head, d_K, d_V, dropout_rate=dropout_rate)
            )

    def forward(self, inputs, pos, return_att=False):
        self_att_mask = utils.get_att_key_pad_mask(seq_k=inputs, seq_q=inputs)
        non_pad_mask = utils.get_non_pad_mask(inputs)

        att_weights_list = []
        output = self.embed(inputs)
        output = output + self.position_enc(pos)
        for enc_layer in self.layers:
            output, att_weights = enc_layer(output, non_pad_mask=non_pad_mask, mh_att_mask=self_att_mask)
            if return_att:
                att_weights_list.append(att_weights)

        return output