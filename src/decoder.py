import torch
import torch.nn as nn
import numpy as np
import constants as C
import utils
from sublayer import DecoderLayer

class TransformerDecoder(nn.Module):
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
                DecoderLayer(d_input, d_inner, n_head, d_K, d_V, dropout_rate=dropout_rate)
            )

        self.proj = nn.Linear(d_input, n_vocab, bias=False)

    def forward(self, inputs, pos, src_inputs, enc_outputs, return_att=False):
        non_pad_mask = utils.get_non_pad_mask(inputs)

        self_attn_mask_subseq = utils.get_subsequent_mask(inputs)
        self_attn_mask_keypad = utils.get_att_key_pad_mask(seq_k=inputs, seq_q=inputs)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = utils.get_att_key_pad_mask(seq_k=src_inputs, seq_q=inputs)

        self_att_list = []
        enc_att_list = []
        output = self.embed(inputs) + self.position_enc(pos)
        for layer in self.layers:
            output, self_att, enc_att = layer(output,
                                              enc_outputs,
                                              non_pad_mask=non_pad_mask,
                                              input_att_mask=self_attn_mask,
                                              enc_att_mask=dec_enc_attn_mask,
                                             )            
            if return_att:
                self_att_list.append(self_att)
                enc_att_list.append(enc_att)

        output = self.proj(output)
        return output