import torch
import torch.nn as nn
import numpy as np
import hparams as args
from model import Transformer
from encoder import TransformerEncoder
from decoder import TransformerDecoder
import constants as C

if __name__ =="__main__":
    batch_size= 32
    src_len = args.src_max_seq_len-1

    src = torch.randint(10,1000,[batch_size, src_len])
    trg = src.clone()

    src = torch.cat([src, torch.ones([batch_size,1]).type(torch.LongTensor) * C.EOS],dim=1)
    trg = torch.cat([torch.ones([batch_size,1]).type(torch.LongTensor) * C.SOS, trg, torch.ones([batch_size,1]).type(torch.LongTensor) * C.EOS],dim=1)

    transformer = Transformer(
        TransformerEncoder(args.src_n_vocab, args.src_max_seq_len, args.src_d_embed, args.src_layers, args.src_n_heads, args.src_d_k, args.src_d_v, args.src_d_input, args.src_d_inner),
        TransformerDecoder(args.trg_n_vocab, args.trg_max_seq_len, args.trg_d_embed, args.trg_layers, args.trg_n_heads, args.trg_d_k, args.trg_d_v, args.trg_d_input, args.trg_d_inner)
    )

    out = transformer(src, trg)
    print(out.shape)

