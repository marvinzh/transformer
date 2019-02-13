import torch
import torch.nn as nn
import numpy as np


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_PE_table(n_position, d_embed, padding_idx=None):
    get_angle = lambda pos,i:pos/ np.power(10000, 2*(i//2) / d_embed)
    
    PEs=[]
    for pos in range(n_position):
        PE=[]
        for i in range(d_embed):
            if i%2==0:
                PE.append(np.sin(get_angle(pos, i)))
            else:
                PE.append(np.cos(get_angle(pos, i)))
            
        PEs.append(PE)
    
    return torch.FloatTensor(PEs)


def get_att_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.shape[1]
    mask = seq_k.eq(PAD)
    mask = mask.unsqueeze(1).expand(-1, len_q, -1)
#     print(seq_q.shape)
#     print(mask.shape)
    return mask

def get_subsequent_mask(seq):
    B, L = seq.shape
    mask = torch.triu(
        torch.ones([L, L], device=seq.device, dtype=torch.uint8),
        diagonal=1
    )
    mask = mask.unsqueeze(0).expand(B,-1,-1)
    return mask
