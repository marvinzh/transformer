import torch
import torch.nn as nn
import numpy as np
import constants as C

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)

def get_PE_table(n_position, d_embed, padding_idx=None):
    '''Generate posistion encoding table
    
    Arguments:
        n_position {[int]} -- # of sequence length
        d_embed {[int]} -- dimension of embedding vector
    
    Keyword Arguments:
        padding_idx {[type]} -- [description] (default: {None})
    
    Returns:
        [torch.FloatTensor] -- Positional Encoding table with shape (n_position, d_embed)
    '''

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
    
    PEs = np.array(PEs)
    if padding_idx is not None:
        PEs[padding_idx] = 0.
    
    return torch.FloatTensor(PEs)


def get_att_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.shape[1]
    mask = seq_k.eq(C.PAD)
    mask = mask.unsqueeze(1).expand(-1, len_q, -1)
    return mask

def get_subsequent_mask(seq):
    B, L = seq.shape
    mask = torch.triu(
        torch.ones([L, L], device=seq.device, dtype=torch.uint8),
        diagonal=1
    )
    mask = mask.unsqueeze(0).expand(B,-1,-1)
    return mask
