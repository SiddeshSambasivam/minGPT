import torch
import torch.nn as nn
import numpy as np
from torch.nn import Functional as F
import logging


logger = logging.getLogger(__name__)
# This means that logger names track the package/module hierarchy, 
# and it’s intuitively obvious where events are logged just from the 
# logger name.

class GPTConfig:

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    class __init__(self, vocab_size, block_size, **kwargs):

        self.vocab_size = vocab_size
        self.block_size = block_size

        for k,v in kwargs.items():
            setattr(self, k,v)
            # Syntax : setattr(obj, var, val)
            # Parameters :
            # obj : Object whose which attribute is to be assigned.
            # var : object attribute which has to be assigned.
            # val : value with which variable is to be assigned.

class GPT1Config(GPTConfig):

    n_layer = 12
    n_head = 12
    n_embed = 768

class SelfAttention(nn.module):

    def __init__(self, config):
        super().__init__()
        
        assert config.n_embed % config.n_head == 0

        # Weight Matrices for the query, key and value.
        # These are just abstractions. These makes computations more 
        # parallelizable and total computational complexity per layer.
        self.query = nn.Linear(config.n_embed,config.n_embed) # emb x emb
        self.key = nn.Linear(config.n_embed,config.n_embed) # emb x emb
        self.value = nn.Linear(config.n_embed,config.n_embed) # emb x emb

        #regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        #output projections
        self.proj = nn.Linear(config.n_embed, config.n_embed)

        # We use `register_buffer` to store a stateful part of the 
        # model that is not a parameter, but we want it in the 
        # state_dict.
        self.register_buffer(
            "mask", torch.tril(
                torch.ones(
                    config.block_size, config.block_size
                )
            ).view(1,1,config.block_size,config.block_size)
        )
        self.n_head = config.n_head
    
    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # | B -> Batch 
        # | T -> Time step (sequence len) 
        # | C -> Embedding Dim

        # B x nh x T x hs
        k = self.key(x).view(B,T, self.n_head, C //  self.n_head).transpose(1,2) 
        q = self.query(x).view(B,T, self.n_head, C //  self.n_head).transpose(1,2)
        v = self.value(x).view(B,T, self.n_head, C //  self.n_head).transpose(1,2)

        
