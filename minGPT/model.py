import torch
import torch.nn as nn
import numpy as np
from torch.nn import Functional as F
import logging
import math

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

        # How does tensor multiplication works? Like how to check 
        # if two tensors are compatible for tensor multiplication
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B,nh,T,hs) => (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.resid_drop(self.proj(y))
        return y 

class TransformerBlock(nn.module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.GELU()
            nn.Linear(4*config.n_embed, config.n_embed)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class minGPT(nn.module):

    def __init__(self, config):
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embed)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(config) for _ in range(config.n_layer)
            ]
        )

        self.ln_f =  nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logging.info(
            "number of parameters: %e", sum(
                p.numel() for p in self.parameters()
            )
        )
    
    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module (nn.Linear, nn.Embedding)):
            module.weight.data.normal(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def configure_optimizers(self, train_config):
        pass

    def forward(self, idx, targets=None):
        b, t = idx.size()
        # t -> len of seq
        # b -> Batch Size
        assert t <= self.block_size, "exhausted the block size"

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:,:t,:]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.view(-1), target.view(-1))
            )

        return logits, loss
        
