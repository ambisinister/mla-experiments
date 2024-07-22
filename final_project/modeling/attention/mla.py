    
'''
fun reference https://github.com/madsys-dev/deepseekv2-profile/blob/924174cb5dc11fad24bdaad3fd820ebf87506368/workspace/blog/optimizing-mla.md

The open source implementation of this actually just uses regular KV caching which is extremely annoying: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/blob/main/modeling_deepseek.py
'''

import torch
import math

from .utils import apply_rope_x

class RopelessMLA_Uncompressed(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3
        self.dh = d_model // n_heads

        # Q projections
        self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        # KV projections
        self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim)))
        self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim, 2*self.d_model)))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        # output projection
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

        # queries, keys, and values
        B, S, D = x.shape

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq

        # KV Projections
        compressed_kv = x @ self.W_dkv
        compressed_kv = self.kv_layernorm(compressed_kv)
        KV = compressed_kv @ self.W_ukv
        K, V = torch.split(KV, self.d_model, dim=-1)

        # split into multiple heads
        q_heads = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        k_heads = K.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        v_heads = V.view(B, -1, self.n_heads, self.dh).transpose(1,2)

        # KV Cache logic
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_heads = torch.cat([k_cache, k_heads], dim=2)
            v_heads = torch.cat([v_cache, v_heads], dim=2)        
        
        S_full = k_heads.size(2)

        # make attention mask
        mask = torch.ones((S,S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]

        sq_mask = mask == 1

        # attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        x = x @ self.W_o.T

        if added_batch:
            x = x[0]

        return x, (k_heads, v_heads)

class RopelessMLA(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3
        self.dh = d_model // n_heads
        
        # Q projections
        self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        
        # KV projections
        self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim)))
        self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim, 2*self.d_model)))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)
        
        # output projection
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]
            
        B, S, D = x.size()

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            compressed_kv = self.kv_layernorm(compressed_kv)
        else:
            new_kv = x @ self.W_dkv
            new_kv = self.kv_layernorm(new_kv)
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)

        KV = compressed_kv @ self.W_ukv
        K, V = torch.split(KV, self.d_model, dim=-1)

        # split into multiple heads
        q_heads = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        k_heads = K.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        v_heads = V.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        
        S_full = k_heads.size(2)

        # make attention mask
        mask = torch.ones((S,S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]

        sq_mask = mask == 1

        # attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        x = x @ self.W_o.T

        if added_batch:
            x = x[0]

        return x, compressed_kv
    
# TODO: Implement with RoPE
class MLA(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3
        self.q_rope_dim = d_model // 8
        self.k_rope_dim = d_model // 8
        
        ## Q projections
        # Lora
        self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        # Rope
        self.W_qr = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, self.q_rope_dim)))
        
        ## KV projections
        # Lora
        self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim)))
        self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim, 2*self.d_model)))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)
        # Rope
        self.W_kr = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_rope_dim)))
        
        # output projection
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        ## RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, self.dh, 2) * -(math.log(self.rope_theta) / self.dh))
        pe = torch.zeros(self.max_seq_len, self.dh)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)

        self.register_buffer('pe', pe)

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]
            
        B, S, D = x.size()

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq

        # Q Decoupled RoPE
        Q_for_rope = compressed_q @ self.W_qr
        cos_q = self.pe[past_length:past_length+S, None, :self.dh // 2].repeat(1, 2)
        sin_q = self.pe[past_length:past_length+S, None, :self.dh // 2].repeat(1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # Get final Q
        Q = torch.cat([Q, Q_for_rope], dim=-1) # need to figure out dim here

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            compressed_kv = self.kv_layernorm(compressed_kv)
        else:
            new_kv = x @ self.W_dkv
            new_kv = self.kv_layernorm(new_kv)
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)

        KV = compressed_kv @ self.W_ukv
        K, V = torch.split(KV, self.d_model, dim=-1)

        S_full = K.size(1)

        # K Rope
        # I think there's some caching to be implemented here...
        K_for_rope = x @ self.W_kr
        cos_k = self.pe[:S_full, None, :self.dh // 2].repeat(1, 2)
        sin_k = self.pe[:S_full, None, :self.dh // 2].repeat(1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
        K = torch.cat([K, K_for_rope], dim=-1)

        # split into multiple heads
        q_heads = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        k_heads = K.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        v_heads = V.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        
        S_full = k_heads.size(2)

        # make attention mask
        mask = torch.ones((S,S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]

        sq_mask = mask == 1

        # attention
        # Scaling factor needs to be changed -- it is dh + dh_r
        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        x = x @ self.W_o.T

        if added_batch:
            x = x[0]

        return x, compressed_kv
    
    
