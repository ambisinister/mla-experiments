    
'''
fun reference https://github.com/madsys-dev/deepseekv2-profile/blob/924174cb5dc11fad24bdaad3fd820ebf87506368/workspace/blog/optimizing-mla.md

The open source implementation of this actually just uses regular KV caching which is extremely annoying: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/blob/main/modeling_deepseek.py

Make sure you check the config file as well, lots of important info there
https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/blob/main/config.json
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

        return x, compressed_kv
    
class MLA(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2
        
        ## Q projections
        # Lora
        self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, self.d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        
        ## KV projections
        # Lora
        self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim,
                                                          self.d_model + (self.n_heads * self.qk_nope_dim))))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)
        
        # output projection
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        # RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        # we do self.dh here instead of self.qk_rope_dim because its better
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x, kv_cache=None, past_length=0):
        B, S, D = x.size()

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Q Decoupled RoPE
        cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim],
                                                  dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)
            

        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh+self.qk_nope_dim).transpose(1,2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)        

        # K Rope
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1,2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

        # apply position encoding to each head
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        # split into multiple heads
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V # already reshaped before the split

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

        return x, compressed_kv
    
    
