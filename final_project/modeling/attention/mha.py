import torch
import math

from .utils import apply_rope

class MHA(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

        # queries, keys, and values
        B, S, D = x.shape
        QKV = x @ self.qkv.T # B, S, 3D

        Q, K, V = torch.chunk(QKV, 3, -1)

        # split into multiple heads
        dh = D//self.n_heads

        # We do this all at once because the reshape options are super expensive
        q_heads = Q.view(B, S, self.n_heads, dh).transpose(1,2)
        k_heads = K.view(B, S, self.n_heads, dh).transpose(1,2)
        v_heads = V.view(B, S, self.n_heads, dh).transpose(1,2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_heads = torch.cat([k_cache, k_heads], dim=2)
            v_heads = torch.cat([v_cache, v_heads], dim=2)

        updated_kv_cache = (k_heads, v_heads)

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
        x = x @ self.wo.T

        if added_batch:
            x = x[0]

        return x, updated_kv_cache

class Rope_MHA(torch.nn.Module):

    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        # RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

        # queries, keys, and values
        B, S, D = x.shape
        QKV = x @ self.qkv.T # B, S, 3D

        Q, K, V = torch.chunk(QKV, 3, -1)

        # split into multiple heads
        q_heads = Q.view(B, S, self.n_heads, self.dh).transpose(1,2)
        k_heads = K.view(B, S, self.n_heads, self.dh).transpose(1,2)
        v_heads = V.view(B, S, self.n_heads, self.dh).transpose(1,2)

        ## Apply RoPE
        cos = self.cos_cached[:, :, past_length:past_length+S, :self.dh//2].repeat(1, 1, 1, 2)
        sin = self.sin_cached[:, :, past_length:past_length+S, :self.dh//2].repeat(1, 1, 1, 2)
        q_heads, k_heads = apply_rope(q_heads, k_heads, cos, sin)

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
        x = x @ self.wo.T

        if added_batch:
            x = x[0]

        return x, (k_heads, v_heads)
    
