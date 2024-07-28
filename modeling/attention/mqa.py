import torch
import math

from .utils import apply_rope_x

class RopelessMQA(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.wq = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.wkv = torch.nn.Parameter(0.01*torch.randn((d_model, 2 * (d_model//n_heads))))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    def forward(self, x, kv_cache=None, past_length=0):
        # queries, keys, and values
        B, S, D = x.shape
        Q = x @ self.wq.T # B, S, D
        KV = x @ self.wkv # B, S, 2*Dh
        K, V = torch.chunk(KV, 2, -1)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            K = torch.cat([k_cache, K], dim=1)
            V = torch.cat([v_cache, V], dim=1)

        updated_kv_cache = (K, V)        
        
        # split into multiple heads
        dh = D//self.n_heads
        K_expand = K.unsqueeze(2).expand(B, -1, self.n_heads, -1)
        V_expand = V.unsqueeze(2).expand(B, -1, self.n_heads, -1)

        # We do this all at once because the reshape options are super expensive
        q_heads = Q.view(B, S, self.n_heads, dh).transpose(1,2)
        k_heads = K_expand.view(B, -1, self.n_heads, dh).transpose(1,2)
        v_heads = V_expand.view(B, -1, self.n_heads, dh).transpose(1,2)

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

        return x, updated_kv_cache    


class Rope_MQA(torch.nn.Module):

    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads        
        self.wq = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.wkv = torch.nn.Parameter(0.01*torch.randn((d_model, 2 * self.dh)))
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
        # queries, keys, and values
        B, S, D = x.shape
        Q = x @ self.wq.T # B, S, D
        KV = x @ self.wkv # B, S, 2*Dh
        K, V = torch.chunk(KV, 2, -1)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            K = torch.cat([k_cache, K], dim=1)
            V = torch.cat([v_cache, V], dim=1)

        updated_kv_cache = (K, V)        
        
        # split into multiple heads
        K_expand = K.unsqueeze(2).expand(B, -1, self.n_heads, -1)
        V_expand = V.unsqueeze(2).expand(B, -1, self.n_heads, -1)
        q_heads = Q.view(B, S, self.n_heads, self.dh).transpose(1,2)
        k_heads = K_expand.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        v_heads = V_expand.view(B, -1, self.n_heads, self.dh).transpose(1,2)

        S_full = k_heads.size(2)        

        ## Apply RoPE        
        cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.dh//2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.dh//2].repeat(1, 1, 1, 2)
        q_heads = apply_rope_x(q_heads, cos_q, sin_q)
        
        cos_k = self.cos_cached[:, :, :S_full, :self.dh//2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.dh//2].repeat(1, 1, 1, 2)
        k_heads = apply_rope_x(k_heads, cos_k, sin_k)

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

        return x, updated_kv_cache    
 
