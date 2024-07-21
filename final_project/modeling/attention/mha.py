import torch
import math

"""
TODO: Implement RoPE https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
"""

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
