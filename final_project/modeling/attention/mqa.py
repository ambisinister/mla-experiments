import torch
import math

from utils import apply_rope_x

class RopelessMQA(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.wq = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.wkv = torch.nn.Parameter(0.01*torch.randn((d_model, 2 * (d_model//n_heads))))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

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

        if added_batch:
            x = x[0]

        return x, updated_kv_cache    


class Rope_MQA(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads        
        self.wq = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.wkv = torch.nn.Parameter(0.01*torch.randn((d_model, 2 * self.dh)))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        ## RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, self.dh, 2) * -(math.log(self.rope_theta) / self.dh))
        pe = torch.zeros(self.max_seq_len, self.dh)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        # we can do self.pe = pe here but it wont save in state dict
        self.register_buffer('pe', pe)

    def forward(self, x, kv_cache=None, past_length=0):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

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

        # We do this all at once because the reshape options are super expensive
        q_heads = Q.view(B, S, self.n_heads, self.dh).transpose(1,2)
        k_heads = K_expand.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        v_heads = V_expand.view(B, -1, self.n_heads, self.dh).transpose(1,2)

        S_full = k_heads.size(2)        

        ## Apply RoPE
        cos_q = self.pe[past_length:past_length+S, None, :self.dh // 2].repeat(1, 2)
        sin_q = self.pe[past_length:past_length+S, None, :self.dh // 2].repeat(1, 2)
        q_heads = apply_rope_x(q_heads, cos_q, sin_q)

        cos_k = self.pe[:S_full, None, :self.dh // 2].repeat(1, 2)
        sin_k = self.pe[:S_full, None, :self.dh // 2].repeat(1, 2)
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

        if added_batch:
            x = x[0]

        return x, updated_kv_cache    
 
