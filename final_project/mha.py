import torch
import math

'''
fun reference https://github.com/madsys-dev/deepseekv2-profile/blob/924174cb5dc11fad24bdaad3fd820ebf87506368/workspace/blog/optimizing-mla.md
'''

class CustomMHA(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    def forward(self, x):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None,:,:]

        # queries, keys, and values
        B, S, D = x.shape
        QKV = x @ self.W_qkv.T # B, S, 3D
        Q, K, V = torch.chunk(QKV, 3, -1)

        # split into multiple heads
        dh = D//self.n_heads
        q_heads = torch.reshape(Q, (B, S, self.n_heads, dh))
        k_heads = torch.reshape(K, (B, S, self.n_heads, dh))
        v_heads = torch.reshape(V, (B, S, self.n_heads, dh))

        # reshape into (B*h, S, dh) so we isolate sequences for each head
        # you could also keep in four dimensions if you want (B,h,S,dh).
        # We are computing B*H independent attentions in parallel.
        q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, S, dh))
        k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, S, dh))
        v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, S, dh))

        # compute QK^T / sqrt(d)
        k_heads_t = torch.transpose(k_heads, 1, 2)
        qkt = torch.matmul(q_heads, k_heads_t)
        qkt = qkt / math.sqrt(float(dh))

        # --------------------------------------
        # make causal attention mask
        # --------------------------------------

        idx = torch.triu_indices(S, S, offset=1)

        # --------------------------------------
        # apply causal mask
        # --------------------------------------

        qkt[:, idx[0], idx[1]] = float('-inf')

        # the rest of the attention equation
        attn = torch.nn.functional.softmax(qkt, dim=-1)
        x = torch.matmul(attn, v_heads)

        # shmush back into the correct shape
        x = torch.reshape(x, (B, self.n_heads, S, dh))
        x = torch.transpose(x, 1, 2) # B, S, h, dh
        x = torch.reshape(x, (B, S, D))

        # apply projection
        x = x @ self.W_o.T

        if added_batch:
            x = x[0]

        return x

class RopelessMLA(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj_dim = d_model // 4
        self.kv_proj_dim = d_model // 4
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

    def forward(self, x):
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
        q_heads = torch.reshape(Q, (B, S, self.n_heads, self.dh))
        k_heads = torch.reshape(K, (B, S, self.n_heads, self.dh))
        v_heads = torch.reshape(V, (B, S, self.n_heads, self.dh))

        # reshape into (B*h, S, self.dh) so we isolate sequences for each head
        q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, S, self.dh))
        k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, S, self.dh))
        v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, S, self.dh))

        # compute QK^T / sqrt(d)
        k_heads_t = torch.transpose(k_heads, 1, 2)
        qkt = torch.matmul(q_heads, k_heads_t)
        qkt = qkt / math.sqrt(float(self.dh))

        # Causal Mask
        idx = torch.triu_indices(S, S, offset=1)
        qkt[:, idx[0], idx[1]] = float('-inf')

        # the rest of the attention equation
        attn = torch.nn.functional.softmax(qkt, dim=-1)
        x = torch.matmul(attn, v_heads)

        # shmush back into the correct shape
        x = torch.reshape(x, (B, self.n_heads, S, self.dh))
        x = torch.transpose(x, 1, 2) # B, S, h, self.dh
        x = torch.reshape(x, (B, S, D))

        # apply projection
        x = x @ self.W_o.T

        if added_batch:
            x = x[0]

        return x


if __name__ == "__main__":

    import numpy as np

    D = 32
    H = 8
    mha = RopelessMLA(D,H)
    
    # make an input tensor
    B = 2
    S = 3
    x = torch.tensor(np.reshape(np.linspace(-1.0, 0.5, B*S*D), (B,S,D))).to(torch.float32)

    # run
    y1 = mha(x)
    print(y1.shape)
    print(y1)
