import torch
import math

'''
Complete this module such that it includes causal masking.
There are two TODOs to fill in below: 
(1) the creation of the mask and (2) the application of the mask.
After applying the mask you proceed with softmax as usual.
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





if __name__ == "__main__":

    # same test case as last week, but
    # should be a different output now that we have added causal mask

    import numpy as np

    D = 6
    H = 2
    mha = CustomMHA(D,H) # 8 d_model, 2 heads

    # make some fixed weights
    tensor1 = torch.tensor(np.reshape(np.linspace(-2.0, 1.5, D*D*3), (D*3,D))).to(torch.float32)
    tensor2 = torch.tensor(np.reshape(np.linspace(-1.0, 2.0, D*D), (D,D))).to(torch.float32)
    
    # copy these into our MHA weights
    mha.W_qkv.data = tensor1
    mha.W_o.data = tensor2

    # make an input tensor
    B = 2
    S = 3
    x = torch.tensor(np.reshape(np.linspace(-1.0, 0.5, B*S*D), (B,S,D))).to(torch.float32)

    # run
    y1 = mha(x)
    print(y1.shape)
    print(y1)

    '''
    Should print out:

    torch.Size([2, 3, 6])
    tensor([[[ 21.7331,   6.4755,  -8.7821, -24.0397, -39.2973, -54.5549],
         [ 19.6692,   6.0497,  -7.5698, -21.1893, -34.8087, -48.4282],
         [ 17.6900,   5.6398,  -6.4105, -18.4607, -30.5110, -42.5612]],

        [[  2.8558,   0.8462,  -1.1635,  -3.1731,  -5.1827,  -7.1924],
         [ -1.2608,  -0.5160,   0.2287,   0.9735,   1.7182,   2.4630],
         [ -5.6875,  -2.0716,   1.5444,   5.1603,   8.7762,  12.3922]]],
       grad_fn=<UnsafeViewBackward0>)
    '''
