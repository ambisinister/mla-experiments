import torch
import math

class CustomLinear(torch.nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = torch.nn.Parameter(0.01*torch.randn((output_size, input_size)))
        self.bias = torch.nn.Parameter(torch.zeros((output_size,)))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class CustomEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(0.01*torch.randn((num_embeddings, embedding_dim)))

    def forward(self, x):
        return self.weight[x]


class CustomMHA(torch.nn.Module):

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
        q_heads = torch.reshape(Q, (B, S, self.n_heads, dh))
        k_heads = torch.reshape(K, (B, S, self.n_heads, dh))
        v_heads = torch.reshape(V, (B, S, self.n_heads, dh))

        # reshape into (B*h, S, dh) so we isolate sequences for each head
        q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, S, dh))
        k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, S, dh))
        v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, S, dh))

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_heads = torch.cat([k_cache, k_heads], dim=1)
            v_heads = torch.cat([v_cache, v_heads], dim=1)

        updated_kv_cache = (k_heads, v_heads)

        S_full = k_heads.size(1)

        # make attention mask
        mask = torch.ones((S,S_full))
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, :, :]
        mask = mask.to(x.device)

        # attention
        k_heads_t = torch.transpose(k_heads, 1, 2)
        qkt = torch.matmul(q_heads, k_heads_t) / math.sqrt(float(dh))
        qkt = qkt*mask
        qkt[qkt==0] = float('-inf')
        attn = torch.nn.functional.softmax(qkt, dim=-1)
        x = torch.matmul(attn, v_heads)

        # shmush back into the correct shape
        x = torch.reshape(x, (B, self.n_heads, S, dh))
        x = torch.transpose(x, 1, 2) # B, S, h, dh
        x = torch.reshape(x, (B, S, D))

        # apply projection
        x = x @ self.wo.T

        if added_batch:
            x = x[0]

        return x, updated_kv_cache

'''
fun reference https://github.com/madsys-dev/deepseekv2-profile/blob/924174cb5dc11fad24bdaad3fd820ebf87506368/workspace/blog/optimizing-mla.md

The open source implementation of this actually just uses regular KV caching which is extremely annoying: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/blob/main/modeling_deepseek.py
'''
        
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
        q_heads = torch.reshape(Q, (B, -1, self.n_heads, self.dh))
        k_heads = torch.reshape(K, (B, -1, self.n_heads, self.dh))
        v_heads = torch.reshape(V, (B, -1, self.n_heads, self.dh))

        # reshape into (B*h, S, self.dh) so we isolate sequences for each head
        q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, -1, self.dh))
        k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, -1, self.dh))
        v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, -1, self.dh))

        # KV Cache logic
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_heads = torch.cat([k_cache, k_heads], dim=1)
            v_heads = torch.cat([v_cache, v_heads], dim=1)

        S_full = k_heads.size(1)

        # compute QK^T / sqrt(d)
        k_heads_t = torch.transpose(k_heads, 1, 2)
        qkt = torch.matmul(q_heads, k_heads_t)
        qkt = qkt / math.sqrt(float(self.dh))

        # Causal Mask
        mask = torch.ones((S, S_full))
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, :, :]
        mask = mask.to(x.device)
        qkt = qkt*mask
        qkt[qkt==0] = float('-inf')

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

        return x, (k_heads, v_heads)



class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, use_mla=True):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((d_model,))
        if use_mla:
            print("using Multi-head Latent Attention")
            self.mha = RopelessMLA(d_model, n_heads)
        else:
            self.mha = CustomMHA(d_model, n_heads)
        self.norm2 = torch.nn.LayerNorm((d_model,))
        self.fc1 = CustomLinear(d_model, 4*d_model)
        self.act = torch.nn.ReLU()
        self.fc2 = CustomLinear(4*d_model, d_model)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, kv_cache=None, past_length=0):
        normed = self.norm1(x)
        if kv_cache is not None:
            mh_x, kv = self.mha(normed, kv_cache=kv_cache, past_length=past_length)
        else:
            mh_x, kv = self.mha(normed)
        x = x + mh_x
        x = x + self.dropout(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return x, kv
        

class GPTModel(torch.nn.Module):

    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len, use_mla=False):
        super().__init__()

        self.word_embeddings = CustomEmbedding(vocab_size, d_model)
        self.position_embeddings = CustomEmbedding(max_seq_len, d_model)

        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            block = TransformerDecoderBlock(d_model, n_heads, use_mla=use_mla)
            self.layers.append(block)

        self.fc_out = CustomLinear(d_model, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, x, kv_cache=None, past_length=0):
        B, S = x.shape
        positions = torch.arange(past_length, past_length + S).to(torch.long).to(x.device)
        positions = positions[None, :]
        positions = positions.repeat(B, 1)
        
        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(positions)
        x = w_emb + p_emb

        updated_kv_cache = []
        for i, layer in enumerate(self.layers):
            if kv_cache is not None:
                layer_cache = kv_cache[i]
            else:
                layer_cache = None
            x, new_kv = layer(x, kv_cache=layer_cache, past_length=past_length)
            updated_kv_cache.append(new_kv)

        logits = self.fc_out(x)

        return logits, updated_kv_cache


if __name__ == "__main__":

    model = GPTModel(128, 8, 4, 1000, 512)
    B = 32
    S = 48
    x = torch.randint(1000, (B, S))
    y = model(x)
