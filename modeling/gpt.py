import torch
import math

from modeling.attention.mha import MHA, Rope_MHA, Decoupled_Rope_MHA
from modeling.attention.mqa import RopelessMQA, Rope_MQA
from modeling.attention.mla import RopelessMLA_Uncompressed, RopelessMLA, MLA
from modeling.layers.customlayers import CustomLinear, CustomEmbedding

class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, use_mla=True, use_mqa=False,
                 cache_compress=True, use_rope=False, use_decoupled=False):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((d_model,))
        if use_mla:
            print("using Multi-head Latent Attention")
            if not cache_compress:
                print("using regular KV Cache")
                self.mha = RopelessMLA_Uncompressed(d_model, n_heads)
            else:
                if use_rope:
                    print("using RoPE")
                    self.mha = MLA(d_model, n_heads)
                else:
                    self.mha = RopelessMLA(d_model, n_heads)
        elif use_mqa:
            print("using Multi-Query Attention")
            if use_rope:
                print("using RoPE")
                self.mha = Rope_MQA(d_model, n_heads)
            else:
                self.mha = RopelessMQA(d_model, n_heads)
        else:
            if use_rope:
                if use_decoupled:
                    print("using decoupled RoPE")
                    self.mha = Decoupled_Rope_MHA(d_model, n_heads)
                else:
                    print("using RoPE")
                    self.mha = Rope_MHA(d_model, n_heads)
            else:
                self.mha = MHA(d_model, n_heads)
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

    def __init__(self, d_model, n_heads, layers, vocab_size,
                 max_seq_len, use_mla=False, use_mqa=False,
                 cache_compress=True, use_rope=False,
                 use_decoupled=False):
        super().__init__()
        self.use_rope = use_rope

        self.word_embeddings = CustomEmbedding(vocab_size, d_model)
        if use_rope == False:
            self.position_embeddings = CustomEmbedding(max_seq_len, d_model)

        self.layers = torch.nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads,
                                    use_mla=use_mla, use_mqa=use_mqa,
                                    cache_compress=cache_compress,
                                    use_rope=use_rope,
                                    use_decoupled=use_decoupled)
            for _ in range(layers)
        ])

        self.fc_out = CustomLinear(d_model, vocab_size)

        self.max_seq_len = max_seq_len

    @torch.autocast(device_type="cuda")
    def forward(self, x, kv_cache=None, past_length=0):
        B, S = x.shape

        w_emb = self.word_embeddings(x)
        
        if self.use_rope == False:
            positions = torch.arange(past_length, past_length + S, device=x.device).unsqueeze(0).expand(B, -1)
            p_emb = self.position_embeddings(positions)
            x = w_emb + p_emb
        else:
            x = w_emb

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
