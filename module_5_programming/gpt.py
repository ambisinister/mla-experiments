import torch
from linear import CustomLinear
from embedding import CustomEmbedding
from mha import CustomMHA

'''
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use pytorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).

For layer norm, you just need to pass in D-model: self.norm1 = torch.nn.LayerNorm((d_model,))

'''
class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((d_model,))
        self.norm2 = torch.nn.LayerNorm((d_model,))
        self.mha = CustomMHA(d_model, n_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4*d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4*d_model, d_model),
            torch.nn.Dropout() # Dropout lives in here
        )
                

    '''
    param x : (tensor) a tensor of size (batch_size, sequence_length, d_model)
    returns the computed output of the block with the same size.
    '''
    def forward(self, x):
        # Section 1, norm + mha
        post_l1 = self.norm1(x)
        post_attn = self.mha(post_l1)
        res1 = post_attn + x

        # Section 2, norm, mlp, dropout (in self.mlp)
        post_l2 = self.norm2(res1)
        post_mlp = self.mlp(post_l2)
        out = post_mlp + x
               
        return out
                


'''
Create a full GPT model which has two embeddings (token and position),
and then has a series of transformer block instances (layers). Finally, the last 
layer should project outputs to size [vocab_size].
'''
class GPTModel(torch.nn.Module):

    '''
    param d_model : (int) the size of embedding vectors and throughout the model
    param n_heads : (int) the number of attention heads, evenly divides d_model
    param layers : (int) the number of transformer decoder blocks
    param vocab_size : (int) the final output vector size
    param max_seq_len : (int) the longest sequence the model can process.
        This is used to create the position embedding- i.e. the highest possible
        position to embed is max_seq_len
    '''
    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
        super().__init__()
        # Embedding layers
        self.tok_embedding = CustomEmbedding(vocab_size, d_model)
        self.pos_embedding = CustomEmbedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = torch.nn.Sequential(
                *[TransformerDecoderBlock(d_model, n_heads) for _ in range(layers)]
        )

        # Next token prediction
        self.output_layer = torch.nn.Linear(d_model, vocab_size)

                

    '''
    param x : (long tensor) an input of size (batch_size, sequence_length) which is
        filled with token ids

    returns a tensor of size (batch_size, sequence_length, vocab_size), the raw logits for the output
    '''
    def forward(self, x):
        # make word embeddings
        # shape (batch, seq_len, d_model)
        tok = self.tok_embedding(x)

        # make position embeddings
        B,S,D = tok.size()
        input_positions = torch.arange(S)
        
        # shape (1, seq_len, d_model)
        pos = self.pos_embedding(input_positions).unsqueeze(0)

        # make position-encoded word embeddings
        # These can be added from broadcasting, it will be the same as expanding it
        seq_in = tok + pos

        ## making sure the above is true, uncomment if needed later
        # pos2 = self.pos_embedding(input_positions).unsqueeze(0).expand(B,S,D)
        # seq2 = tok + pos2
        # print(torch.all(seq_in == seq2))
        # assert torch.all(seq_in == seq2)

        # pass through transformer blocks
        post_blocks = self.blocks(seq_in)
        
        # asks for logits, so no softmax
        return self.output_layer(post_blocks) 




if __name__ == "__main__":

    # example of building the model and doing a forward pass
    D = 128
    H = 8
    L = 4
    model = GPTModel(D, H, L, 1000, 512)
    B = 32
    S = 48 # this can be less than 512
    x = torch.randint(1000, (B, S))
    y = model(x) # this should give us logits over the vocab for all positions

    # should be size (B, S, 1000)
    print(y.size())
