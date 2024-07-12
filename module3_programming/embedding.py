import torch

'''
Complete this class by instantiating a parameter called "self.weight", and
use it to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # embedding layer is a trained matrix (in, out)
        self.weight = torch.nn.Parameter(torch.randn((num_embeddings, embedding_dim)))

    def forward(self, x):
        # embedding table is just an indexed lookup
        # we don't need to iterate bc of broadcasting
        return self.weight[x]
