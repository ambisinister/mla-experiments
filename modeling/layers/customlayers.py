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
