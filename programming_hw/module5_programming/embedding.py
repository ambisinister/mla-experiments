import torch

class CustomEmbedding(torch.nn.Module):

	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((num_embeddings, embedding_dim)))

	def forward(self, x):
		return self.weight[x]
		