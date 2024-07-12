import torch

'''
Complete this class by instantiating parameters called "self.weight" and "self.bias", and
use them to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		init_w = 0.1*torch.randn((output_size, input_size))
		self.weight = torch.nn.Parameter(init_w)
		init_b = 0.1*torch.randn((output_size,))
		self.bias = torch.nn.Parameter(init_b)

	def forward(self, x):
		x = x @ self.weight.T
		x += self.bias
		return x
		
if __name__ == '__main__':
	## Sanity Checks
	# Test input, batch 64, 1024 features
	inp = torch.randn(64, 1024)

	# Two layers, one outputs 1024, one outputs 2
	lay1 = CustomLinear(1024, 1024)
	lay2 = CustomLinear(1024, 2)

	# Layers have right shape
	print(inp.size()) # (64, 1024)
	print(lay1(inp).size()) # (64, 1024)
	print(inp.size()) # (64, 1024)
	print(lay2(inp).size()) # (64, 2)

	# Layers can be chained
	print(lay2(lay1(inp)).size()) # (64, 2)

	# confirm bias does get added
	inp_w = inp @ lay1.weight.T
	print(torch.all(inp_w == lay1(inp))) # False
	inp_wb = inp_w + lay1.bias
	print(torch.all(inp_wb == lay1(inp))) # True