from mha import *
# your class up here


if __name__ == "__main__":

	import numpy as np

	D = 6
	H = 2
	mha = CustomMHA(D,H)

	# make some fixed weights
	# this just makes a really long 1-D np array and then reshapes it into the size we need
	tensor1 = torch.tensor(np.reshape(np.linspace(-2.0, 1.5, D*D*3), (D*3,D))).to(torch.float32)
	tensor2 = torch.tensor(np.reshape(np.linspace(-1.0, 2.0, D*D), (D,D))).to(torch.float32)
	
	# copy these into our MHA weights, so we don't need to worry about random initializations for testing
	mha.W_qkv.data = tensor1
	mha.W_o.data = tensor2

	print(mha.W_qkv.data)
	print(mha.W_o.data)        

	# make an input tensor
	B = 2
	S = 3
	x = torch.tensor(np.reshape(np.linspace(-1.0, 0.5, B*S*D), (B,S,D))).to(torch.float32)
	print(x)

	# run
	y1 = mha(x)
	print(y1.shape)
	print(y1)

	'''
	Should print out:

	torch.Size([2, 3, 6])
	tensor([[[ 17.2176,   5.5439,  -6.1297, -17.8034, -29.4771, -41.1508],
         [ 17.4543,   5.5927,  -6.2688, -18.1304, -29.9920, -41.8536],
         [ 17.6900,   5.6398,  -6.4105, -18.4607, -30.5110, -42.5612]],

        [[ -1.3639,  -0.1192,   1.1256,   2.3703,   3.6151,   4.8598],
         [ -5.5731,  -1.9685,   1.6361,   5.2407,   8.8453,  12.4499],
         [ -5.6875,  -2.0716,   1.5444,   5.1603,   8.7762,  12.3922]]],
       grad_fn=<UnsafeViewBackward0>)

	'''
