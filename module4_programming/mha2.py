import torch
import math
import numpy as np

'''
Complete this module such that it computes queries, keys, and values,
computes attention, and passes through a final linear operation W_o.

You do NOT need to apply a causal mask (we will do that next week).
If you don't know what that is, don't worry, we will cover it next lecture.

Be careful with your tensor shapes! Print them out and try feeding data through
your model. Make sure it behaves as you would expect.
'''
class CustomMHA(torch.nn.Module):
    '''
    param d_model : (int) the length of vectors used in this model
    param n_heads : (int) the number of attention heads. You can assume that
       this even divides d_model.
    '''
    def __init__(self, d_model, n_heads):
        super().__init__()
	# TODO
	# please name your parameters "self.W_qkv" and "self.W_o" to aid in grading
	# self.W_qkv should have shape (3D, D) or (D, 3D) depending on how you'd like to set it up
	# self.W_o should have shape (D,D)
        
        self.d = d_model
        self.h = n_heads

        # scaling factor for self-attention: must be an integer
        # must be for the size of the head, not the whole  matrix
        self.dh = d_model // n_heads

        # self.heads = [{'W_q': torch.nn.Linear(self.d, self.dh, bias=False),
        #                'W_k': torch.nn.Linear(self.d, self.dh, bias=False),
        #                'W_v': torch.nn.Linear(self.d, self.dh, bias=False)} for _ in range(self.h)]
        
        # self.W_q = torch.nn.Linear(self.d, self.d, bias=False)
        # self.W_k = torch.nn.Linear(self.d, self.d, bias=False)
        # self.W_v = torch.nn.Linear(self.d, self.d, bias=False)

        self.W_qkv = torch.nn.Linear(self.d, 3*self.d, bias=False)

        # projection matrix from concatenated head output
        self.W_o = torch.nn.Linear(self.d, self.d, bias=False)

    # The self-attention operation
    # https://en.wikipedia.org/wiki/Attention_(machine_learning)#Standard_Scaled_Dot-Product_Attention
    def attn(self, q, k, v):
        num =(q@k.transpose(-2, -1))/math.sqrt(self.dh)
        sm = torch.nn.Softmax(dim=-1)(num)
        return sm @ v

    def split_into_heads(self, m):
        return m.chunk(self.h, dim=-1)

    def split_into_qkv(self, m):
        return m.chunk(3, dim=-1)
    
    '''
    param x : (tensor) an input batch, with size (batch_size, sequence_length, d_model)
    returns : a tensor of the same size, which has had MHA computed for each batch entry.
    '''
    def forward(self, x):
        # TODO

        # for head in self.heads:
        #     q = head['W_q'](x)
        #     k = head['W_k'](x)
        #     v = head['W_v'](x)
        #     head_outs.append(self.attn(q, k, v))

        #q = self.W_q(x)
        #k = self.W_k(x)
        #v = self.W_v(x)

        # get q, k, v from initial big matmul
        qkv = self.W_qkv(x)
        (q, k, v) = self.split_into_qkv(qkv)

        # split q, k, v into heads
        q_heads = self.split_into_heads(q)
        k_heads = self.split_into_heads(k)
        v_heads = self.split_into_heads(v)

        # perform attention in each of the heads
        head_outs = []
        for (qh, kh, vh) in zip(q_heads, k_heads, v_heads):
            head_outs.append(self.attn(qh, kh, vh))        

        # concat together and project with W_o
        pre_o = torch.cat(head_outs, dim=-1)
        return self.W_o(pre_o)

if __name__ == "__main__":
    D = 6
    H = 2
    mha = CustomMHA(D,H)

    # make some fixed weights
    # this just makes a really long 1-D np array and then reshapes it into the size we need
    tensor1 = torch.tensor(np.reshape(np.linspace(-2.0, 1.5, D*D*3), (D*3,D))).to(torch.float32)
    tensor2 = torch.tensor(np.reshape(np.linspace(-1.0, 2.0, D*D), (D,D))).to(torch.float32)
    
    # copy these into our MHA weights, so we don't need to worry about random initializations for testing
    mha.W_qkv.weight = torch.nn.Parameter(tensor1)
    mha.W_o.weight = torch.nn.Parameter(tensor2)
        
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
    tensor([[[ 17.2176,   5.5439,  -6.1297, -17.8034, -29.4771, -41.1508],
         [ 17.4543,   5.5927,  -6.2688, -18.1304, -29.9920, -41.8536],
         [ 17.6900,   5.6398,  -6.4105, -18.4607, -30.5110, -42.5612]],

        [[ -1.3639,  -0.1192,   1.1256,   2.3703,   3.6151,   4.8598],
         [ -5.5731,  -1.9685,   1.6361,   5.2407,   8.8453,  12.4499],
         [ -5.6875,  -2.0716,   1.5444,   5.1603,   8.7762,  12.3922]]],
       grad_fn=<UnsafeViewBackward0>)

    '''
