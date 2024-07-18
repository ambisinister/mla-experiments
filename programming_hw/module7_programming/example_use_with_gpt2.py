import numpy as np
import torch
from tqdm import tqdm

'''
This script uses the Sampler class to sample text from GPT2-small.

You do not need to utilize this script for the assignment, but it may be
helpful or informative to see your Sampler applied to a real model.

Note this will download about 550MB of parameter data so you can run gpt2.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from sampler import Sampler

samp = Sampler(top_k=1)

# download gpt2 and the associated tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.eval()

# some text
intial_text = "Thomas Jefferson was the"
token_ids = tokenizer.encode(intial_text, return_tensors='pt')[0]
print(token_ids)

# generate N more tokens. We are not using kv cache or anything smart.
# This may be pretty slow.
for i in tqdm(range(50)):

	# pass tokens through the model to get logits
	output = model(token_ids)["logits"][-1,:]

	# sample from the logits
	token_ids_np = token_ids.data.cpu().numpy()
	tok = samp(output.data.cpu().numpy(), token_ids_np)

	# add the resulting token id to our list
	token_ids_np = np.append(token_ids_np, tok)
	token_ids = torch.from_numpy(token_ids_np)


# print out resulting ids
print(token_ids)

# print out the decoded text
print(tokenizer.decode(token_ids))
