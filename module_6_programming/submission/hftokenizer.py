'''
For module 6 onward we are using huggingface tokenizers in the interest of speed.
These are heavily optimized to tokenize large amounts of text quickly.

This tokenizer is similar to the tokenizer we trained in module 2, except:
- Uses Ġ to denote space (if you look in the outputs from saving)
- Byte-level BPE
- does a better job of preserving whitespace sequences

'''

from transformers import AutoTokenizer

class HFTokenizer():

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
		self.tokenizer.eos_token = "<|endoftext|>"

	def train(self, datafile):
		self.tokenizer = self.tokenizer.train_new_from_iterator(
			open(datafile, "r").readlines(), 
			10000,
			limit_alphabet=500,
		)
		self.tokenizer.save_pretrained("./hftokenizer/")

	def load(self):
		self.tokenizer = AutoTokenizer.from_pretrained("./hftokenizer/")

	def encode(self, string):
		return self.tokenizer(string)["input_ids"]

	def decode(self, list_of_ids):
		return self.tokenizer.decode(list_of_ids)





if __name__ == "__main__":

	tokenizer = HFTokenizer()
	tokenizer.train("./data.txt")
	tokenizer.load()

	x = "I want to go eat ice   <eos>cream<eos> yes."
	y = tokenizer.encode(x)
	x2 = tokenizer.decode(y)

	print(x)
	print(y)
	print(x2)