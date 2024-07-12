'''
Using this script is optional, It will provide you with a good amount
of raw text to test your tokenizer training.

Requires the datasets library (pip install datasets).
'''

from datasets import load_dataset

ds = load_dataset("ubaada/booksum-complete-cleaned", "chapters")["train"]
lines = []
for entry in ds:
	text = entry["text"]
	text = text.replace("\n", " ") # remove newline formatting
	text = " ".join(text.split()) # remove sequences of whitespace
	lines.append(text+"\n")

	# you can get even more data by increasing this above 100. The highest is ~5600.
	if len(lines) == 100:
		break

f = open("data.txt", "w")
f.writelines(lines)
f.close()
