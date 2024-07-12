'''
This will download about 5600 chapters from famous books and put them
in a single text file, one chapter per line. This will be used as our
training corpus.

The final file (data.txt) is around 150MB of text.

Requires the datasets library (pip install datasets).
'''

from datasets import load_dataset

# ds = load_dataset("ubaada/booksum-complete-cleaned", "chapters")["train"]
ds = load_dataset("wikitext", "wikitext-103-v1")["train"]

lines = []
for entry in ds:
	text = entry["text"]
	text = text.replace("\n", " ") # remove newline formatting
	text = " ".join(text.split()) # remove sequences of whitespace
	lines.append(text+"\n")

f = open("data.txt", "w")
f.writelines(lines)
f.close()
