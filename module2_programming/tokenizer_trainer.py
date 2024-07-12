import json
from collections import defaultdict

'''
Your assignment is to implement BPE in the following method. You can add
classes or other routines if you find them helpful. 

This method should save two output files:
./vocab.txt : a list of the final vocabulary in order, one entry per line
./merges.json : a list of tuples of merges, in order

NOTE: Typically these file extensions are reversed (the vocabulary is a
json file and the merge list is a txt file), but for our purposes this way seems
simplier.

Does not need to return anything.

-------

This should implement a GPT-style tokenizer which prefixes words with a space.
You can assume that the base vocabulary contains all single characters that will occur.
You do NOT need to worry about using a placeholder token in place of a space. 
You do NOT need to worry about special tokens (pad, bos, eos, unk, etc.). We have not covered these yet.

'''

def train_tokenizer(txt_file, vocab_size, base_vocabulary):
    '''
    param : txt_file - a string path to a text file of data, i.e. "./data.txt"
    param : vocab_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default
    '''
    #read data
    with open(txt_file, 'r') as f:
        data = f.read()

    #get word frequencies
    frequencies = defaultdict(int)
    for word in data.split():
        # prepend with a space like in the gpt tokenizer
        frequencies[f' {word}'] += 1

    #make words split up
    split_up = {x: [i for i in x] for x in frequencies.keys()}

    #start getting the pairs
    current_vocab_count = len(base_vocabulary)
    all_merges = []
    
    #not necessary but feels icky to append to "base vocabulary"
    vocab = base_vocabulary.copy()
    
    while current_vocab_count < vocab_size:
        #count adjacent tokens
        pairs = defaultdict(int)
        for w, frequency in frequencies.items():
            s = split_up[w]

            if len(s) == 1:
                continue
            else:
                for i in range(len(s)-1):
                    letter_pair = (s[i], s[i+1])
                    pairs[letter_pair] += frequency

        #get best token to merge and add to vocab
        most_common_pair = max(pairs, key=pairs.get)
        merged_key = ''.join(most_common_pair)
        vocab.append(merged_key)
        current_vocab_count += 1
        left_t = most_common_pair[0]
        right_t = most_common_pair[1]

        #json output
        print(f"Added token {merged_key}, freq: {pairs[most_common_pair]}, size => {current_vocab_count}")
        all_merges.append(most_common_pair)

        #un-split-up these tokens
        for w in frequencies:
            s = split_up[w]

            i = 0
            while len(s) > 1 and i < len(s)-1:
                if s[i] == left_t and s[i+1] == right_t:
                    #python is so great
                    s[i:i+2] = [f'{left_t}{right_t}']
                else:
                    i += 1 
            split_up[w] = s

    #save vocab
    with open('./vocab.txt', 'w') as f:
        f.writelines(f"{x}\n" for x in vocab)

    #save merges
    with open('./merges.json', 'w') as f:
        json.dump(all_merges, f)

from transformers import GPT2Tokenizer, GPT2TokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
        
if __name__ == "__main__":

    # example of using this method.

    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    #train_tokenizer("./data.txt", len(base)+1000, [c for c in base])

    # Initialize tokenizer
    hf_tokenizer = Tokenizer(BPE())
    hf_tokenizer.pre_tokenizer = Whitespace()
    hf_trainer = BpeTrainer(vocab_size=len(base) + 1000, initial_alphabet=[c for c in base])
    print("Initial alphabet check:", 'a' in [c for c in base])  # This should print True
    hf_tokenizer.train(files=["./data.txt"], trainer=hf_trainer)
    #hf_tokenizer.save("./hf_tokenizer.json")
    #hf_tokenizer = Tokenizer.from_file("./hf_tokenizer.json")

    with open('./vocab.txt', 'r') as f:
        custom_vocab = [line.strip() for line in f.readlines()]

    hf_vocab = hf_tokenizer.get_vocab()
    hf_vocab_list = list(hf_vocab.keys())


    yes = 0
    no = 0
    for char in custom_vocab:
        if char in hf_vocab:
            yes += 1
        else:
            no += 1
            print(f"{char} \t is missing from the Hugging Face tokenizer's vocabulary.")

    print(yes/(yes+no))
    # close enough given whitespace nonsense
