import json
import numpy as np

from embedding import CustomEmbedding
    
'''
This class should be constructed with trained tokenizer data:
vocab_file : a string path to a vocab.txt file
merges_file : a string path to a merges.json file

The class should implement two methods:
encode(string): returns a list of integer ids (tokenized text)
decode(list_of_ids): returns a string re-assembled from token ids

A good sanity check is that decode(encode(x)) should return x.

You may assume that only a single sample is passed in at a time (no batching).
You can add additional methods, classes, etc as you find helpful.
'''

class Tokenizer:
    
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, 'r') as f:
            vocablist = [x[:-1] for x in f.readlines()]

        # tables to go back and forth from vocab to index in O(1)
        self.vocab_to_i = {x: i for i,x in enumerate(vocablist)}
        self.i_to_vocab = {i: x for i,x in enumerate(vocablist)}

        with open(merges_file, 'r') as f:
            self.merges = json.load(f)

        # table to merge two tokens together
        # input is tuple (token_ind_1, token_ind_2)
        # output is index of merged token in vocabulary
        self.merge_tokens = {(self.vocab_to_i[a], self.vocab_to_i[b]):
                              self.vocab_to_i[a+b] for a,b in self.merges}

    def encode(self, string):
        '''
        param string : a string to be encoded
        returns a list of integers (token ids)
        '''

        # tokenize characters first
        encoding = [self.vocab_to_i[s] for s in string]

        # loop until can't merge
        while True:
            merged = False

            # go through merges in order
            for m in self.merges:
                tup = (self.vocab_to_i[m[0]], self.vocab_to_i[m[1]])
                new_encoding = []
                i = 0
                while i < len(encoding)-1:
                    # get adjacent token pair
                    pair = (encoding[i], encoding[i+1])

                    # if this is the merge, merge it
                    if pair == tup:
                        new_encoding.append(self.merge_tokens[pair])
                        #print(f"merging {m}")
                        merged = True
                        i += 2
                    # if not, go to the next pair
                    else:
                        new_encoding.append(encoding[i])
                        i += 1

                # if the last one was not a merge, add the last token
                if i == len(encoding)-1:
                    new_encoding.append(encoding[i])

                # update new encoding
                encoding = new_encoding

            if not merged:
                break

        return encoding


    def decode(self, list_of_integers):
        '''
        param list_of_integers : a list of token ids
        returns a string formed by decoding these ids.
        '''
        decoded = []

        for i in list_of_integers:
            decoded.append(self.i_to_vocab[i])

        return ''.join(decoded)


if __name__ == "__main__":

    # example of using this class
    #emb = CustomEmbedding(5096, 2048)
    
    tok = Tokenizer("./vocab.txt", "./merges.json")
    x = tok.encode("Peter piper picked a peck of pickled peppers.")
    print(x)
    #feats = emb(x)
    #print(feats)
    #print(np.shape(feats))
    print([tok.i_to_vocab[i] for i in x])
    x = tok.decode(x)
    print(x) # should be our original text.
    
    x = tok.encode("I gotta testify")
    print(x)
    #feats = emb(x)
    #print(feats)
    #print(np.shape(feats))
    print([tok.i_to_vocab[i] for i in x])
    x = tok.decode(x)
    print(x) # should be our original text.

    x = tok.encode("Ammon and the beast have been ordained by Nebuchadrezzar")
    print(x)
    #feats = emb(x)
    #print(feats)
    #print(np.shape(feats))
    print([tok.i_to_vocab[i] for i in x])
    x = tok.decode(x)
    print(x) # should be our original text.
