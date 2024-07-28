import torch
from tqdm import tqdm
import numpy as np
from hftokenizer import HFTokenizer


def construct_dataset(data_txt_file, sequence_length=256):
    '''
    data_txt_file : a string path to a text file containing training data, one sample per line
    sequence_length : int, the desired length of each training sequence

    This method should use the trained tokenizer to convert samples to token_ids, and
    then pack them into a training set represented as a 2D array of size (sequences, sequence_length+1).
    The +1 is very important! It lets us compare our model outputs to the sequence shifted by one.

    You can save this training set in whatever format you wish for loading into the training script.
    I recommend using numpy's np.save() method or the pickle module.

    The saved data should be shuffled so we can directly load it and train on it in the training script.
    '''

    # construct tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # get all samples
    print("loading data...")
    f = open(data_txt_file, "r")
    samples = f.readlines()

    # ----------------------------------------
    dataset = []
    packed_sequence = []

    # loop over all the samples
    sample_idx = 0
    i = 0
    while sample_idx < len(samples):
        # just to see it working since it takes a while
        if sample_idx % 10000 == 0:
            print(f"{sample_idx/len(samples)}")

        # tokenize sample
        f = samples[sample_idx]
        t = tokenizer.encode(f)
        
        t_len = len(t)
        current_len = len(packed_sequence)

        # pack to packed sequence until sequence is packed or sample ends
        while current_len < sequence_length and i < t_len:
            packed_sequence.append(t[i])
            i += 1
            current_len += 1

        # if sample ends, go to next sample
        if i == t_len:
            sample_idx += 1
            i = 0
        # if sequence is packed, append + pack new sequence
        # note: we don't reset i here since we need to pick up where we left off in the sample
        # note note: this is an if, not an elif, in case they happen at the same time
        if current_len == sequence_length:
            # eos token
            packed_sequence.append(0)
            dataset.append(packed_sequence)
            packed_sequence = []

    # save as shuffled np array
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    print(np.shape(dataset))

    with open('./packed_data.npy', 'wb') as f:
        np.save(f, dataset)

    ## uncomment to show the actual words in each sample to verify it's working
    ## (do this with a subsample of the dataset not the whole thing)
    #for f in dataset:
    #    print(tokenizer.decode(f))
    #    print("~~~~~~~")


if __name__ == "__main__":
    construct_dataset("./data.txt", 1024)
