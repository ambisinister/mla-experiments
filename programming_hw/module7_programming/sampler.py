
import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

    def __init__(
        self,
        top_k=None,
        top_p=None,
        frequency_penalty=1.0,
        presence_penalty=1.0
    ):
        '''
        param top_k : (None or int)
            If specified, only the top k logits should be used during sampling
            If this is specified, top_p should be None

        param top_p : (None or int)
            If specified, only the logits representing the probability mass p should be used during sampling.
            Or, if the top token has mass greater than p, the top token is returned.
            If this is specified, top_k should be None

        If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

        param frequency_penalty : (float)
            A penalty applied to tokens that have previously occured in the sequence. Along with
            presence_penalty, this adjusts the per-token softmax temperature.
            A penalty of 1.0 indicates no change from normal softmax.

        param presence_penalty : (float)
            A penalty applied to tokens IF they have previously occured in the sequence. Along with
            frequency_penalty, this adjusts the per-token softmax temperature.
            A penalty of 1.0 indicates no change from normal softmax.
        '''
        self.topk = top_k
        self.topp = top_p
        self.frq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = 1

    def sample_token(self, raw_unsorted_logits, previous_token_ids):
        '''
        param: raw_unsorted_logits (float numpy array)
            A one dimensional list of logits representing an unnormalized distribution over next tokens
            These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

        param: previous_token_ids (int numpy array)
            A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

        returns: a single token id (integer), sampled according to the specified sampling parameters
        '''

        # Sanity test
        #next_token = torch.argmax(torch.tensor(raw_unsorted_logits), dim=-1).unsqueeze(-1)
        
        logits_tensor = torch.tensor(raw_unsorted_logits)

        # move logits to positive range
        logits_tensor -= torch.min(logits_tensor).item()        

        # does the same as argmax sanity check with provided
        # torch.topk doesnt work for me, does it use abs values??
        if self.topk:
            sorted_logits, sorted_idx = torch.sort(logits_tensor, descending=True)
            logits_tensor[sorted_idx[self.topk:]] = 0
        
        if self.topp:
            sorted_logits, sorted_idx = torch.sort(logits_tensor, descending=True)
            c_probs = torch.cumsum(torch.nn.Softmax()(sorted_logits), dim=-1)

            remove_me = c_probs > self.topp
            remove_idx = [x for x,y in zip(sorted_idx, remove_me) if y == True]

            # always leave at least one token
            if len(remove_idx) == len(sorted_logits):
                remove_idx = remove_idx[1:]
            logits_tensor[remove_idx] = 0

        # presence penalty (on unique set)
        pres_vals = torch.ones_like(logits_tensor)
        pres_vals[list(set(previous_token_ids))] = self.presence_penalty

        # frequency penalty (can be applied multiple times)
        frq_vals = torch.ones_like(logits_tensor)
        tokens, counts = torch.unique(torch.tensor(previous_token_ids), return_counts=True)
        frq_vals[tokens] = self.frq_penalty * counts

        # total penalty vector
        penalty = pres_vals + frq_vals

        # softmax with changes for sampling
        logit_exp = torch.exp(logits_tensor) / (self.temperature * penalty)
        logit_sum = torch.sum(logit_exp) / (self.temperature * penalty)
        smax_logits = logit_exp / logit_sum
        next_token = torch.multinomial(smax_logits, 1)
        return next_token


    # an alternative way to call sample_token(), for convenience
    def __call__(self, raw_unsorted_logits, previous_token_ids):
        return self.sample_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data

    sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

    sequence = [1,2,3,4,5]

    for i in range(10):
        # fake logits for a vocab of size 500
        logits = np.random.randn(500)

        # get next token in sequence
        next_token = sampler(logits, sequence)
        sequence.append(next_token)

    print(sequence)
