# On Multi-Head Latent Attention

Multi-head Latent Attention (MLA) is a variant of multi-head attention which was introduced in the DeepSeek-V2 paper. There are several variants of multi-head attention whose purpose is primarily to reduce the KV-cache size, which is a memory bottleneck that emerges from scaling large models. These methods, which include Group-Query Attention and Multi-Query Attention, are primarily considered /performance tradeoffs/, i.e. the performance is worse, but you get to scale them much further by reducing the memory overhead.

In comparison, MLA accomplishes this by using a low-rank factorized projection matrix, operating a bit like multi-query attention where instead of repeating a single head several times, you decompress a latent vector to yield a unique, appropriate KV for each particular head. DeepSeek claims this not only helps the memory overhead, but also /improves/ the model rather than suffering for its inclusion. The basic idea is as follows:

1. Replace the QKV computation by using low rank factorization to turn one matrix of dim $(in, out)$ to two matrices of $(in, rank)$ and $(rank, out)$.
2. Project the compressed KV latent vector for each head to get the full K and V head corresponding to each Q head.
3. Cache the compressed latent KV vector instead of each of the KV heads, and compute the KV heads on the fly from the latent vector.

There is also an additional component of MLA which outlines /decoupled RoPE/. In this component, they make MLA compatible with RoPE by designating a specific part of each Q and K head to specifically carry RoPE, calculate this directly from the compressed KV vector, and then duplicate this across all Q and K heads to avoid each head redundantly learning the significance of position embeddings. For simplicity's sake, we start with a version which only uses the low-rank factorization, and then add this decoupled RoPE back in later.

## Usage

You need to set up the experiment to do everything in this repo. Once you have everything installed, run the following

```
sh prepare_for_use.sh
```

To run an experiment, we can use the following command

```
python train_model.py
```

You'll need to modify train_model.py to define your model for the experiment. To get training perplexity we can use:

```
python eval_model.py
```

Most of this repo's contributions can be found in ./modeling/attention/, specifically in mla.py. A longform writeup for this can be found [here](https://planetbanatt.net/articles/mla.html).