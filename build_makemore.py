# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Makemore
#     language: python
#     name: makemore
# ---

# %%
words = open('names.txt', 'r').read().splitlines()

# %%
words[:10]

# %%
len(words)

# %% [markdown]
# A bigram model is one where we are dealing with two characters. Given one character we'll try to predict the next character in the sequence.

# %%
'''
Compute bigrams for all the words
'''
RUN = False
if RUN:
    b = {}
    for w in words:
        chs = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1

# %%
RUN = False
if RUN:
    sorted(b.items(), key = lambda x: -x[1])      # sort bigrams by count descending

# %% [markdown]
# It's better to store this bigram information in a 2-d array where row is the 1st char,
# column is the 2nd char and the value of an entry is how often the 2nd char follows the
# 1st char. We'll use PyTorch.

# %%
import torch

# %%
'''
N is the array we use to store bigram data. 26 chars plus start and end chars ( both '.' ). 2 dims.
'''
N = torch.zeros((27, 27), dtype=torch.int32)

# %%
# List of chars we're dealing with
chars = sorted(list(set(''.join(words))))

# %%
# Create a mapping of chars and their positions
# stoi = {s:i for i,s in enumerate(chars)}
# stoi['<S>'] = 26
# stoi['<E>'] = 27

'''
Instead of having two separate special chars to
denote start and end, we'll have just one special
char '.' and we'll have it as the 1st char.
'''
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# %%
'''
Populate the bigram tensor ( N )
'''
for w in words:
    # chs = ['<S>'] + list(w) + ['<E>']    # We'll change this to have '.' char as both the start and end 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1][ix2] += 1

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')

# %%
N[0]

# %%
# These are probabilities.
p = N[0].float()
p = p / p.sum()
p

# %%
g = torch.Generator().manual_seed(2147483648)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]

# %%
'''
We need to sample from this distribution. For that purpose, we'll use torch.multinomial
'''
RUN = False
if RUN:
    g = torch.Generator().manual_seed(2147483648) # Used for repeatability
    p = torch.rand(3, generator=g)
    p = p / p.sum()
    p

# %%
'''
replacement=True means after drawing a sample, you may put it back.
generator=g gives predictable results when repeated.
The result should mimic the probabilities shown in the tensor above.
Meaning we should see 0s about 60% of the time etc.
'''
RUN = False
if RUN:
    torch.multinomial(p, num_samples=100, replacement=True, generator=g)

# %%
'''
We'll generate tensor of probabilities so that we can use it instead of 
computing the probabilities everytime in the loop.
'''
P = (N+1).float() # Floating point copy of N. Use of N+1 here is model smoothing.
# P = P / P.sum(1, keepdim=True)
P /=  P.sum(1, keepdim=True) # Do this instead of above line because it does the operation in-place without creating extra memory

# %%
g = torch.Generator().manual_seed(2147483648)
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix] # The below two lines are the same as what this is doing
        # p = N[ix].float()
        # p = p / p.sum()
        # p = torch.ones(27) / 27.0  # This line is just to replace the bigram model with a uniform dist and see how the names look
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
'''
The names generated are bad.
'''

# %% [markdown]
# So far we tried to train a bigram language model simply by counting and sampling from
# it using torch.multinomial. Now we want to assess the quality of the model.

# %%
# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)

# %%
'''
We're trying to determine the probability corresponding to each bigram. 
Doing it for the first 3 words.
'''
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f'{ch1}{ch2}: {logprob:.4f}')
# print(f'{log_likelihood=}')
nll = -log_likelihood
# print(f'{nll=}')
print(f'{nll/n}')

# %% [markdown]
# We now have a loss function ( Negative Log Likelihood ). We'll now approach
# this problem using a neural network and try to arrive at a similar place.

# %%
# create the training set of bigrams
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# %%
xs

# %%
ys

# %% [markdown]
# We cannot feed these integers ( which are actually indices of the characters in the 
# bigram ) to a neural network. We have to encode these. One popular way
# of encoding is called one-hot encoding which we'll use for this purpose.
#
# torch.nn.functional.one_hot helps with this.
#
#

# %%
import torch.nn.functional as f
xenc = f.one_hot(xs, num_classes=27).float() # we want to feed floats into neural net. Not ints

# %%
xenc.shape

# %%
xenc

# %%
plt.imshow(xenc)

# %%
xenc.dtype

# %%
W = torch.randn((27, 27)) # torch.randn fills a tensor with random numbers drawn from a normal distribution 
xenc @ W

# %% [markdown]
# The above output (5, 27) tensor contains numbers signifying the firing rate
# of each of the 27 neurons for the 5 members of the input tensor xenc.

# %%
(xenc @ W).exp()

# %%
 
