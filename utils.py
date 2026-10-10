import random
import torch


#create dataset
def build_dataset(words, block_size, stoi):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y


def split_dataset(words, block_size, splits, stoi=None, seed=42):
  """Shuffle words, split them by `splits` (e.g. [0.8, 0.1, 0.1] or [80, 10, 10])
  and return a list of (X, Y) tuples, one per split."""
  if stoi is None:
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0

  words = words[:] # copy so the caller's list isn't shuffled
  random.Random(seed).shuffle(words)

  total = sum(splits)
  datasets = []
  start = 0
  cumulative = 0
  for i, s in enumerate(splits):
    cumulative += s
    # last split takes the remainder so no words are lost to rounding
    end = len(words) if i == len(splits) - 1 else int(cumulative / total * len(words))
    datasets.append(build_dataset(words[start:end], block_size, stoi))
    start = end

  return datasets




def stoi_and_itos(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return (stoi,itos)