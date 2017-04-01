import os
import string
import random
import numpy as np

data_dir = "data"
dataset = "toy"

if not os.path.exists(os.path.join(data_dir, dataset)):
  os.makedirs(os.path.join(data_dir, dataset))
train_path = os.path.join(data_dir, dataset, "train.txt")
test_path = os.path.join(data_dir, dataset, "test.txt")
valid_path = os.path.join(data_dir, dataset, "valid.txt")

words = list(string.ascii_lowercase + "0123456789")

def get_neighbors(words, word, window_size=2):
  if type(word) == str:
    idx = words.index(word)
  elif type(word) == int:
    idx = word
  else:
    raise Exception(" [!] Invalid type for word: %s" % type(word))

  if idx < window_size:
    ans = words[-(window_size - idx):] + words[:idx + window_size + 1]
  elif idx >= len(words) - window_size:
    ans = words[idx-window_size:] + words[:window_size + idx - len(words) + 1]
  else:
    ans = words[idx-window_size:idx+window_size+1]

  for _ in xrange(15):
    if random.random() < 0.1:
      ans.append(random.choice(ans))

  random.shuffle(ans)
  return ans

for path, size in [(train_path, 10000), (test_path, 2000), (valid_path, 2000)]:
  with open(path, "w") as f:
    for idx in xrange(size):
      pivot = np.random.randint(len(words))
      window_size = np.random.randint(1, 4)

      f.write(" ".join(get_neighbors(words, pivot, window_size)) + "\n")
