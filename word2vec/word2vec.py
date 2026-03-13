import numpy as np
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from vocabulary import Vocabulary

EMBED_DIM = 50
BATCH_SIZE = 10000
WINDOW_SCOPE = 2
K = 10
LR = 0.01
EPOCHS = 10

def generate_batches(sequence: list[int], max_scope: int, K: int, vocab: Vocabulary, batch_size: int = BATCH_SIZE) -> tuple[np.array, np.array, np.array]:
  """
  Generates fixed size batches for training

  sequence: training text sample which is a list of word's ids
  max_scope: number of how far context words can be
  K: number of negative samples
  vocab: instance of class Vocabulary that was initialized on training sample
  batch_size: size of the returned batch (last batch can have smaller size)

  returns: batch that consists of center_words, context_words and negative samples
  """

  batch_centers = []
  batch_contexts = []
  batch_negatives = []

  n = len(sequence)
  for c in range(n):
    center_word = sequence[c]
    start_idx = max(0, c - max_scope)
    end_idx = min(n, c + max_scope + 1)

    context_words = [sequence[o] for o in range(start_idx, end_idx) if o != c]

    if not context_words: 
      continue

    all_negative_samples = np.random.choice(
      vocab.vocab_size,
      size=len(context_words) * K,
      p=vocab.ng_probabilities
    )

    for i, context_word in enumerate(context_words):
      batch_centers.append(center_word)
      batch_contexts.append(context_word)
      batch_negatives.append(all_negative_samples[(i * K) : ((i + 1) * K)])

      if len(batch_centers) == batch_size:
        yield (
          np.array(batch_centers, dtype=np.int32), 
          np.array(batch_contexts, dtype=np.int32), 
          np.array(batch_negatives, dtype=np.int32)
        )

        batch_centers, batch_contexts, batch_negatives = [], [], []

  if batch_centers:
    yield (
      np.array(batch_centers, dtype=np.int32), 
      np.array(batch_contexts, dtype=np.int32), 
      np.array(batch_negatives, dtype=np.int32)
    )

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes sigmoid function (clip used to prevent from overflow)

    x: array of numbers 

    returns: array of values in sigmoid function for each value
    """

    x_clipped = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x_clipped))

def train_step_vectorized(center_batch: np.ndarray, context_batch: np.ndarray, neg_batch: np.ndarray, W: np.ndarray, C: np.ndarray, lr: float) -> float:
  """
  Performs training step that includes computing loss, computing gradient and loss

  center: batch of center_words (ids of center_word)
  context: batch of context_words (ids of contex_word)
  negative_samples: batch of negative samples (list of word's ids)
  W: center weight matrix
  C: context weight matrix
  lr: learning rate

  returns: average batch loss
  """
  v_c = W[center_batch]            # (B, D)
  u_o = C[context_batch]           # (B, D)
  u_k = C[neg_batch]               # (B, K, D)


  z_pos = np.sum(v_c * u_o, axis=1)    # (B,)
  p_pos = sigmoid(z_pos)               # (B,)
  e_pos = p_pos - 1.0                  # (B,)

  v_c_expanded = v_c[:, np.newaxis, :]                # (B, 1, D)
  z_neg = np.sum(u_k * v_c_expanded, axis=2)          # (B, K)
  p_neg = sigmoid(z_neg)                              # (B, K)
  e_neg = p_neg - 0.0                                 # (B, K)

  loss_pos = -np.log(p_pos + 1e-10)                         # (B,)
  loss_neg = -np.sum(np.log(1.0 - p_neg + 1e-10), axis=1)   # (B,)
  batch_loss = np.sum(loss_pos + loss_neg) / len(center_batch)

  grad_u_o = e_pos[:, np.newaxis] * v_c               # (B, D)

  grad_u_k = e_neg[:, :, np.newaxis] * v_c_expanded   # (B, K, D)

  grad_v_c_pos = e_pos[:, np.newaxis] * u_o                       # (B, D)
  grad_v_c_neg = np.sum(e_neg[:, :, np.newaxis] * u_k, axis=1)    # (B, D)
  grad_v_c = grad_v_c_pos + grad_v_c_neg                          # (B, D)

  grad_u_o = np.clip(grad_u_o, -5.0, 5.0)
  grad_u_k = np.clip(grad_u_k, -5.0, 5.0)
  grad_v_c = np.clip(grad_v_c, -5.0, 5.0)

  np.add.at(W, center_batch, -lr * grad_v_c)
  np.add.at(C, context_batch, -lr * grad_u_o)
  np.add.at(C, neg_batch, -lr * grad_u_k)

  return batch_loss

def save_weights(vocab: Vocabulary, W: np.array, C: np.array) -> None:
  embed_dim = W.shape[1]

  with open("saved_weights.txt", "w", encoding="utf-8") as f:
    f.write(f"{len(vocab.word2id)} {embed_dim}\n")

    for word, idx in vocab.word2id.items():
      w_str = " ".join(map(str, W[idx]))
      f.write(f"{word} {w_str}\n")
    
    f.write("\n")

    for word, idx in vocab.word2id.items():
      c_str = " ".join(map(str, C[idx]))
      f.write(f"{word} {c_str}\n")    

dataset = load_dataset("afmck/text8")
raw_text = dataset['train'][0]['text']
words = raw_text.split()

vocab = Vocabulary()
vocab.build_vocab(words)
vocab.init_negative_sampling_distribution()

W = np.random.randn(vocab.vocab_size, EMBED_DIM) * 0.01
C = np.random.randn(vocab.vocab_size, EMBED_DIM) * 0.01

encoded_corpus = vocab.encode(words)
epochs_pbar = tqdm(range(EPOCHS), desc="Epochs")

for e in epochs_pbar:
  sum_loss = 0
  count = 0

  batch_generator = generate_batches(encoded_corpus, WINDOW_SCOPE, K, vocab, BATCH_SIZE)
  for center_batch, context_batch, neg_batch in batch_generator:
    sum_loss += train_step_vectorized(center_batch, context_batch, neg_batch, W, C, LR)
    count += 1

    avg_loss = sum_loss / count

    epochs_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
  

save_weights(vocab, W, C)
