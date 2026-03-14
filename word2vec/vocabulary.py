import numpy as np

class Vocabulary:
  def __init__(self):
    self.word2id = {}
    self.id2word = {}
    self.word_counts = {}
    self.vocab_size = 0
    self.unknown_symbol = '<UNK>'
    self.unknown_id = 0

    self.ng_probabilities = None

  def build_vocab(self, text: list[str], min_count: int = 5) -> None:
    """
      Processes the input text and based on it creates dictionaries
    """

    raw_counts = Counter(text)

    self.word2id[self.unknown_symbol] = self.unknown_id
    self.id2word[self.unknown_id] = self.unknown_symbol
    self.word_counts[self.unknown_id] = 0

    for word, count in raw_counts.items():
      if count >= min_count:
        new_id = len(self.word2id)
        self.word2id[word] = new_id
        self.id2word[new_id] = word
        self.word_counts[new_id] = count
      else:
        self.word_counts[self.unknown_id] += count

    self.vocab_size = len(self.id2word)

  def encode(self, words: list[str]) -> list[int]:
    """
      Checks if given words appear in the vocabulary and returns their ids
    """

    return [self.word2id.get(word, self.unknown_id) for word in words]

  def init_negative_sampling_distribution(self, power: float = 0.75) -> None:
    """
      Calculates probabilities for words
    """

    if self.vocab_size == 0:
      raise ValueError(f"Error: build vocabulary first! (invoke build_vocab())")

    counts = np.array([self.word_counts[i] for i in range(self.vocab_size)])
    powered_counts = counts ** power

    self.ng_probabilities = powered_counts / np.sum(powered_counts)
