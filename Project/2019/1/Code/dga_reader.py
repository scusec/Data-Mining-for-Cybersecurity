from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
from random import shuffle

import numpy as np
import pickle


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def change(self, tokens):
        res = ""
        for token in tokens:
            res += str(self.token(token))
        return res

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_word_length):

    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    actual_max_word_length = 0
    char_tokens = collections.defaultdict(list)

    for fname in ['train']:
        print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) > max_word_length:
                    continue
                # line += '*'
                # line = line.split(".")[0]
                char_array = [char_vocab.feed(c) for c in line]
                char_tokens[fname].append(char_array)

                actual_max_word_length = max(actual_max_word_length, len(char_array))

    print('actual longest token length is:', actual_max_word_length)
    print('size of char vocabulary:', char_vocab.size)
    assert actual_max_word_length <= max_word_length


    # now we know the sizes, create tensors
    char_tensors = {}
    char_lens = {}
    for fname in ['train']:
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)
        char_lens[fname] = np.zeros([len(char_tokens[fname])], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname][i, :len(char_array)] = char_array
            char_lens[fname][i] = len(char_array)

    return char_vocab, char_tensors, char_lens, actual_max_word_length


class DataReader:

    def __init__(self, char_tensor, char_lens, batch_size):

        max_word_length = char_tensor.shape[1]

        rollup_size = char_tensor.shape[0] // batch_size * batch_size
        char_tensor = char_tensor[: rollup_size]
        char_lens = char_lens[: rollup_size]
        self.indexes = list(range(rollup_size // batch_size))
        shuffle(self.indexes)
        # round down length to whole number of slices
        x_batches = char_tensor.reshape([batch_size, -1, max_word_length])
        y_batches = char_lens.reshape([batch_size, -1])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2))
        y_batches = np.transpose(y_batches, axes=(1, 0))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        self.batch_size = batch_size
        self.length = len(self._x_batches)

    def shuf(self):
        shuffle(self.indexes)

    def iter(self):
        for i in self.indexes:
            yield self._x_batches[i], self._y_batches[i]


if __name__ == '__main__':

    _, ct, cl, _ = load_data('dga_data', 65)
    print(ct.keys())

    count = 0
    for x, y in DataReader(ct['train'], cl['train'], 35).iter():
        count += 1
        print(y)
        if count > 0:
            break
