import os
import sys
import operator
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from .dataset_base import BaseDataset

plt.style.use('ggplot')

class IMDBDataset(BaseDataset):
    dataset_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    PAD_IDX = 0
    EOS_IDX = 1
    UNK_IDX = 2

    PAD_TOKEN = 'PAD'
    EOS_TOKEN = 'EOS'
    UNK_TOKEN = 'UNK'

    def __init__(self):
        super().__init__()
        self.__token2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.__idx2token = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.EOS_IDX: self.EOS_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN
        }
        self.__token2count = {}
        self.vocab_size = 3
        self.__tokenizer = RegexpTokenizer(r'\w+')
        self.__is_processed = False

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        self._load_training_data()

    def _add_line(self, line):
        line_tokens = self._tokenize_line(line)
        for l in line_tokens:
            self.__token2count[l] = self.__token2count.get(l, 0) + 1

    def _add_lines(self, lines):
        for l in lines:
            self._add_line(l)

    def _tokenize_line(self, line):
        l_proc = line.strip().replace('<br >', '').replace('<br />', '').lower()
        l_tok = self.__tokenizer.tokenize(l_proc)
        return l_tok

    def line2seq(self, line):
        line_tokens = self._tokenize_line(line)
        l_seq = [self.__token2idx[t] if t in self.__token2idx else self.UNK_IDX for t in line_tokens]

        return np.array(l_seq)

    def seq2line(self, seq):
        line = ' '.join([self.__idx2token[idx] for idx in seq])
        return line

    def _process_vocab(self):
        '''
        First 50k tokens in the vocabulary
        :return:
        '''
        assert len(self.__token2count) > 0, 'Nothing in vocabulary'

        self.__token2count = dict(sorted(self.__token2count.items(), key=operator.itemgetter(1), reverse=True)[:50000])

        for k, _ in self.__token2count.items():
            self.__token2idx[k] = self.vocab_size
            self.__idx2token[self.vocab_size] = k
            self.vocab_size += 1

        self.__token2count[self.PAD_TOKEN] = -1
        self.__token2count[self.EOS_TOKEN] = -1
        self.__token2count[self.UNK_TOKEN] = -1

        assert (set(self.__token2count.keys()) == set(self.__token2idx.keys())), 'token2count and token2idx does not have same tokens'

    def _load_files(self, files):
        file_texts = []
        for f_name in files:
            with open(f_name, 'r', encoding='utf-8') as f:
                line = f.readline()
                file_texts.append(line)
        return np.array(file_texts)

    def _load_training_data(self):
        print("Reading training dataset...")

        dataset_path = os.path.join(self.dataset_path, 'aclImdb')

        train_path_neg = '{}/train/neg'.format(dataset_path)
        train_path_pos = '{}/train/pos'.format(dataset_path)
        train_path_unsup = '{}/train/unsup'.format(dataset_path)

        pos_files = [os.path.join(train_path_pos, x) for x in os.listdir(train_path_pos) if 'txt' in x]
        neg_files = [os.path.join(train_path_neg, x) for x in os.listdir(train_path_neg) if 'txt' in x]
        unsup_files = [os.path.join(train_path_unsup, x) for x in os.listdir(train_path_unsup) if 'txt' in x]

        print("Positive...", flush=True)
        pos_data = self._load_files(pos_files)

        print("Negative...", flush=True)
        neg_data = self._load_files(neg_files)

        print("Unsupervised...", flush=True)
        unsup_data = self._load_files(unsup_files)

        print("Forming vocabulary...", flush=True)
        training_texts = np.concatenate((pos_data, neg_data))
        self._add_lines(training_texts)
        self._process_vocab()

        print("Forming sequence data...", flush=True)
        # c = []

        train_seq = []
        for i, text in enumerate(training_texts):
            text_seq = self.line2seq(text)
            train_seq.append(text_seq)

        self.training_data = train_seq

        # c.sort()

        # fit = norm.pdf(c, np.mean(c), np.std(c))
        # plt.plot(c, fit)
        # plt.hist(c, normed=True)
        # plt.show()

        # plt.show()

        print('Total token count: {}'.format(len(self.__token2count)))

class CIFAR10Dataset(BaseDataset):
    dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self):
        super().__init__()

        with open(self.dataset_path + '/cifar-10-batches-py/batches.meta', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        self._load_training_data()

    def _load_training_data(self):
        for i in range(1,6):
            batch_dir = self.dataset_path + '/cifar-10-batches-py/data_batch_{}'.format(i)

            with open(batch_dir, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                data = dict[b'data']
                labels = dict[b'labels']

                data_rgb = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0,2,3,1)) # R G B channels

                if i == 1:
                    training_data = data_rgb
                    training_labels = labels
                    continue

                training_data = np.concatenate((training_data, data_rgb))
                training_labels = np.append(training_labels, labels)

        self.training_data = training_data
        self.training_labels = training_labels