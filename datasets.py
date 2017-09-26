import os
import sys
sys.path.append(os.getcwd()) # Import from current path
from dataset_base import BaseDataset
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
import pickle
import operator

class IMDBDataset(BaseDataset):
    dataset_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    def __init__(self):
        super().__init__()
        self.__token2idx = {
            'PAD': 0,
            'EOS': 1,
            'UNK': 2,
        }
        self.__idx2token = {
            0: 'PAD',
            1: 'EOS',
            2: 'UNK'
        }
        self.__token2count = {}
        self.vocab_size = 3
        self.__tokenizer = RegexpTokenizer(r'\w+')
        self.__is_processed = False

    def get_training_data(self):
        assert self.__is_processed, "First call process() method"
        return self.__training_data

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        self._load_training_data()
        self.__is_processed = True

    def _add_line(self, line):
        line_tokens = self._tokenize_line(line)
        for l in line_tokens:
            if l not in self.__token2count:
                self.__token2count[l] = 1
            else:
                self.__token2count[l] += 1

    def _tokenize_line(self, line):
        l_proc = line.strip().replace('<br >', '').replace('<br />', '').lower()
        l_tok = self.__tokenizer.tokenize(l_proc)
        return l_tok

    def line2seq(self, line):
        line_tokens = self._tokenize_line(line)
        l_seq = [self.__token2idx[t] if t in self.__token2idx else 2 for t in line_tokens]

        return l_seq

    def seq2line(self, seq):
        line = ' '.join([self.__idx2token[idx] for idx in seq])
        return line

    def _process_vocab(self):
        '''
        First 50k tokens in the vocabulary
        :return:
        '''
        self.__token2count = dict(sorted(self.__token2count.items(), key=operator.itemgetter(1), reverse=True)[:50000])

        for k, _ in self.__token2count.items():
            self.__token2idx[k] = self.vocab_size
            self.__idx2token[self.vocab_size] = k
            self.vocab_size += 1

        self.__token2count['PAD'] = -1
        self.__token2count['UNK'] = -1
        self.__token2count['EOS'] = -1

        assert (set(self.__token2count.keys()) == set(self.__token2idx.keys())), 'token2count and token2idx does not have same tokens'

    def _load_training_data(self):
        print("Reading training dataset...")

        dataset_path = os.path.join(self.dataset_path, 'aclImdb')

        train_path_neg = '{}/train/neg'.format(dataset_path)
        train_path_pos = '{}/train/pos'.format(dataset_path)
        train_path_unsup = '{}/train/unsup'.format(dataset_path)

        pos_files = [os.path.join(train_path_pos, x) for x in os.listdir(train_path_pos) if 'txt' in x]
        neg_files = [os.path.join(train_path_neg, x) for x in os.listdir(train_path_neg) if 'txt' in x]
        unsup_files = [os.path.join(train_path_unsup, x) for x in os.listdir(train_path_unsup) if 'txt' in x]

        pos_data = []
        neg_data = []
        unsup_data = []

        print("Positive...", flush=True)
        for f_name in pos_files:
            with open(f_name, 'r', encoding='utf-8') as f:
                line = f.readline()
                pos_data.append(line)
                self._add_line(line)

        print("Negative...", flush=True)
        for f_name in neg_files:
            with open(f_name, 'r', encoding='utf-8') as f:
                line = f.readline()
                neg_data.append(line)
                self._add_line(line)

        print("Unsupervised...", flush=True)
        for f_name in unsup_files:
            with open(f_name, 'r', encoding='utf-8') as f:
                line = f.readline()
                unsup_data.append(line)
                self._add_line(line)

        print('Total token count: {}'.format(len(self.__token2count)))
        print("Processing vocabulary...", flush=True)
        self._process_vocab()
        pickle.dump(self.__token2idx, open('token2idx.pkl', 'wb'))

        print("Forming sequeunces...", flush=True)
        pos_data = [self._line2seq(l) for l in pos_data]
        neg_data = [self._line2seq(l) for l in neg_data]
        unsup_data = [self._line2seq(l) for l in unsup_data]

        self.__training_data = {'positive': pos_data, 'negative': neg_data, 'unsupervised': unsup_data}

#class CIFAR10Dataset(BaseDataset):
