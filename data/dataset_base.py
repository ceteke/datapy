import abc
import urllib.request as urq
import tarfile
import zipfile
import os
import numpy as np
from math import ceil
import operator
from nltk.tokenize import RegexpTokenizer

class BaseDataset(object, metaclass=abc.ABCMeta):
    should_delete = True
    data_type = '.tar.gz'
    dataset_url = None

    def __init__(self):
        self.__data_types = ['.zip', '.tar.gz']
        self.training_data = None
        self.training_labels = None
        self.test_data = None
        self.test_labels = None

    @property
    def _download_file_name(self):
        return self.dataset_url.split('/')[-1]

    @property
    def file_name(self):
        return self._download_file_name.replace(self.data_type, '')

    @property
    def dataset_path(self):
        current_path = '' # TODO: we should get path
        return os.path.join(current_path, self.file_name)

    @property
    def is_test_loaded(self):
        return self.test_data is not None

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError


    def _unarchive_data(self):
        print("Unarchiving dataset {} -> {}...".format(self._download_file_name, self.file_name))
        if self.data_type == '.tar.gz':
            tar = tarfile.open(self._download_file_name, "r:gz")
            tar.extractall(self.file_name)
            tar.close()
        else:
            zip_ref = zipfile.ZipFile(self._download_file_name, 'r')
            zip_ref.extractall(self.file_name)
            zip_ref.close()

    def _download_dataset(self):
        assert self.data_type in self.__data_types, 'Given data type: {} not in known types'.format(self.data_type)
        print("Downloading dataset: {}...".format(self._download_file_name), flush=True)
        urq.urlretrieve(self.dataset_url, self._download_file_name)
        self._unarchive_data()
        if self.should_delete:
            os.remove(self._download_file_name)

    def get_batches(self, batch_size, train=True, shuffle=True):
        '''
        If self.training_label is None, it is assumed that there are no labels for this dataset
        '''
        assert self.training_data is not None, "Load the data first"

        if shuffle and train:
            self._shuffle_data()

        if train:
            training_batches = np.array_split(self.training_data, len(self.training_data)//batch_size)
            if self.training_labels is not None:
                label_batches = np.array_split(self.training_labels, len(self.training_labels)//batch_size)
                return training_batches, label_batches
            return training_batches
        else:
            test_batches = np.array_split(self.test_data, len(self.test_data) // batch_size)
            if self.test_labels is not None:
                label_batches = np.array_split(self.test_labels, len(self.test_labels) // batch_size)
                return test_batches, label_batches
            return test_batches

    def _shuffle_data(self):
        rand_indxs = np.random.permutation(len(self.training_data))
        for i, rand in enumerate(rand_indxs):
            self.training_data[rand] = self.training_data[i]
            if self.training_labels is not None:
                self.training_labels[rand] = self.training_labels[i]

    def get_batches_sequence(self, batch_size, max_length, pad_token=0, shuffle=True, train=True):
        '''
        self.training_labels must be list of np.array NOT 2D np.array
        '''
        assert self.training_data is not None, "Load the data first"

        if train:
            data = self.training_data

            if shuffle:
                self._shuffle_data()

        else:
            data = self.test_data

        data_batches_unproc = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        data_batches_processed = []
        data_batch_lengths = []

        for unproc_batch in data_batches_unproc:
            max_len_batch = max(len(seq) for seq in unproc_batch)
            batch_lengths = []
            if max_len_batch > max_length:
                max_len_batch = max_length

            for i, sequence in enumerate(unproc_batch): # This is np.array
                if max_len_batch < len(sequence): # There may still values greater than max_len_batch even if the max_len_batch bigger than max_len Ex: [7,8] max_len is 6
                    sequence = sequence[:max_len_batch].reshape(1, -1)
                    batch_lengths.append(max_len_batch)
                else:
                    batch_lengths.append(len(sequence))
                    sequence = np.lib.pad(sequence, (0,max_len_batch-len(sequence)), 'constant', constant_values=pad_token).reshape(1, -1)
                if i == 0:
                    batch_sequences = sequence
                else:
                    batch_sequences = np.concatenate((batch_sequences, sequence))

            data_batch_lengths.append(batch_lengths)
            data_batches_processed.append(batch_sequences)

        if self.training_labels is not None and train:
            label_batches = [self.training_labels[i:i + batch_size] for i in range(0, len(self.training_labels), batch_size)]
            return data_batches_processed, data_batch_lengths, label_batches
        if self.test_labels is not None and not train:
            label_batches = [self.test_labels[i:i + batch_size] for i in range(0, len(self.test_labels), batch_size)]
            return data_batches_processed, data_batch_lengths, label_batches
        return data_batches_processed, data_batch_lengths


class SequenceDataset(BaseDataset):
    PAD_IDX = 0
    EOS_IDX = 1
    UNK_IDX = 2

    PAD_TOKEN = 'PAD'
    EOS_TOKEN = 'EOS'
    UNK_TOKEN = 'UNK'

    def __init__(self, max_vocab_size):
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
        self.max_vocab_size = max_vocab_size

    def _add_line(self, line):
        line_tokens = self._tokenize_line(line)
        for l in line_tokens:
            self.__token2count[l] = self.__token2count.get(l, 0) + 1

    def _add_lines(self, lines):
        for l in lines:
            self._add_line(l)
        print("Total number of tokens: {}".format(len(self.__token2count)))

    def _tokenize_line(self, line):
        l_proc = line.strip().replace('<br >', '').replace('<br />', '').lower()
        l_tok = self.__tokenizer.tokenize(l_proc)
        return l_tok

    def line2seq(self, line):
        line_tokens = self._tokenize_line(line)
        miss_count = 0
        l_seq = []

        for t in line_tokens:
            if t in self.__token2idx:
                l_seq.append(self.__token2idx[t])
            else:
                l_seq.append(self.UNK_IDX)
                miss_count += 1

        return np.array(l_seq), miss_count

    def seq2line(self, seq):
        line = ' '.join([self.__idx2token[idx] for idx in seq])
        return line

    def save_metadata(self):
        lines = ["Word\tIdx\n"]
        for k, v in self.__token2idx.items():
            lines.append("{}\t{}\n".format(k,v))
        with open('metadata.tsv', 'w+') as f:
            f.writelines(lines)

    def _process_vocab(self):
        '''
        First 50k tokens in the vocabulary
        :return:
        '''
        assert len(self.__token2count) > 0, 'Nothing in vocabulary'

        self.__token2count = dict(sorted(self.__token2count.items(), key=operator.itemgetter(1), reverse=True)[:self.max_vocab_size])

        for k, _ in self.__token2count.items():
            self.__token2idx[k] = self.vocab_size
            self.__idx2token[self.vocab_size] = k
            self.vocab_size += 1

        self.__token2count[self.PAD_TOKEN] = -1
        self.__token2count[self.EOS_TOKEN] = -1
        self.__token2count[self.UNK_TOKEN] = -1

        assert (set(self.__token2count.keys()) == set(self.__token2idx.keys())), 'token2count and token2idx does not have same tokens'