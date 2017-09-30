import abc
import urllib.request as urq
import tarfile
import os
import numpy as np
from math import ceil

class BaseDataset(object, metaclass=abc.ABCMeta):
    should_delete = True
    data_type = '.tar.gz'
    dataset_url = None

    def __init__(self):
        self.__data_types = ['.tar.gz']
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

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _load_training_data(self):
        raise NotImplementedError

    def _unarchive_data(self):
        print("Unarchiving dataset {} -> {}...".format(self._download_file_name, self.file_name))
        tar = tarfile.open(self._download_file_name, "r:gz")
        tar.extractall(self.file_name)
        tar.close()

    def _download_dataset(self):
        assert self.data_type in self.__data_types, 'Given data type: {} not in known types'.format(self.data_type)
        print("Downloading dataset: {}...".format(self._download_file_name), flush=True)
        urq.urlretrieve(self.dataset_url, self._download_file_name)
        self._unarchive_data()
        if self.should_delete:
            os.remove(self._download_file_name)

    def get_batches(self, batch_size, shuffle=True):
        '''
        If self.training_label is None, it is assumed that there are no labels for this dataset
        '''
        assert self.training_data is not None, "Load the data first"

        if shuffle:
            self._shuffle_data()

        training_batches = np.array_split(self.training_data, ceil(len(self.training_data)/batch_size))
        if self.training_labels is not None:
            label_batches = np.array_split(self.training_labels, ceil(len(self.training_labels)/batch_size))
            return training_batches, label_batches
        return training_batches

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

        training_batches_unproc = [self.training_data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        training_batches_processed = []
        training_batch_lengths = []

        for unproc_batch in training_batches_unproc:
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
            training_batch_lengths.append(batch_lengths)
            training_batches_processed.append(batch_sequences)

        if self.training_labels is not None and train:
            label_batches = [self.training_labels[i:i + batch_size] for i in range(0, len(self.training_labels), batch_size)]
            return training_batches_processed, training_batch_lengths, label_batches
        if self.test_labels is not None and not train:
            label_batches = [self.test_labels[i:i + batch_size] for i in range(0, len(self.test_labels), batch_size)]
            return training_batches_processed, training_batch_lengths, label_batches
        return training_batches_processed, training_batch_lengths