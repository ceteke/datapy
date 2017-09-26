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

    @property
    def _download_file_name(self):
        return self.dataset_url.split('/')[-1]

    @property
    def file_name(self):
        return self._download_file_name.replace(self.data_type, '')

    @property
    def dataset_path(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
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

        rand_indxs = np.random.permutation(len(self.training_labels))

        if shuffle:
            for i, rand in enumerate(rand_indxs):
                self.training_data[rand] = self.training_data[i]
                self.training_labels[rand] = self.training_labels[i]

        training_batches = np.array_split(self.training_data, ceil(len(self.training_data)/batch_size))
        if self.training_labels is not None:
            label_batches = np.array_split(self.training_labels, ceil(len(self.training_labels)/batch_size))
            return training_batches, label_batches
        return training_batches