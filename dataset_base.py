import abc
import urllib.request as urq
import tarfile
import os

class BaseDataset(object, metaclass=abc.ABCMeta):
    should_delete = True
    data_type = '.tar.gz'
    dataset_url = None

    def __init__(self):
        self.__data_types = ['.tar.gz']

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