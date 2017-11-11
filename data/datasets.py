import os
import pickle
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from .dataset_base import SequenceDataset, BaseDataset
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
#plt.style.use('ggplot')
import urllib.request as urllib
import sys, tarfile
from scipy.ndimage.interpolation import rotate

class CornellMovie(SequenceDataset):
    dataset_url = 'http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip'
    data_type = '.zip'

    def __init__(self, max_vocab_size, test_data_size):
        super().__init__(max_vocab_size)
        self.test_data_size = test_data_size

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        return self._load_training_data()

    def form_corpus(self):
        self.process()
        return self.training_data

    def _form_test_data(self, lines):
        test_lines = lines[0:int(len(lines)*self.test_data_size)]
        train_lines = lines[int(len(lines)*self.test_data_size):len(lines)]
        print("Training data size:{}, Test data size:{}".format(len(train_lines), len(test_lines)))
        return train_lines, test_lines

    def _load_training_data(self):
        dataset_path = os.path.join(self.dataset_path, 'cornell movie-dialogs corpus')

        lines_file = os.path.join(dataset_path, 'movie_lines.txt')

        print("Reading dataset")
        file_texts = []
        with open(lines_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for l in lines:
                utterance = l.split(' +++$+++ ')[-1]
                file_texts.append(utterance)
        print("Forming vocabulary...", flush=True)
        train_lines, test_lines = self._form_test_data(file_texts)
        self._add_lines(train_lines)
        self._process_vocab()
        self.training_data = []
        self.test_data = []
        print("Forming sequence data")
        total_miss = 0

        # c = []

        for text in train_lines:
            seq, miss = self.line2seq(text)
            total_miss += miss
            self.training_data.append(seq)

        for text in test_lines:
            seq, miss = self.line2seq(text)
            self.test_data.append(seq)

        # c.sort()
        # fit = norm.pdf(c, np.mean(c), np.std(c))
        # plt.plot(c, fit)
        # plt.hist(c, normed=True)
        # plt.show()

        return total_miss

class IMDBDataset(SequenceDataset):
    dataset_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    def __init__(self, token2idx=None, idx2token=None, max_vocab_size=None, load_unsup=True, corpus_only=False):
        super().__init__(max_vocab_size, token2idx, idx2token)
        self.__is_processed = False
        self.__is_vocab_given = True if token2idx else False
        self.load_unsup = load_unsup
        self.corpus = None
        self.corpus_only = corpus_only

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        corpus = self._load_training_data()
        self._load_test_data()
        if self.corpus_only:
            return corpus

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

        if self.load_unsup or self.corpus_only:
            print("Unsupervised...", flush=True)
            unsup_data = self._load_files(unsup_files)

        training_texts = np.concatenate((pos_data, neg_data))

        if self.load_unsup or self.corpus_only:
            training_texts = np.concatenate((training_texts, unsup_data))
            corpus = training_texts
            if self.corpus_only:
                return corpus
        else:
            self.training_labels = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))

        if not self.__is_vocab_given:
            print("Forming vocabulary...", flush=True)
            self._add_lines(training_texts)
            self._process_vocab()

        print("Forming sequence data...", flush=True)
        # c = []

        train_seq = []
        total_miss = 0
        for i, text in enumerate(training_texts):
            text_seq, miss = self.line2seq(text)
            total_miss += miss
            train_seq.append(text_seq)

        self.training_data = train_seq

        # c.sort()

        # fit = norm.pdf(c, np.mean(c), np.std(c))
        # plt.plot(c, fit)
        # plt.hist(c, normed=True)
        # plt.show()

        # plt.show()

        return total_miss

    def _load_test_data(self):
        print("Reading test dataset...")

        dataset_path = os.path.join(self.dataset_path, 'aclImdb')

        train_path_neg = '{}/test/neg'.format(dataset_path)
        train_path_pos = '{}/test/pos'.format(dataset_path)

        pos_files = [os.path.join(train_path_pos, x) for x in os.listdir(train_path_pos) if 'txt' in x]
        neg_files = [os.path.join(train_path_neg, x) for x in os.listdir(train_path_neg) if 'txt' in x]

        print("Positive...", flush=True)
        pos_data = self._load_files(pos_files)

        print("Negative...", flush=True)
        neg_data = self._load_files(neg_files)

        test_texts = np.concatenate((pos_data, neg_data))
        self.test_labels = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))

        if self.corpus_only:
            self.test_data = test_texts

        print("Forming sequence data...", flush=True)

        test_seq = []
        for i, text in enumerate(test_texts):
            text_seq, miss = self.line2seq(text)
            test_seq.append(text_seq)

        self.test_data = test_seq

class CIFAR10Dataset(BaseDataset):
    dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self):
        super().__init__()

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        self._load_training_data()
        self._load_test_data()
        self.training_data = self.training_data / 255.0
        self.test_data = self.test_data / 255.0
        print("Training shape:", self.training_data.shape)
        print("Test shape:", self.test_data.shape)

        with open(self.dataset_path + '/cifar-10-batches-py/batches.meta', 'rb') as fo:
            self.label_names = pickle.load(fo, encoding='bytes')
            self.label_names = self.label_names[b'label_names']

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

    def _load_test_data(self):
        batch_dir = self.dataset_path + '/cifar-10-batches-py/test_batch'

        with open(batch_dir, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            data = dict[b'data']
            labels = dict[b'labels']

            data_rgb = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))

            self.test_data = data_rgb
            self.test_labels = labels

    def get_metadata(self):
        meta_data_lines = ['idx\tlabel\n']

        for i, _ in enumerate(self.test_data):
            line = '{}\t{}\n'.format(i, self.label_names[self.test_labels[i]])
            meta_data_lines.append(line)

        return meta_data_lines

    def get_sprite(self, path):
        result = Image.new("RGB", (3200, 3200))
        test_images = list((self.test_data*255).astype('uint8'))
        for index, i in enumerate(test_images):
            x = index // 100 * 32
            y = index % 100 * 32

            img = Image.fromarray(i, 'RGB')
            result.paste(img, (y, x, y + 32, x + 32))
        print(self.label_names[self.test_labels[0]])
        result.save(path)


class FashionDataset(BaseDataset):
    def __init__(self, flip=True):
        super().__init__()
        self.tf_dataset = input_data.read_data_sets('data/fashion', one_hot=False, reshape=False,  validation_size=0)
        self.label_names = [i for i in range(0,10)]
        self.degrees = [0, 90, 180, 270]
        self.flip = flip

    def process(self):
        self.form_data()
        if self.flip:
            rotated = []
            actuals = []
            for t in self.training_data:
                for d in self.degrees:
                    rot = rotate(t, d)
                    actual = t
                    rotated.append(rot)
                    actuals.append(actual)
            self.training_data = np.array(rotated)
            self.training_actual = np.array(actuals)

    def form_data(self):
        self.training_data = self.tf_dataset.train.images.copy()
        self.training_labels = self.tf_dataset.train.labels.copy()
        self.test_data = self.tf_dataset.test.images.copy()
        self.test_labels = self.tf_dataset.test.labels.copy()

    def get_metadata(self):
        meta_data_lines = ['idx\tlabel\n']

        for i, _ in enumerate(self.test_data):
            line = '{}\t{}\n'.format(i, self.label_names[self.test_labels[i]])
            meta_data_lines.append(line)

        return meta_data_lines

    def get_sprite(self, path):
        result = Image.new("L", (2800, 2800))
        test_images = list((self.test_data*255).astype('uint8'))
        for index, i in enumerate(test_images):
            x = index // 100 * 28
            y = index % 100 * 28

            img = Image.fromarray(i.squeeze(), 'L')
            result.paste(img, (y, x, y + 28, x + 28))
        print(self.label_names[self.test_labels[0]])
        result.save(path)


class MNISTDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.tf_dataset = input_data.read_data_sets("MNIST_data/", one_hot=False, reshape=False)
        self.label_names = [i for i in range(0,10)]
        self.degrees = [0, 90, 180, 270]

    def process(self):
        self.form_data()
        rotated = []
        actuals = []
        for t in self.training_data:
            for d in self.degrees:
                rot = rotate(t, d)
                actual = t
                rotated.append(rot)
                actuals.append(actual)
        self.training_data = np.array(rotated)
        self.training_actual = np.array(actuals)

    def form_data(self):
        self.training_data = self.tf_dataset.train.images.copy()
        self.training_labels = self.tf_dataset.train.labels.copy()
        self.test_data = self.tf_dataset.test.images.copy()
        self.test_labels = self.tf_dataset.test.labels.copy()

    def get_metadata(self):
        meta_data_lines = ['idx\tlabel\n']

        for i, _ in enumerate(self.test_data):
            line = '{}\t{}\n'.format(i, self.label_names[self.test_labels[i]])
            meta_data_lines.append(line)

        return meta_data_lines

    def get_sprite(self, path):
        result = Image.new("L", (2800, 2800))
        test_images = list((self.test_data*255).astype('uint8'))
        for index, i in enumerate(test_images):
            x = index // 100 * 28
            y = index % 100 * 28

            img = Image.fromarray(i.squeeze(), 'L')
            result.paste(img, (y, x, y + 28, x + 28))
        print(self.label_names[self.test_labels[0]])
        result.save(path)

class STLDataset(BaseDataset):
    def __init__(self, is_ae=True):
        super().__init__()
        self.dataset_url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
        self.label_names = [i for i in range(0,10)]
        self.data_type = '.tar.gz'
        self.is_ae = is_ae

    def process(self):
        data_file = Path(self.dataset_path)
        if not data_file.exists():
            self._download_dataset()
        self.load_dataset()
        self.training_data = self.training_data / 255.0
        self.test_data = self.test_data / 255.0

    def _read_binary(self, path):
        with open(path, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))

            return images

    def _read_labels(self, path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def load_dataset(self):
        train_dir_data = self.dataset_path + '/stl10_binary/train_X.bin'
        train_dir_label = self.dataset_path + '/stl10_binary/train_y.bin'
        test_dir_data = self.dataset_path + '/stl10_binary/test_X.bin'
        test_dir_label = self.dataset_path + '/stl10_binary/test_y.bin'
        unlabeled_dir = self.dataset_path + '/stl10_binary/unlabeled_X.bin'

        train_data = self._read_binary(train_dir_data)
        test_data = self._read_binary(test_dir_data)

        if self.is_ae:
            unlabeled_data = self._read_binary(unlabeled_dir)
            train_data = np.concatenate((train_data, unlabeled_data))
        else:
            train_labels = self._read_labels(train_dir_label)
            test_labels = self._read_labels(test_dir_label)
            self.test_labels = test_labels
            self.training_labels = train_labels

        self.training_data = train_data
        self.test_data = test_data


