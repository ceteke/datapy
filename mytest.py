from data.datasets import IMDBDataset, CIFAR10Dataset
import matplotlib.pyplot as plt


ds = IMDBDataset(40000, load_unsup=False)
ds.process()
print(len(ds.training_data))
print(len(ds.training_labels))
ds.get_batches_sequence(32,400)
ds.get_batches_sequence(32,400, train=False)
