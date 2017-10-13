from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie, MNISTDataset, FashionDataset
import matplotlib.pyplot as plt


ds = MNISTDataset()
ds.process()
ds.sample_dataset()

print(ds.training_labels.shape, ds.training_data.shape)