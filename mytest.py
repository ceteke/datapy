from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie, MNISTDataset, FashionDataset, STLDataset
import matplotlib.pyplot as plt


ds = MNISTDataset()
ds.process()
print(ds.training_data.shape, ds.training_actual.shape)