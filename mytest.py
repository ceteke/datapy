from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie, MNISTDataset, FashionDataset, STLDataset
import matplotlib.pyplot as plt


ds = STLDataset(is_ae=True)
ds.process()
print(ds.training_data[0])
print(ds.test_data.shape)
