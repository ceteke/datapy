from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie, MNISTDataset, FashionDataset
import matplotlib.pyplot as plt


ds = FashionDataset()
ds.process()
X_test, y_test = ds.get_batches(32, train=True)

for x in X_test:
    print(x.shape)