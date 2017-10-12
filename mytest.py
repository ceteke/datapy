from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie, MNISTDataset
import matplotlib.pyplot as plt


ds = CIFAR10Dataset()
ds.process()
X_test, y_test = ds.get_batches(32, train=True)

for x in X_test:
    print(x.shape)