from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie
import matplotlib.pyplot as plt


ds = CornellMovie(10000, 0.1)
ds.process()
X, X_lens = ds.get_batches_sequence(32, 25)
X_test, X_test_lens = ds.get_batches_sequence(32,25,train=False)
print(len(X), len(X_test))
print(ds.is_test_loaded)

