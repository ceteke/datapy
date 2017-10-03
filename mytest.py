from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie
import matplotlib.pyplot as plt


ds = CornellMovie(10000)
ds.process()
X, X_lens = ds.get_batches_sequence(32, 25)

for step in range(len(X)):
    for x in X[step]:
        print(ds.seq2line(x))
