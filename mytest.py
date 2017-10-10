from data.datasets import IMDBDataset, CIFAR10Dataset, CornellMovie, MNISTDataset
import matplotlib.pyplot as plt


ds = MNISTDataset()
ds.process()
ds.get_sprite('sprite.png')
X_test, y_test = ds.get_batches(32, train=False)

for i, t in enumerate(X_test):
    for b, img in enumerate(t):
        print(ds.label_names[y_test[i][b]])
        print(img.shape)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()

