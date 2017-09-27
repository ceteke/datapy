from data.datasets import IMDBDataset, CIFAR10Dataset
import matplotlib.pyplot as plt

cds = CIFAR10Dataset()
cds.process()
plt.imshow(cds.training_data[0])
plt.show()