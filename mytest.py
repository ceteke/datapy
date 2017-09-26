from datasets import IMDBDataset, CIFAR10Dataset

cds = CIFAR10Dataset()
cds.process()
cds.get_batches(32)
