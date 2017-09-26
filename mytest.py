from data.datasets import IMDBDataset, CIFAR10Dataset

cds = IMDBDataset()
cds.process()
batches, batch_lens = cds.get_batches_sequence(32, 600)