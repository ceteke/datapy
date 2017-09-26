from datasets import IMDBDataset, CIFAR10Dataset

cds = IMDBDataset()
cds.process()
batches, batch_lens = cds.get_batches_sequence(32, 600)

asd = batches[0][0]
print(len(cds.training_data[0]))
print(batch_lens[0][0])