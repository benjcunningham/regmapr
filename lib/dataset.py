import torch
from torchtext.data import Field, TabularDataset, BucketIterator

class SICKDataset():

    def __init__(self, data, batch_size, train_phases, **kwargs):

        self.data = data
        self.batch_size = batch_size
        self.train_phases = train_phases
        self.cols, self.fields = self.set_fields()
        self.datasets = self.create_datasets()
        self.build_vocab()
        self.iterators = self.create_iterators()


    def create_datasets(self):
        return {split: self.create_dataset(path, self.fields)
                for split, path in self.data.items()}


    def build_vocab(self):
        train = [self.datasets[split] for split in self.train_phases]
        self.cols["SENT"].build_vocab(*train)


    def create_iterators(self):
        return {split: self.create_iterator(dataset, self.batch_size[split])
                for split, dataset in self.datasets.items()}


    @staticmethod
    def set_fields():
        cols = {
            "SENT": Field(sequential=True, lower=True),
            "TARG": Field(sequential=False, use_vocab=False, dtype=torch.float)
        }
        fields = [("s1", cols["SENT"]),
                  ("s2", cols["SENT"]),
                  ("target", cols["TARG"])]
        return cols, fields


    @staticmethod
    def create_dataset(path, fields):
        return TabularDataset(
            path=path,
            format="tsv",
            skip_header=True,
            fields=fields
        )


    @staticmethod
    def create_iterator(dataset, batch_size):
        return BucketIterator(
            dataset,
            batch_size=batch_size,
            sort_key=lambda x: len(x.s1),
            sort_within_batch=False,
            repeat=False
        )
