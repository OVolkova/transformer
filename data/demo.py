import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class NumbersRule(Dataset):
    """
    Generate a dataset of examples of the form:
    input: numbers from 1 to N
    output: bos + (reversed input divided by 2 with skipped odd numbers) + eos

    """

    def __init__(self, split, max_length=10, input_size=50):
        assert split in {"train", "test"}
        self.split = split
        self.length = max_length
        self.input_size = input_size

    def __len__(self):
        return 10000  # ...

    def get_input_vocab_size(self):
        return self.input_size

    def get_output_vocab_size(self):
        return self.input_size // 2 + 2  # +2 for <sos> and <eos>

    def get_block_size(self):
        return self.length

    def __getitem__(self, idx):
        n = np.random.randint(1, self.length + 1)
        while True:
            src = torch.randint(1, self.input_size, size=(n,), dtype=torch.long)
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(src.tolist())) % 4
            inp_split = (
                "test" if h == 0 else "train"
            )  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok
        tgt = torch.tensor(
            [self.input_size // 2]
            + [i // 2 for i in torch.flip(src, dims=(0,)) if i % 2 == 0]
            + [self.input_size // 2 + 1],
            dtype=torch.long,
        )
        tgt_len = tgt.size(0)
        tgt = torch.cat((tgt, torch.zeros(self.length + 2 - tgt_len).long()), dim=0)
        src = torch.cat((src, torch.zeros(self.length - n).long()), dim=0)
        padding_mask = torch.cat(
            (torch.ones(n).long(), torch.zeros(self.length - n).long()), dim=0
        )
        return src, tgt, padding_mask, tgt_len


if __name__ == "__main__":
    train = NumbersRule("train")
    test = NumbersRule("test")
    for k in range(10):
        print(train[k])
    for k in range(10):
        print(test[k])

    from torch.utils.data import DataLoader

    loader = DataLoader(train, batch_size=4, shuffle=True)
    for batch in loader:
        print(batch)
        break
