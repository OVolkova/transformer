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

    def __init__(self, split, max_length=20, input_size=50, length=10000):
        assert split in {"train", "test"}
        self.split = split
        self.max_length = max_length
        self.input_size = input_size
        self.dataset_length = length

    def __len__(self):
        return self.dataset_length

    def get_vocab_size(self):
        return self.input_size + 2

    def get_block_size(self):
        return self.max_length

    def __getitem__(self, idx):
        n = np.random.randint(1, self.max_length // 2 + 1)
        while True:
            question = torch.randint(1, self.input_size, size=(n,), dtype=torch.long)
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(question.tolist())) % 4
            inp_split = "test" if h == 0 else "train"  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok
        question_length = question.size(0)
        answer = torch.tensor(
            [self.input_size]
            + [i // 2 for i in torch.flip(question, dims=(0,)) if i % 2 == 0]
            + [self.input_size + 1],
            dtype=torch.long,
        )
        answer_length = answer.size(0)
        full_sequence = torch.cat((question, answer), dim=0)
        if len(full_sequence) > self.max_length:
            full_sequence = full_sequence[: self.max_length]
        else:
            full_sequence = torch.cat(
                (
                    full_sequence,
                    torch.zeros(self.max_length - len(full_sequence), dtype=torch.long),
                ),
                dim=0,
            )
        x = full_sequence[:-1].clone()
        y = full_sequence[1:].clone()
        y[: question_length - 1] = -1
        y[question_length + answer_length - 1 :] = -1
        return x, y, question_length, answer_length


if __name__ == "__main__":
    train = NumbersRule("train")
    test = NumbersRule("test")
    for k in range(10):
        print(train[k])
    for k in range(10):
        print(test[k])

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train,
        batch_size=4,
        shuffle=True,
    )
    for batch in loader:
        # x, y, question_length, answer_length = batch
        print(batch)
        break
